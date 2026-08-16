# =============================================================================
# Quality Assurance Framework for AI Retrieval Systems in Oncology Regulatory Guidance
# =============================================================================

# =============================================================================
# Imports
# =============================================================================

import os
import re
import json
import datetime
import pickle
import hashlib
import fitz
import torch
import faiss
import numpy as np
import pandas as pd
from spacy.lang.en import English
from sentence_transformers import SentenceTransformer
from groq import Groq
from openai import AzureOpenAI, OpenAI
from google import genai   # new SDK (replaces deprecated google.generativeai)
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional

from rapidfuzz import fuzz

from deepeval import evaluate
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase
from deepeval.models.base_model import DeepEvalBaseLLM
from statistics import mean, median, pstdev

# =============================================================================
# Read PDFs and enter model details
# =============================================================================

PDF_PATHS = [
    "Real-Time Oncology Review _ FDA.pdf",
    "UCM609513.pdf",
    "26076956fnl_Expansion-Cohorts-Use-in-First-in-Humans-Clinical-Trials.pdf",
    "2021-629 Prior Therapy Final Guidance for publication July 2022.pdf",
    "cct_ps_draft_guidance_april_2024.pdf",
    "2022-244 Draft_AAinOncology March 2023.pdf"
]

groq_api_key = "INSERT_API_KEY"

GOOGLE_API_KEY = "INSERT_API_KEY"
GEMINI_MODEL   = "gemma-4-31b-it"

AZURE_OPENAI_ENDPOINT    = "INSERT_ENDPOINT"
AZURE_OPENAI_API_KEY     = "INSERT_API_KEY"
AZURE_OPENAI_API_VERSION = "2024-12-01-preview"
AZURE_OPENAI_DEPLOYMENT  = "gpt-5.4-mini"

DEEPSEEK_API_KEY = "INSERT_API_KEY"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL   = "deepseek-chat"

model_name  = AZURE_OPENAI_DEPLOYMENT
judge_model = DEEPSEEK_MODEL

JUDGE_PANEL = [
    ("deepseek", DEEPSEEK_MODEL),
    ("groq",     "llama-3.1-8b-instant")
    # ("gemini", GEMINI_MODEL),
]

chunk_size = 6
min_tokens = 15
top_k      = 10

# -----------------------------------------------------------------------------
# Retrieval-precision controls  [ADDED for tighter, higher-precision highlights]
# -----------------------------------------------------------------------------

RETRIEVAL_SCORE_FLOOR = 0.42
RETRIEVAL_REL_GAP     = 0.72
DOC_DOMINANCE_RATIO   = 0.60
MAX_HIGHLIGHT_CHUNKS  = 6

USE_GENERIC_STATIC_EXPANSION = False

# =============================================================================
# Per-document caching
# =============================================================================

CACHE_DIR     = "vector_cache"
DOC_CACHE_DIR = os.path.join(CACHE_DIR, "docs")
MANIFEST_PATH = os.path.join(CACHE_DIR, "manifest.json")

os.makedirs(DOC_CACHE_DIR, exist_ok=True)

def text_formatter(text: str) -> str:
    text = text.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_ws(text: str) -> str:
    """Canonical whitespace normaliser used for BOTH chunk text and the
    reconstructed full-document text, so exact substring matching works."""
    return re.sub(r"\s+", " ", text.replace("\r", " ").replace("\n", " ")).strip()


EVIDENCE_GROUNDED_THRESHOLD = 90.0
EVIDENCE_PARTIAL_THRESHOLD  = 78.0
# Ignore trivially short "quotes" that would match almost anywhere.
EVIDENCE_MIN_CHARS = 12


def _flatten_with_map(text: str):
    """Collapse runs of whitespace to a single space while keeping a map from
    each flattened-string position back to the original character offset. This
    lets us match on normalised whitespace but report spans in the ORIGINAL
    (un-normalised) text so highlight offsets are exact."""
    out, idx, prev_space = [], [], False
    for i, ch in enumerate(text):
        if ch.isspace():
            if prev_space:
                continue
            out.append(" ")
            idx.append(i)
            prev_space = True
        else:
            out.append(ch)
            idx.append(i)
            prev_space = False
    idx.append(len(text))
    return "".join(out), idx


def align_evidence(evidence: str, haystack: str) -> dict:
    """
    Locate a (near-)verbatim `evidence` quote inside `haystack`.
    Returns {"start": int|None, "end": int|None, "score": float(0-100)}.
    Offsets are into the ORIGINAL haystack string.
    """
    ev = re.sub(r"\s+", " ", (evidence or "").strip())
    if len(ev) < EVIDENCE_MIN_CHARS or not haystack:
        return {"start": None, "end": None, "score": 0.0}

    flat, index_map = _flatten_with_map(haystack)

    # ---- exact (whitespace-insensitive) ------------------------------------
    pos = flat.find(ev)
    if pos != -1:
        start = index_map[pos]
        end   = index_map[min(pos + len(ev), len(index_map) - 1)]
        return {"start": start, "end": end, "score": 100.0}

    # ---- fuzzy sliding alignment -------------------------------------------
    try:
        result = fuzz.partial_ratio_alignment(ev, flat, score_cutoff=EVIDENCE_PARTIAL_THRESHOLD)
    except Exception:
        result = None
    if result is None:
        return {"start": None, "end": None, "score": 0.0}

    s = index_map[min(result.dest_start, len(index_map) - 1)]
    e = index_map[min(result.dest_end,   len(index_map) - 1)]
    return {"start": s, "end": e, "score": round(float(result.score), 1)}


def evidence_grounding_label(score: float) -> str:
    if score >= EVIDENCE_GROUNDED_THRESHOLD:
        return "grounded"
    if score >= EVIDENCE_PARTIAL_THRESHOLD:
        return "partial"
    return "ungrounded"


# Overlapping chunks to avoid answers being split across chunk boundaries
def split_list_overlap(lst, size, overlap=3):
    chunks = []
    i = 0
    while i < len(lst):
        chunks.append(lst[i:i + size])
        i += size - overlap
    return chunks

def compute_doc_fingerprint(pdf_path: str, chunk_size: int, min_tokens: int) -> str:
    stat = os.stat(pdf_path)
    raw = f"{pdf_path}|{stat.st_size}|{stat.st_mtime}|chunk_size={chunk_size}|min_tokens={min_tokens}"
    return hashlib.sha256(raw.encode()).hexdigest()

def doc_cache_filename(pdf_path: str) -> str:
    doc_name   = os.path.basename(pdf_path).replace(".pdf", "")
    short_hash = hashlib.sha256(pdf_path.encode()).hexdigest()[:10]
    return os.path.join(DOC_CACHE_DIR, f"{doc_name}__{short_hash}.pkl")

def load_manifest() -> dict:
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH, "r") as f:
            return json.load(f)
    return {}

def save_manifest(manifest: dict) -> None:
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)


# =============================================================================
# Document versioning
# =============================================================================

MONTH_NAME_TO_NUM = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
}

def parse_pdf_internal_date(raw: str | None):
    if not raw:
        return None
    match = re.match(r"D:(\d{4})(\d{2})(\d{2})", raw)
    if not match:
        return None
    year, month, day = match.groups()
    try:
        return datetime.date(int(year), int(month), int(day))
    except ValueError:
        return None

def extract_date_from_text(text: str):
    pattern = r"\b(" + "|".join(MONTH_NAME_TO_NUM.keys()) + r")\s+(\d{4})\b"
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if not match:
        return None
    month_name, year = match.groups()
    month = MONTH_NAME_TO_NUM[month_name.lower()]
    try:
        return datetime.date(int(year), month, 1)
    except ValueError:
        return None

def extract_document_title(doc) -> str:
    """
    Best-effort human-readable document title. Prefers PDF metadata when it is
    substantive; otherwise reconstructs the title from the heading lines on the
    first page (the lines before 'Guidance for Industry' / 'U.S. Department...').
    """
    _STOP = re.compile(
        r"^(guidance for industry|u\.s\. department|food and drug|draft guidance|"
        r"center for|oncology center|additional copies|contains nonbinding|"
        r"clinical/medical|procedural|docket|reprint|revision|"
        r"january|february|march|april|may|june|july|august|september|october|"
        r"november|december)\b", re.I)

    def _clean(t: str) -> str:
        t = re.sub(r"\s*\|\s*FDA\s*$", "", t, flags=re.I)     # strip "| FDA"
        t = re.sub(r"\s+\d{6,8}\s*$", "", t)                  # strip trailing yyyymmdd
        t = re.sub(r",?\s*Guidance for Industry.*$", "", t, flags=re.I)
        return re.sub(r"\s+", " ", t).strip(" ,:")

    meta = (doc.metadata or {}).get("title", "").strip()
    mt = _clean(meta) if meta else ""
    if mt and len(mt.split()) >= 3 and mt.lower() != "guidance for industry":
        return mt

    # Fall back to first-page heading lines.
    try:
        lines = [l.strip() for l in doc[0].get_text().splitlines() if l.strip()]
    except Exception:
        lines = []
    heading = []
    for l in lines:
        if _STOP.match(l):
            break
        heading.append(l)
        if len(" ".join(heading)) > 130:
            break
    fp = re.sub(r"\s+", " ", " ".join(heading)).strip(" ,:")
    return fp or mt or "Untitled document"


def extract_document_date_display(doc, pdf_path: str):
    """
    A human-friendly document date for display next to the title. Prefers an
    in-document 'Month YYYY' (the guidance date), then the PDF metadata creation
    date (e.g. 6/9/26), then the file modified date. Returns a string or None.
    """
    scan_text = ""
    for page_num in range(min(2, len(doc))):
        scan_text += doc[page_num].get_text()
    m = re.search(
        r"\b(January|February|March|April|May|June|July|August|September|"
        r"October|November|December)\s+(\d{4})\b", scan_text)
    if m:
        return f"{m.group(1)} {m.group(2)}"

    meta = doc.metadata or {}
    for key in ("creationDate", "modDate"):
        parsed = parse_pdf_internal_date(meta.get(key))
        if parsed:
            try:
                return parsed.strftime("%-m/%-d/%y")
            except ValueError:
                return parsed.strftime("%m/%d/%y")

    try:
        mtime = datetime.date.fromtimestamp(os.path.getmtime(pdf_path))
        try:
            return mtime.strftime("%-m/%-d/%y")
        except ValueError:
            return mtime.strftime("%m/%d/%y")
    except Exception:
        return None


def determine_doc_version(doc, pdf_path: str) -> dict:
    # Human-readable title + display date, computed once here so both the CLI
    # and the web app can show the real document title instead of the filename.
    title        = extract_document_title(doc)
    display_date = extract_document_date_display(doc, pdf_path)

    meta = doc.metadata or {}
    for key in ("creationDate", "modDate"):
        parsed = parse_pdf_internal_date(meta.get(key))
        if parsed:
            return {
                "version_label" : parsed.strftime("%Y-%m"),
                "version_source": "pdf_metadata",
                "version_date"  : parsed.isoformat(),
                "doc_title"     : title,
                "display_date"  : display_date,
            }

    scan_text = ""
    for page_num in range(min(2, len(doc))):
        scan_text += doc[page_num].get_text()
    parsed = extract_date_from_text(scan_text)
    if parsed:
        return {
            "version_label" : parsed.strftime("%Y-%m"),
            "version_source": "document_text",
            "version_date"  : parsed.isoformat(),
            "doc_title"     : title,
            "display_date"  : display_date,
        }

    mtime = datetime.date.fromtimestamp(os.path.getmtime(pdf_path))
    return {
        "version_label" : mtime.strftime("%Y-%m"),
        "version_source": "file_modified_date",
        "version_date"  : mtime.isoformat(),
        "doc_title"     : title,
        "display_date"  : display_date,
    }


def process_document(pdf_path: str, chunk_size: int, min_tokens: int, embedding_model) -> tuple:
    doc_name = os.path.basename(pdf_path).replace(".pdf", "")
    doc      = fitz.open(pdf_path)
    print(f"    -> Parsing '{doc_name}' ({len(doc)} pages)...")

    version_info = determine_doc_version(doc, pdf_path)

    pages_and_texts = []
    for page_number, page in enumerate(doc):
        text = text_formatter(page.get_text())
        pages_and_texts.append({
            "source_document": doc_name,
            "page_number"    : page_number,
            "text"           : text,
        })

    nlp = English()
    nlp.add_pipe("sentencizer")
    for item in pages_and_texts:
        item["sentences"] = [str(s) for s in nlp(item["text"]).sents]

    doc_chunks = []
    for item in pages_and_texts:
        for chunk in split_list_overlap(item["sentences"], chunk_size, overlap=3):
            sent_list = [normalize_ws(s) for s in chunk if normalize_ws(s)]
            joined      = " ".join(sent_list).strip()
            token_count = len(joined) / 4
            if token_count > min_tokens:
                doc_chunks.append({
                    "source_document"  : item["source_document"],
                    "page_number"      : item["page_number"],
                    "sentence_chunk"   : joined,
                    "chunk_sentences"  : sent_list,
                    "chunk_token_count": token_count,
                })

    print(f"    -> Embedding {len(doc_chunks)} chunks from '{doc_name}'...")
    text_chunks = [c["sentence_chunk"] for c in doc_chunks]

    if text_chunks:
        embeddings = embedding_model.encode(
            text_chunks,
            batch_size=16,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)
    else:
        embeddings = np.zeros((0, embedding_model.get_sentence_embedding_dimension()), dtype=np.float32)

    return doc_chunks, embeddings, version_info


def get_or_build_doc_cache(pdf_path: str, chunk_size: int, min_tokens: int,
                            embedding_model, manifest: dict) -> tuple:
    doc_name    = os.path.basename(pdf_path).replace(".pdf", "")
    fingerprint = compute_doc_fingerprint(pdf_path, chunk_size, min_tokens)
    cache_file  = doc_cache_filename(pdf_path)

    entry = manifest.get(pdf_path)
    if entry and entry.get("fingerprint") == fingerprint and os.path.exists(cache_file):
        try:
            with open(cache_file, "rb") as f:
                chunks, embeddings, version_info = pickle.load(f)
            try:
                _doc = fitz.open(pdf_path)
                version_info["doc_title"]    = extract_document_title(_doc)
                version_info["display_date"] = extract_document_date_display(_doc, pdf_path)
                _doc.close()
            except Exception as _e:
                print(f"    [title refresh warning] {doc_name}: {_e}")
            print(f"    [cache hit]  '{doc_name}' unchanged — loading cached chunks/embeddings")
            return chunks, embeddings, version_info, fingerprint, cache_file
        except Exception as e:
            print(f"    [cache warning] Failed to load cache for '{doc_name}' ({e}) — reprocessing")

    print(f"    [cache miss] '{doc_name}' new or changed — processing from scratch")
    chunks, embeddings, version_info = process_document(pdf_path, chunk_size, min_tokens, embedding_model)
    with open(cache_file, "wb") as f:
        pickle.dump((chunks, embeddings, version_info), f)
    return chunks, embeddings, version_info, fingerprint, cache_file


# =============================================================================
# Load embedding model, then build/load per-document caches
# =============================================================================

print("################ Loading embedding model ################")

device = "cuda" if torch.cuda.is_available() else "cpu"
embedding_model = SentenceTransformer("all-mpnet-base-v2", device=device)

print("################ Loading / building per-document caches ################")

manifest         = load_manifest()
updated_manifest = {}
all_chunks       = []
embeddings_list  = []
doc_entries_raw  = []

for pdf_path in PDF_PATHS:
    if not os.path.exists(pdf_path):
        print(f"    File not found, skipping: {pdf_path}")
        continue

    chunks, embeddings, version_info, fingerprint, cache_file = get_or_build_doc_cache(
        pdf_path, chunk_size, min_tokens, embedding_model, manifest
    )

    base_name  = os.path.basename(pdf_path).replace(".pdf", "")
    start_idx  = len(all_chunks)
    all_chunks.extend(chunks)
    end_idx    = len(all_chunks)
    embeddings_list.append(embeddings)

    doc_entries_raw.append({
        "pdf_path"    : pdf_path,
        "base_name"   : base_name,
        "version_info": version_info,
        "start_idx"   : start_idx,
        "end_idx"     : end_idx,
        "num_chunks"  : len(chunks),
    })

    updated_manifest[pdf_path] = {
        "fingerprint": fingerprint,
        "cache_file" : cache_file,
        "num_chunks" : len(chunks),
    }

# Clean up cache files for PDFs that were removed from PDF_PATHS
removed_paths = set(manifest.keys()) - set(updated_manifest.keys())
for stale_path in removed_paths:
    stale_cache = manifest[stale_path].get("cache_file")
    if stale_cache and os.path.exists(stale_cache):
        os.remove(stale_cache)
    print(f"    [cache] Removed stale cache for '{stale_path}' (no longer in PDF_PATHS)")

save_manifest(updated_manifest)

name_counts = {}
for entry in doc_entries_raw:
    name_counts[entry["base_name"]] = name_counts.get(entry["base_name"], 0) + 1

DOC_REGISTRY      = []
DOC_CHUNK_INDICES = {}
seen_labels       = set()

for i, entry in enumerate(doc_entries_raw):
    base_name    = entry["base_name"]
    version_info = entry["version_info"]

    if name_counts[base_name] > 1:
        final_label = f"{base_name} [{version_info['version_label']}]"
        if final_label in seen_labels:
            short_hash  = hashlib.sha256(entry["pdf_path"].encode()).hexdigest()[:6]
            final_label = f"{final_label} ({short_hash})"
    else:
        final_label = base_name

    seen_labels.add(final_label)

    for idx in range(entry["start_idx"], entry["end_idx"]):
        all_chunks[idx]["source_document"]    = final_label
        all_chunks[idx]["doc_version_label"]  = version_info["version_label"]
        all_chunks[idx]["doc_version_source"] = version_info["version_source"]

    DOC_CHUNK_INDICES[final_label] = list(range(entry["start_idx"], entry["end_idx"]))

    DOC_REGISTRY.append({
        "doc_id"         : f"d{i + 1}",
        "source_document": final_label,
        "base_name"      : base_name,
        "doc_title"      : version_info.get("doc_title") or base_name,
        "display_date"   : version_info.get("display_date"),
        "version_label"  : version_info["version_label"],
        "version_source" : version_info["version_source"],
        "path"           : entry["pdf_path"],
        "num_chunks"     : entry["num_chunks"],
    })

# =============================================================================
# Build FAISS vector index
# =============================================================================

print("################ Building FAISS vector index ################")

if embeddings_list:
    embeddings_np = np.concatenate(embeddings_list, axis=0)
else:
    embeddings_np = np.zeros((0, embedding_model.get_sentence_embedding_dimension()), dtype=np.float32)

embedding_dim = embeddings_np.shape[1]
faiss_index   = faiss.IndexFlatIP(embedding_dim)
if embeddings_np.shape[0] > 0:
    faiss_index.add(embeddings_np)

print(f"    Total chunks in index: {len(all_chunks)}  (from {len(DOC_REGISTRY)} document(s))")

print("################ Building per-document FAISS sub-indices ################")

DOC_FAISS_INDEX     = {}
DOC_LOCAL_TO_GLOBAL = {}

for label, global_indices in DOC_CHUNK_INDICES.items():
    if not global_indices:
        continue
    sub_embeddings = embeddings_np[global_indices]
    sub_index = faiss.IndexFlatIP(embedding_dim)
    if sub_embeddings.shape[0] > 0:
        sub_index.add(sub_embeddings)
    DOC_FAISS_INDEX[label]     = sub_index
    DOC_LOCAL_TO_GLOBAL[label] = global_indices

def print_doc_registry() -> None:
    print("\n" + "-"*78)
    print("LOADED DOCUMENTS")
    print("-"*78)
    for entry in DOC_REGISTRY:
        print(f"  [{entry['doc_id']}] {entry['source_document']}")
        print(f"        version   : {entry['version_label']}  (source: {entry['version_source']})")
        print(f"        chunks    : {entry['num_chunks']}")
        print(f"        path      : {entry['path']}")
    print("-"*78 + "\n")

print_doc_registry()

#### used AI to generate a basic list of common sub queries
generic_static_sub_queries = [
    {
        "match_any": ["eligib"],
        "sub_queries": [
            "criteria requirements inclusion exclusion patient population",
            "who qualifies conditions that must be met to participate",
        ],
    },
    {
        "match_any": ["require", "must", "mandatory", "obligat"],
        "sub_queries": [
            "mandatory obligations regulatory requirement shall must",
            "what is required by regulation or guidance document",
        ],
    },
    {
        "match_any": ["when ", "timing", "deadline", "how soon", "within"],
        "sub_queries": [
            "timing sequence order of events timeline milestones",
            "conditions that trigger this action deadline",
        ],
    },
    {
        "match_any": ["difference between", "differ", "vs ", "versus", "compared to", "contrast"],
        "sub_queries": [
            "comparison contrast distinction how does X differ from Y",
            "similarities and differences between two approaches or programmes",
        ],
    },
    {
        "match_any": ["defin", "what is", "what are", "meaning of"],
        "sub_queries": [
            "definition meaning how is this concept defined in the guidance",
            "scope and purpose of this term or programme",
        ],
    },
    {
        "match_any": ["process", "procedure", "how to", "steps", "workflow"],
        "sub_queries": [
            "step-by-step procedure workflow how to submit or implement",
            "timeline sequence submission process regulatory pathway",
        ],
    },
    {
        "match_any": ["exception", "special case", "unless", "except", "not required", "waive"],
        "sub_queries": [
            "exception to the general rule special circumstance edge case",
            "circumstances where standard rules or requirements do not apply",
        ],
    },
    {
        "match_any": ["inform", "consent", "disclosure", "patient information"],
        "sub_queries": [
            "informed consent disclosure requirements regulatory obligations",
            "patient information risks benefits alternative treatments",
        ],
    },
    {
        "match_any": ["safety", "adverse", "toxicity", "risk", "harm"],
        "sub_queries": [
            "safety monitoring adverse event reporting toxicity risk management",
            "sponsor obligations for subject safety in clinical trials",
        ],
    },
    {
        "match_any": ["submit", "submission", "application", "filing", "dossier"],
        "sub_queries": [
            "submission requirements application content format regulatory filing",
            "what must be included in the submission or application",
        ],
    },
    {
        "match_any": ["approv", "approval", "marketing authorisation", "nda", "bla"],
        "sub_queries": [
            "marketing application approval regulatory pathway NDA BLA",
            "conditions for approval review process decision",
        ],
    },
    {
        "match_any": ["monitor", "committee", "oversight", "review board", "irb"],
        "sub_queries": [
            "independent oversight committee monitoring responsibilities",
            "IRB review board safety monitoring committee obligations",
        ],
    },
    {
        "match_any": ["dose", "dosing", "dosage", "exposure", "pharmacokinetic", "pk"],
        "sub_queries": [
            "dose selection dosage optimisation pharmacokinetics exposure",
            "recommended phase 2 dose dose-limiting toxicity dose adjustment",
        ],
    },
    {
        "match_any": ["biomarker", "diagnostic", "ivd", "companion", "assay"],
        "sub_queries": [
            "biomarker IVD companion diagnostic analytical validation",
            "patient selection biomarker-defined population assay performance",
        ],
    },
    {
        "match_any": ["pediatric", "paediatric", "adolescent", "child", "minor"],
        "sub_queries": [
            "paediatric adolescent patient inclusion criteria dose adjustment",
            "age-appropriate dosing safety monitoring younger patients",
        ],
    },
    {
        "match_any": ["protocol", "amendment", "ind", "investigational new drug"],
        "sub_queries": [
            "protocol amendment IND submission content format requirements",
            "changes to protocol sponsor notification FDA communication",
        ],
    },
    {
        "match_any": ["cohort", "expansion", "phase 1", "first-in-human", "fih"],
        "sub_queries": [
            "expansion cohort FIH trial design objectives safety monitoring",
            "multiple cohort trial dose escalation subject population",
        ],
    },
]

# =============================================================================
# In-session cache for query expansion + judge output
# =============================================================================

_query_expansion_cache: dict[str, list[str]] = {}
_judge_cache: dict[str, "JudgeResult"] = {}

def _hash_key(*parts: str) -> str:
    raw = "||".join(parts)
    return hashlib.sha256(raw.encode()).hexdigest()

def _context_fingerprint(context_items: list[dict]) -> str:
    return "|".join(
        f"{c['source_document']}:{c['page_number']}:{c['confidence_score']:.4f}"
        for c in context_items
    )


def expand_query(query: str, client: AzureOpenAI, model: str, n_llm: int = 3) -> list[str]:
    """Generate LLM-assisted sub-queries to widen retrieval recall.

    Runs on Azure OpenAI (same deployment used for final answer generation),
    not the judge models -- see call sites in retrieve_with_confidence().
    """
    cache_key = _hash_key("expand", query.strip().lower())
    cached = _query_expansion_cache.get(cache_key)
    if cached is not None:
        print(f"    [query expansion] Cache hit — reusing {len(cached)} sub-queries from earlier this session")
        return cached

    sub_queries = [query]

    llm_prompt = f"""
    You are a search query expansion assistant helping retrieve information from regulatory documents.

    Given a user question, generate {n_llm} alternative search queries that together improve recall by:
    - Rephrasing using synonyms or related regulatory terminology
    - Targeting different aspects or sub-components of the question
    - Using both formal regulatory language and plain phrasing

    Rules:
    - Do NOT answer the question
    - Each sub-query should be meaningfully different from the others
    - Return ONLY a JSON array of strings, no preamble, no markdown fences

    User question: {query}

    Example output: ["sub-query 1", "sub-query 2", "sub-query 3"]"""

    def _call_llm(token_param: str):
        return client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": llm_prompt}],
            temperature=0.3,
            **{token_param: 250},
        )

    try:
        try:
            response = _call_llm("max_completion_tokens")
        except Exception as e:
            # Some deployments/API versions still expect the older 'max_tokens'
            # name instead of 'max_completion_tokens' (the naming requirement
            # differs by model generation). Retry once with the other name
            # before giving up, so a deployment-specific quirk can't silently
            # and permanently disable LLM-based query expansion.
            if "max_tokens" in str(e) or "max_completion_tokens" in str(e) or "unsupported_parameter" in str(e).lower():
                response = _call_llm("max_tokens")
            else:
                raise
        raw = response.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        llm_queries = json.loads(raw)

        if isinstance(llm_queries, list):
            added = 0
            for q in llm_queries:
                if isinstance(q, str) and q.strip() and q.strip() not in sub_queries:
                    sub_queries.append(q.strip())
                    added += 1
            print(f"    [query expansion] LLM added {added} sub-queries")
        else:
            print("    [query expansion] LLM returned unexpected format — skipping LLM queries")

    except Exception as e:
        print(f"    [query expansion] LLM call failed ({e}) — using generic fallback only")

    query_lower  = query.lower()
    generic_added = 0

    # Generic keyword-triggered expansions are the main source of cross-document
    # highlight bleed on narrow factual questions. Only apply them when
    # explicitly enabled (they can still help recall on broad/exploratory
    # questions). See USE_GENERIC_STATIC_EXPANSION.
    _rules = generic_static_sub_queries if USE_GENERIC_STATIC_EXPANSION else []
    for rule in _rules:
        match_any = rule.get("match_any", [])
        match_all = rule.get("match_all", [])

        any_hit = any(kw in query_lower for kw in match_any)
        all_hit = all(kw in query_lower for kw in match_all) if match_all else False

        if any_hit or all_hit:
            for q in rule["sub_queries"]:
                if q not in sub_queries:
                    sub_queries.append(q)
                    generic_added += 1

    if generic_added:
        print(f"    [query expansion] Generic patterns added {generic_added} sub-queries")

    print(f"    [query expansion] Total sub-queries: {len(sub_queries)} "
          f"(1 original + {len(sub_queries) - 1} expansions)")

    _query_expansion_cache[cache_key] = sub_queries
    return sub_queries


# =============================================================================
# Helper functions and prompt
# =============================================================================

GOLDEN_EXAMPLES = """
            ## EXAMPLES OF CORRECT ANSWERS (FORMAT ONLY)
            The following are examples of the expected answer FORMAT and quality.
            Each shows a User Query followed by the EXACT JSON object shape you must
            return (matching the LLMAnswer schema, see OUTPUT FORMAT).

            CRITICAL: These examples are for FORMAT ONLY. Do NOT reuse any facts,
            criteria, numbers, or wording from them in your answer. Some examples
            discuss topics (RTOR eligibility, adolescent dosing, expansion cohorts)
            that may resemble the current question -- ignore their CONTENT entirely.
            Every fact in your answer must come from the RETRIEVED CONTEXT passages
            above, not from these examples. If the retrieved context does not contain
            the answer, say so; do not fall back on an example's content.
            ---

            ### Example 1
            **User Query:** What criteria must a submission meet to be eligible for Real-Time Oncology Review (RTOR)?

            **Expected JSON Output:**
            {
              "document_title": "Real-Time Oncology Review (RTOR) — FDA programme page and RTOR Guidance for Industry",
              "document_date": "November 2023",
              "answer": "To be considered for RTOR, a submission must relate to a drug likely to demonstrate substantial improvement over available therapy or meeting Expedited Program criteria, use a straightforward study design with easily interpreted endpoints (e.g., overall survival, response rates), and have completed adequate dose optimisation.\\n\\nAccording to the FDA's RTOR guidance and the RTOR programme page, three core eligibility criteria must be satisfied. First, the drug must be likely to demonstrate substantial improvement over available therapies or must qualify for one of FDA's Expedited Programs (e.g., Breakthrough Therapy, Fast Track). Second, the study design must be straightforward and the clinical endpoints must be easily interpreted — examples given include overall survival and response rates. Third, FDA will also consider whether adequate dose optimisation has been performed to support the proposed dosage. Submissions with greater complexity may still be considered for RTOR on a case-by-case basis.\\n\\nProcedurally, the applicant initiates the process by contacting the appropriate application regulatory project manager (RPM) to request the RTOR Request Table after the database lock for the pivotal trial. The completed table — including justification for RTOR consideration and a proposed pre-submission timeline — is submitted to the IND or NDA/BLA file. A decision is communicated in approximately 20 business days of receipt. Under RTOR, components of the marketing application are submitted in up to three pre-submissions before the final complete application; the PDUFA review clock starts only upon receipt of the complete application.\\n\\nIt is important to note that acceptance into RTOR does not guarantee approvability of the application and does not alter PDUFA review timelines. Participation is voluntary.",
              "key_requirements": [
                "Drug likely to demonstrate substantial improvement over available therapy or meets Expedited Program criteria",
                "Straightforward study design with easily interpreted endpoints (e.g., overall survival, response rates)",
                "Adequate dose optimisation performed",
                "Pivotal trial database lock must be completed before requesting RTOR",
                "Decision communicated within ~20 business days of receipt of the completed RTOR Request Table",
                "PDUFA timelines are not affected; early approval is possible but not guaranteed"
              ],
              "source_evidence": [
                "Drugs likely to demonstrate substantial improvements over available therapy or meeting criteria for Expedited Programs.",
                "Straightforward study designs.",
                "Endpoints that can be easily interpreted (e.g., overall survival, response rates, etc.).",
                "FDA will also consider whether adequate dose optimization has been performed to support the proposed dosage.",
                "Acceptance into the RTOR program does not guarantee or influence approvability of the application",
                "PDUFA timelines will not be affected."
              ],
              "source_passages_used": "Context 1 and Context 2 (Real-Time Oncology Review FDA programme page — Scope and Standard Operating Procedures sections; RTOR FAQs, questions 3 and 5)",
              "confidence_score": 0.95
            }

            ---

            ### Example 2
            **User Query:** In the non-curative setting, must patients have received available therapy before enrolling in a clinical trial of an investigational drug?

            **Expected JSON Output:**
            {
              "document_title": "Cancer Clinical Trial Eligibility Criteria: Available Therapy in Non-Curative Settings",
              "document_date": "2022-07",
              "answer": "No — in the non-curative setting, patients may be eligible for clinical trials of investigational drugs regardless of whether they have previously received available therapy, provided that adequate informed consent is obtained disclosing all available treatment options.\\n\\nThe FDA guidance \\"Cancer Clinical Trial Eligibility Criteria: Available Therapy in Non-Curative Settings\\" (July 2022) addresses this directly. Historically, eligibility criteria in early-stage oncology drug development required patients to have received a certain number of available therapies or all standard-of-care options before enrolling. FDA notes that this type of requirement can inadvertently limit patient access to trials and preclude participation.\\n\\nIn the non-curative setting — defined as unresectable, locally advanced, or metastatic solid tumour disease, or haematologic malignancies with unfavourable long-term overall survival — it is considered reasonable for patients to receive an investigational drug in lieu of available therapy. Patients may therefore be enrolled in trials, including first-in-human trials, regardless of prior treatment history in that setting.\\n\\nThe critical safeguard is robust informed consent: per 21 CFR part 50.25, the informed consent must clearly state that other treatment options known to confer clinical benefit exist, must disclose reasonably foreseeable risks and potential benefits of the investigational drug, and must include a disclosure of appropriate alternative procedures or courses of treatment that might be advantageous to the subject.\\n\\nAn exception applies when available therapy offers the potential for cure in a substantial proportion of patients (e.g., paediatric ALL, classic Hodgkin lymphoma, testicular cancer): in those cases, patients should generally have received such curative-intent therapy first, or it should be administered concurrently as an add-on trial.",
              "key_requirements": [
                "In the non-curative setting, prior receipt of available therapy is NOT required for trial eligibility",
                "Applies to all trial phases, including first-in-human trials",
                "Informed consent (21 CFR part 50.25) must disclose all available therapies conferring clinical benefit and associated risks/benefits",
                "If efficacy interpretation requires a homogeneous population, patients with and without prior therapy should be evaluated in separate cohorts or pre-specified subgroup analyses",
                "Exception: curative-intent therapy must precede or be added on when cure is possible in a substantial proportion of patients"
              ],
              "source_passages_used": "Context 3 (Available Therapy in Non-Curative Settings guidance, July 2022, Section II — Background, page 2; Section III — Recommendations, pages 2-3)",
              "confidence_score": 0.93
            }

            ---

            ### Example 3
            **User Query:** What are the IDMC requirements for FIH multiple expansion cohort trials?

            **Expected JSON Output:**
            {
              "document_title": "Expansion Cohorts: Use in First-in-Human Clinical Trials to Expedite Development of Oncology Drugs and Biologics",
              "document_date": "2022-03",
              "answer": "An independent data monitoring committee (IDMC) must be established for all FIH multiple expansion cohort protocols, with responsibilities spanning real-time safety review, cumulative adverse event assessment, and protocol modification recommendations.\\n\\nThe FDA guidance on Expansion Cohorts: Use in First-in-Human Clinical Trials (March 2022) mandates that an IDMC — or other appropriately independent entity structured to assess both safety and efficacy — be established for all FIH multiple expansion cohort protocols. This requirement is driven by the inherent complexity of these trials, which involve different cohort objectives, multiple trial populations, and several dosages evaluated simultaneously, all of which create potential for increased risks to subjects.\\n\\nThe IDMC's core responsibilities include analysis of incoming expedited safety reports, review of cumulative summaries of all adverse events, and making recommendations to the IND sponsor regarding protocol modifications to reduce subject risk. Critically, the IDMC must conduct real-time review of all serious adverse events and meet periodically to assess the totality of safety information across the development programme.\\n\\nThe IDMC is also charged with performing both prespecified and ad hoc assessments of safety and futility for each cohort. Specific actions the IDMC may recommend include changing eligibility criteria if risks appear elevated in a subgroup, altering dosage or schedule, modifying safety monitoring plans, and requiring updates to the informed consent document — including, where necessary, reconsent of currently enrolled subjects.\\n\\nAn exception to the full external IDMC requirement applies in trials of limited sample size: no more than 90 subjects in the dose-finding portion and no more than 200 subjects across all dose-expansion cohorts. In such cases, sponsors may consider an internal safety assessment committee, provided a firewall is established ensuring that individuals performing safety assessments are not otherwise involved in trial conduct or management.",
              "key_requirements": [
                "IDMC (or equivalent independent entity) is mandatory for all FIH multiple expansion cohort protocols",
                "Must conduct real-time review of all serious adverse events and meet periodically to review cumulative safety data",
                "Responsibilities include: expedited safety report analysis, cumulative adverse event review, and protocol modification recommendations",
                "May recommend eligibility changes, dose/schedule alterations, monitoring plan modifications, and informed consent updates",
                "For trials <=90 subjects (dose-finding) and <=200 subjects (all expansion cohorts): an internal safety assessment committee with an appropriate firewall is acceptable",
                "Firewall required if internal: assessors must not be involved in trial conduct or management"
              ],
              "source_passages_used": "Context 4 (Expansion Cohorts FIH Guidance, March 2022, Section VII.B — Independent Data Monitoring Committee, pages 11-12)",
              "confidence_score": 0.9
            }

            ---

            ### Example 4 — thin context: correct behaviour is a SHORT, grounded answer
            **User Query:** Are juvenile animal studies required before enrolling adolescent patients in adult oncology clinical trials?

            **Retrieved context available for this example (for illustration only):**
            [Context 1] Source: Considerations for the Inclusion of Adolescent Patients in Adult Oncology Clinical Trials | Page: 4
            "Juvenile animal studies are not routinely needed before the enrollment of adolescent patients in adult oncology clinical trials, unless clinical and/or nonclinical data do not provide sufficient information on toxicities."

            Only ONE short passage was retrieved for this query. There is no supporting detail in the
            context about *why* this is the case, no mention of "21st Century Cures" initiatives, no
            general commentary on animal-testing policy, and no additional sponsor obligations. The
            CORRECT response reflects exactly that — it does not invent reasoning, background, or
            extra requirements the passage does not contain.

            Note on this example: this is also the reference case for answer LENGTH. A one-paragraph
            answer is correct here not because the question was simple, but because the retrieved
            context was thin -- compare this to Examples 1-3, where broader multi-part questions and
            richer retrieved context earned multi-paragraph answers. Match your answer's length to
            what's actually retrieved, not to the apparent complexity of the question in isolation.

            **CORRECT Expected JSON Output (grounded, appropriately short):**
            {
              "document_title": "Considerations for the Inclusion of Adolescent Patients in Adult Oncology Clinical Trials",
              "document_date": "2019-03",
              "answer": "No — juvenile animal studies are not routinely required before enrolling adolescent patients in adult oncology clinical trials, unless the existing clinical and/or nonclinical data do not adequately characterise toxicities. The context does not specify what would constitute 'sufficient' toxicity information or describe FDA's broader rationale for this position, so no further detail can be provided beyond this rule as stated.",
              "key_requirements": [
                "Juvenile animal studies are not routinely required before enrolling adolescent patients in adult oncology trials",
                "Exception: required when clinical and/or nonclinical data do not adequately characterise toxicities"
              ],
              "source_evidence": [
                "Juvenile animal studies are not routinely needed before the enrollment of adolescent patients in adult oncology clinical trials, unless clinical and/or nonclinical data do not provide sufficient information on toxicities."
              ],
              "source_passages_used": "Context 1 (Considerations for the Inclusion of Adolescent Patients in Adult Oncology Clinical Trials, March 2019, Section V — Safety Monitoring, page 4)",
              "confidence_score": 0.9
            }

            **INCORRECT Expected-to-avoid Output — DO NOT PRODUCE ANSWERS LIKE THIS:**
            {
              "document_title": "Considerations for the Inclusion of Adolescent Patients in Adult Oncology Clinical Trials",
              "document_date": "2019-03",
              "answer": "No — juvenile animal studies are not routinely required before enrolling adolescent patients in adult oncology clinical trials. Juvenile animal studies are not routinely needed before the enrollment of adolescent patients in adult oncology clinical trials, unless clinical and/or nonclinical data do not provide sufficient information on toxicities. This reflects FDA's broader effort, consistent with 21st Century Cures Act initiatives, to reduce reliance on animal testing and modernise nonclinical evaluation in paediatric drug development. Sponsors should also conduct a formal risk-benefit assessment weighing the developmental risks of juvenile toxicity testing against the potential benefit of earlier adolescent access, and should document this assessment in the IND. In practice, most sponsors elect to proceed directly to adolescent enrollment once adult pharmacokinetic and safety data are available, since juvenile animal models are widely considered to have limited translational value for predicting human paediatric toxicity.\\n\\nAdditionally, sponsors are encouraged to consult FDA's paediatric-focused advisory committees early in development to align on the toxicology package, and should anticipate that the requirement for juvenile animal data will vary by drug class and mechanism of action.",
              "key_requirements": [
                "Juvenile animal studies not routinely required",
                "Exception for insufficient toxicity data",
                "Reflects 21st Century Cures Act philosophy on reducing animal testing",
                "Sponsors should conduct and document a formal risk-benefit assessment",
                "Consult paediatric advisory committees early in development"
              ],
              "source_passages_used": "Context 1 (Considerations for the Inclusion of Adolescent Patients in Adult Oncology Clinical Trials, March 2019, page 4)",
              "confidence_score": 0.85
            }

            **Why the second version is WRONG — unsupported elaboration:**
            - "consistent with 21st Century Cures Act initiatives" — not stated anywhere in the retrieved
              passage; this is invented background presented as if it were sourced.
            - "Sponsors should also conduct a formal risk-benefit assessment... document this in the IND" —
              a fabricated procedural requirement with no basis in the context.
            - "juvenile animal models are widely considered to have limited translational value" —
              an unsupported generalisation inserted as if it were fact from the guidance.
            - "Consult FDA's paediatric-focused advisory committees early in development" and "the
              requirement... will vary by drug class and mechanism of action" — both are plausible-sounding
              regulatory statements that are NOT in the retrieved passage. Plausibility is not grounding.
            - The `key_requirements` list mirrors this by listing invented obligations (risk-benefit
              assessment, advisory committee consultation) as if they were binding requirements from the
              guidance, which will mislead an SME reviewer relying on this output.
            - Note that the single retrieved passage fully supports the CORRECT version above — the
              failure here is not insufficient context, it is the model adding unrequested content to
              reach a longer, more "complete-sounding" answer. A short, fully-grounded answer is the
              correct and preferred output whenever the context is limited, even for high-stakes
              regulatory queries.

            ---
        """

# =============================================================================
# Structured output schema (pydantic)
# =============================================================================

class LLMAnswer(BaseModel):
    document_title       : str = Field(..., description="Title of the primary source document(s) the answer is drawn from")
    document_date        : str = Field(..., description="Version/date of the primary source document(s), e.g. '2022-07' or 'November 2023'")
    answer               : str = Field(
        ...,
        description=(
            "Direct answer to the query, grounded only in the retrieved context. "
            "Length is NOT fixed -- it should scale naturally with the complexity of the "
            "question and the amount of directly relevant supporting text. A simple, "
            "narrow factual question (e.g. a single threshold, date, or yes/no) gets a "
            "short one-to-two sentence answer. A broad or multi-part question (e.g. "
            "'what are the requirements for X') gets a fully developed answer with "
            "multiple paragraphs if the context supports it. Never pad a simple answer "
            "to sound more complete, and never compress a multi-part answer to fit a "
            "target length. Use \\n\\n between paragraphs if more than one is needed."
        ),
    )
    key_requirements     : List[str] = Field(..., description="Key requirements / criteria, as a list of bullet points")
    source_evidence      : List[str] = Field(
        default_factory=list,
        description=(
            "VERBATIM quotes copied character-for-character from the retrieved "
            "context passages -- one short quote (roughly 8-30 words) for each "
            "distinct claim in the answer. These are NOT paraphrases: each string "
            "must appear literally in one of the Context passages so it can be "
            "located and highlighted in the source PDF for the SME. Quote only the "
            "operative words that actually answer the question -- do not quote whole "
            "paragraphs, headings, or surrounding narrative. If a fact cannot be "
            "quoted verbatim from the context, do not include it."
        ),
    )
    source_passages_used : str = Field(..., description="Which Context number(s) most supported the answer (document name and pages)")
    confidence_score     : float = Field(..., ge=0.0, le=1.0, description="Model's self-assessed confidence in the answer, between 0 and 1")


class QAResultRecord(LLMAnswer):
    query             : str
    source_references : str
    num_chunks_used   : int
    reviewer          : Optional[str]   = None
    review_priority   : Optional[float] = None
    review_timestamp  : Optional[str]   = None
    judge_confidence  : Optional[float] = None
    judge_reasoning   : Optional[str]   = None
    # Grader 1 — multi-judge consensus
    judge_mean        : Optional[float] = None
    judge_median      : Optional[float] = None
    judge_lower       : Optional[float] = None
    judge_upper       : Optional[float] = None
    judge_variance    : Optional[float] = None
    judge_deepseek_confidence : Optional[float] = None
    judge_deepseek_reasoning  : Optional[str]   = None
    judge_groq_confidence     : Optional[float] = None
    judge_groq_reasoning      : Optional[str]   = None
    semantic_match    : Optional[float] = None

class JudgeResult(BaseModel):
    judge_confidence  : Optional[float] = Field(None, ge=0.0, le=1.0, description="Consensus (mean) groundedness confidence score, 0-1, or None if all judge calls failed")
    judge_reasoning   : str = Field("", description="One short sentence explaining the judge's confidence score")
    judge_mean        : Optional[float] = Field(None, description="Mean groundedness score across the judge panel")
    judge_median      : Optional[float] = Field(None, description="Median groundedness score across the judge panel")
    judge_lower       : Optional[float] = Field(None, description="Lower bound (min) score across the judge panel")
    judge_upper       : Optional[float] = Field(None, description="Upper bound (max) score across the judge panel")
    judge_variance    : Optional[float] = Field(None, description="Disagreement signal: population std-dev of scores across the panel")
    judge_scores      : dict = Field(default_factory=dict, description="Per-model raw scores, keyed by 'provider:model'")
    judge_reasons     : dict = Field(default_factory=dict, description="Per-model raw reasoning strings, keyed by 'provider:model' (captures WHY each judge scored as it did, not just the pooled consensus reason)")
    judge_deepseek_confidence : Optional[float] = Field(None, description="DeepSeek judge's raw groundedness confidence, 0-1, or None if that call failed")
    judge_deepseek_reasoning  : Optional[str]   = Field(None, description="DeepSeek judge's own one-sentence reasoning for its score")
    judge_groq_confidence     : Optional[float] = Field(None, description="Groq judge's raw groundedness confidence, 0-1, or None if that call failed")
    judge_groq_reasoning      : Optional[str]   = Field(None, description="Groq judge's own one-sentence reasoning for its score")

def parse_llm_answer(raw_content: str) -> LLMAnswer:
    cleaned = raw_content.strip().replace("```json", "").replace("```", "").strip()
    return LLMAnswer.model_validate_json(cleaned)

def format_answer_for_judge_and_history(llm_answer: LLMAnswer) -> str:
    key_reqs = "\n".join(f"- {req}" for req in llm_answer.key_requirements)
    return (
        f"Answer: {llm_answer.answer}\n\n"
        f"Key Requirements / Criteria:\n{key_reqs}\n\n"
        f"Source Passages Used: {llm_answer.source_passages_used}"
    )

def print_llm_answer(llm_answer: LLMAnswer) -> None:
    print(f"Document Title:        {llm_answer.document_title}")
    print(f"Document Date:         {llm_answer.document_date}")
    print("Answer:")
    print(f"  {llm_answer.answer}")
    print("Key Requirements / Criteria:")
    for req in llm_answer.key_requirements:
        print(f"  - {req}")
    if llm_answer.source_evidence:
        print("Source Evidence (verbatim quotes for highlighting):")
        for ev in llm_answer.source_evidence:
            print(f"  » {ev}")
    print(f"Source Passages Used:  {llm_answer.source_passages_used}")
    print(f"Confidence Score:      {llm_answer.confidence_score:.2f}")

def build_prompt(query: str, context_items: list[dict], chat_history: list[dict]) -> str:
    context_passages = ""
    for i, item in enumerate(context_items, 1):
        context_passages += (
            f"\n[Context {i}] "
            f"Source: {item['source_document']} | "
            f"Page: {item['page_number']} | "
            f"Confidence: {item['confidence_score']:.4f}\n"
            f"{item['sentence_chunk']}\n"
        )

    history_text = ""
    if chat_history:
        history_text = "\n## CONVERSATION HISTORY\n"
        for turn in chat_history:
            role  = "User" if turn["role"] == "user" else "Assistant"
            history_text += f"{role}: {turn['content']}\n"
        history_text += "\n"

    prompt = f"""
            ## 1. ROLE
            You are a senior regulatory affairs specialist and AI assistant with deep expertise in FDA oncology
            drug development guidance, clinical trial design, and regulatory submission processes for NDAs and BLAs.
            You are supporting a Quality Assurance Framework for AI Retrieval Systems used by Clinical Research
            Organisations (CROs) to interpret FDA Oncology Center of Excellence (OCE) guidance documents.
            You are engaged in a multi-turn conversation — maintain context from previous turns.

            ## 2. TASK / INTENT
            Answer the current user query using ONLY the retrieved context passages provided.
            The answer will be reviewed by an SME as part of a human-in-the-loop validation process.
            Refer back to earlier parts of the conversation where relevant.
            {history_text}
            ## 3. STEP-BY-STEP INSTRUCTIONS
            1. Read all retrieved context passages (labelled with source document and page).
            2. Identify passages most directly relevant to the current query.
            3. If the question refers to something from the conversation history, incorporate that context.
            4. Synthesise a clear answer drawing only from the retrieved passages.
            5. Cite which Context number(s) most supported your answer.
            6. If the context does not contain sufficient information, explicitly state:
               "The provided context does not contain enough information to fully answer this query."
            7. Do NOT add explanatory rationale, background, caveats, generalisations, or "common
               knowledge" about FDA practice that is not explicitly stated in the retrieved passages —
               even if it is true, even if it sounds plausible, and even if it would make the answer
               feel more complete. If a passage states a rule but not the reason for it, report the
               rule only; do not construct or guess at a reason.
            8. Let the answer's length follow the question, not a fixed target. A narrow factual
               question (a single date, threshold, or yes/no) should get a short, direct answer --
               one to two sentences. A broad or multi-part question ("what are the requirements
               for...", "how does X differ from Y...") should get a fully developed answer with as
               many paragraphs as the retrieved passages genuinely support. Do not pad a short
               answer to look more complete, and do not compress a genuinely multi-part answer
               just to be brief. See Example 4 in the EXAMPLES section for a case where a short
               answer is correct even for a nontrivial regulatory question, because the retrieved
               context itself was thin.

            ## 4. INPUT DATA — RETRIEVED CONTEXT PASSAGES
            The following {len(context_items)} passages are ranked by semantic relevance to the current query.
            {context_passages}

            ## 5. CONTEXT & CONSTRAINTS
            - Answer ONLY from the retrieved passages. Do not use outside knowledge.
            - Use specific regulatory terminology (e.g. NDA, BLA, PDUFA, RTOR, pivotal trial).
            - Do not speculate beyond what is explicitly stated in the context.
            - This output may be cited in a formal QA validation report.

            ## 6. OUTPUT FORMAT
            Respond with ONLY a single valid JSON object — no markdown fences, no preamble, no
            postamble, no text before or after the JSON. The object MUST contain exactly these
            keys, matching the structure and level of detail shown in the examples below:

            {{
              "document_title": "<title of the primary source document(s) this answer is drawn from, taken from the Source field(s) in the retrieved context>",
              "document_date": "<version/date of the primary source document(s), e.g. '2022-07' or 'November 2023'>",
              "answer": "<the answer to the query, scaled in length to the question's complexity and to what the retrieved context supports -- short (1-2 sentences) for narrow factual questions, longer with \\n\\n-separated paragraphs for broad/multi-part questions. Every claim must trace back to a retrieved passage -- no inferred rationale, no outside/background knowledge, no invented requirements. See Example 4 (CORRECT vs INCORRECT) below.>",
              "key_requirements": ["<bullet point — only requirements explicitly stated in the retrieved passages>", "<bullet point>", "..."],
              "source_evidence": ["<a SHORT VERBATIM quote copied word-for-word from a Context passage that directly supports a claim in your answer -- roughly 8-30 words, just the operative wording, NOT the whole paragraph>", "<another verbatim quote for another claim>", "..."],
              "source_passages_used": "<e.g. 'Context 2 and Context 4 (document name, pages 3 and 5)'>",
              "confidence_score": <float between 0.0 and 1.0>
            }}

            ## 6a. SOURCE_EVIDENCE — READ CAREFULLY (drives source highlighting)
            The "source_evidence" strings are used to highlight the exact supporting
            text inside the source PDF for a human reviewer. Therefore:
            - Each string MUST be copied character-for-character from the Context
              passages above. Do NOT paraphrase, summarise, reorder, or stitch
              together words from different places into one quote.
            - Quote ONLY the minimal operative span that answers the question --
              the specific sentence, clause, threshold, or list item. Never quote a
              whole paragraph, a heading, or narrative scaffolding around the fact.
            - Provide one quote per distinct claim (typically 1-4 quotes total for a
              narrow question). If two claims share one sentence, one quote is fine.
            - If a claim in your answer cannot be supported by a verbatim quote from
              the context, remove that claim -- do not invent a quote.
            - Prefer several short, precise quotes over one long one.

            ## 7. TONE
            Precise, objective, formal regulatory language.

            ## 8. EXAMPLES
            {GOLDEN_EXAMPLES}

            ---
            Current User Query: {query}

            Answer:"""

    return prompt

# =============================================================================
# Document selection
# =============================================================================

def find_doc_matches(user_text: str) -> list[str]:
    text_lower = user_text.strip().lower()
    matches = []

    for entry in DOC_REGISTRY:
        if text_lower == entry["doc_id"].lower():
            return [entry["source_document"]]
        if text_lower in entry["source_document"].lower():
            matches.append(entry["source_document"])
        elif text_lower in entry["base_name"].lower():
            matches.append(entry["source_document"])

    seen   = set()
    deduped = []
    for m in matches:
        if m not in seen:
            seen.add(m)
            deduped.append(m)
    return deduped


def infer_doc_filter_from_query(query: str) -> list[str]:
    query_lower = query.lower()
    matches = []

    for entry in DOC_REGISTRY:
        significant_words = re.findall(r"[a-zA-Z]{4,}", entry["base_name"].lower())
        word_hits = sum(1 for w in significant_words if w in query_lower)
        year      = entry["version_label"].split("-")[0]
        year_hit  = year in query_lower

        if year_hit or word_hits >= 2:
            matches.append(entry["source_document"])

    seen    = set()
    deduped = []
    for m in matches:
        if m not in seen:
            seen.add(m)
            deduped.append(m)
    return deduped


# =============================================================================
# Retrieval with hybrid query expansion + optional document filtering
# =============================================================================

def search_subset(query_embedding: np.ndarray, doc_filter: str, k: int):
    sub_index      = DOC_FAISS_INDEX.get(doc_filter)
    local_to_global = DOC_LOCAL_TO_GLOBAL.get(doc_filter, [])

    if sub_index is None or sub_index.ntotal == 0:
        return [], []

    top_n = min(k, sub_index.ntotal)
    scores_arr, local_idxs_arr = sub_index.search(query_embedding, top_n)
    scores      = list(scores_arr[0])
    global_idxs = [local_to_global[int(i)] for i in local_idxs_arr[0]]
    return scores, global_idxs


def _apply_precision_filters(results: list[dict]) -> list[dict]:
    """
    Turn a raw, score-sorted candidate list into a tight, high-precision set
    suitable for HIGHLIGHTING to an SME. Applies, in order:

      1. Absolute score floor  (drop noise below RETRIEVAL_SCORE_FLOOR)
      2. Relative gap floor     (drop the long tail below REL_GAP * top_score)
      3. Single-document dominance (if one doc owns >= DOC_DOMINANCE_RATIO of the
         surviving score mass, keep ONLY that document -- kills cross-doc bleed)
      4. Hard cap of MAX_HIGHLIGHT_CHUNKS

    Rationale: on a narrow factual question the answer lives in ONE document and
    typically ONE or TWO passages. Returning a padded top_k across six documents
    is exactly what made the highlighter light up unrelated guidances.
    """
    if not results:
        return results

    # 1. Absolute floor
    kept = [r for r in results if r["confidence_score"] >= RETRIEVAL_SCORE_FLOOR]
    # If the floor removed everything (unusual phrasing / weak match), keep the
    # single best candidate so the SME still sees the most plausible source.
    if not kept:
        return [results[0]]

    # 2. Relative gap floor
    top_score = kept[0]["confidence_score"]
    kept = [r for r in kept if r["confidence_score"] >= RETRIEVAL_REL_GAP * top_score]

    # 3. Single-document dominance
    doc_mass: dict[str, float] = {}
    for r in kept:
        doc_mass[r["source_document"]] = doc_mass.get(r["source_document"], 0.0) + r["confidence_score"]
    total_mass = sum(doc_mass.values()) or 1.0
    best_doc, best_mass = max(doc_mass.items(), key=lambda kv: kv[1])
    if (best_mass / total_mass) >= DOC_DOMINANCE_RATIO:
        kept = [r for r in kept if r["source_document"] == best_doc]

    # 4. Hard cap
    return kept[:MAX_HIGHLIGHT_CHUNKS]


def retrieve_with_confidence(query: str, client: AzureOpenAI, model: str,
                              doc_filter: str | None = None, top_k: int = top_k,
                              apply_precision: bool = True,
                              return_diagnostics: bool = False):
    sub_queries  = expand_query(query, client, model)
    best_score   = {}   # idx -> best score across sub-queries (take MAX, not first)
    per_sub_query_ids: dict[str, set] = {}   # sub-query -> set of chunk idxs it retrieved

    for sub_query in sub_queries:
        query_embedding = embedding_model.encode(
            [sub_query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)

        if doc_filter:
            scores, idxs = search_subset(query_embedding, doc_filter, top_k)
        else:
            scores_arr, indices_arr = faiss_index.search(query_embedding, top_k)
            scores, idxs = list(scores_arr[0]), list(indices_arr[0])

        sub_ids = set()
        for score, idx in zip(scores, idxs):
            idx = int(idx)
            s   = float(score)
            sub_ids.add(idx)
            # Keep the BEST score a chunk earned across all sub-queries, rather
            # than whichever sub-query happened to surface it first. This stops
            # a weak generic sub-query from stamping a low score onto a chunk
            # that the main query matched strongly.
            if idx not in best_score or s > best_score[idx]:
                best_score[idx] = s
        per_sub_query_ids[sub_query] = sub_ids

    results = []
    for idx, s in best_score.items():
        chunk = all_chunks[idx].copy()
        chunk["confidence_score"] = round(s, 4)
        # Stamp the FAISS index as a stable per-chunk id, used by Grader 9
        # (sub-query agreement) to tell which sub-queries retrieved each chunk.
        chunk["chunk_id"] = idx
        results.append(chunk)

    results = sorted(results, key=lambda x: x["confidence_score"], reverse=True)

    if apply_precision:
        results = _apply_precision_filters(results)
    else:
        results = results[:top_k]

    if return_diagnostics:
        diagnostics = {
            "subquery_count"   : len(sub_queries),
            "per_sub_query_ids": per_sub_query_ids,
        }
        return results, diagnostics
    return results


# =============================================================================
# LLM as a judge  (multi-judge panel: DeepSeek + Groq llama-3.1-8b-instant)
# =============================================================================

def _extract_judge_score(raw: str) -> tuple[Optional[float], str]:
    """
    Robustly pull a confidence score out of a judge model's reply. Judges often
    wrap the JSON in prose ("Here is my assessment: {...} Hope this helps!") or
    markdown, which naive json.loads rejects -- previously this silently dropped
    the score and, with only one surviving judge, produced a flat constant.

    Strategy, in order:
      1. Strip code fences, try json.loads on the whole string.
      2. Find the first {...} block with a regex and json.loads that.
      3. Regex out a `confidence`/`score` number even from non-JSON prose.
    Returns (confidence in [0,1] or None, reasoning).
    """
    if not raw:
        return None, "[empty judge reply]"

    cleaned = raw.replace("```json", "").replace("```", "").strip()

    def _coerce(val) -> Optional[float]:
        try:
            f = float(val)
        except Exception:
            return None
        # Accept 0-1 floats or 0-100 percents.
        if f > 1.0:
            f = f / 100.0
        return max(0.0, min(1.0, f))

    # 1) whole-string JSON
    for candidate in (cleaned,):
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict) and ("confidence" in parsed or "score" in parsed):
                conf = _coerce(parsed.get("confidence", parsed.get("score")))
                if conf is not None:
                    return conf, str(parsed.get("reasoning", ""))
        except Exception:
            pass

    # 2) first {...} block embedded in prose
    m = re.search(r"\{.*?\}", cleaned, re.DOTALL)
    if m:
        try:
            parsed = json.loads(m.group(0))
            if isinstance(parsed, dict) and ("confidence" in parsed or "score" in parsed):
                conf = _coerce(parsed.get("confidence", parsed.get("score")))
                if conf is not None:
                    return conf, str(parsed.get("reasoning", ""))
        except Exception:
            pass

    # 3) regex a bare number after "confidence"/"score"
    m = re.search(r'(?:confidence|score)\D{0,8}(\d*\.?\d+)', cleaned, re.IGNORECASE)
    if m:
        conf = _coerce(m.group(1))
        if conf is not None:
            return conf, "[score recovered from non-JSON reply]"

    return None, f"[unparseable judge reply: {cleaned[:80]!r}]"


def _run_single_judge(provider: str, model: str, judge_prompt: str) -> tuple[Optional[float], str]:
    """Call one judge model and return (confidence, reasoning). confidence is None on failure."""
    try:
        if provider == "groq":
            resp = groq_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": judge_prompt}],
                max_tokens=300,
                temperature=0,
                # Ask Groq for strict JSON when the model/endpoint supports it.
                response_format={"type": "json_object"},
            )
            raw = resp.choices[0].message.content.strip()
        elif provider == "deepseek":
            # DeepSeek exposes an OpenAI-compatible endpoint, so we drive it
            # through the standard OpenAI SDK with a custom base_url. The
            # deepseek-chat model supports response_format={"type":"json_object"}
            # for guaranteed JSON output, which is exactly what the judge
            # extractor expects.
            resp = deepseek_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": judge_prompt}],
                max_tokens=300,
                temperature=0,
                response_format={"type": "json_object"},
            )
            raw = resp.choices[0].message.content.strip()
        elif provider == "gemini":
            resp = gemini_client.models.generate_content(
                model=model,
                contents=judge_prompt,
            )
            raw = (resp.text or "").strip()
        else:
            return None, f"[unknown provider: {provider}]"

        return _extract_judge_score(raw)
    except Exception as e:
        # If response_format isn't supported by the model, retry without it.
        if provider in ("groq", "deepseek") and "response_format" in str(e):
            try:
                client_obj = groq_client if provider == "groq" else deepseek_client
                resp = client_obj.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": judge_prompt}],
                    max_tokens=300,
                    temperature=0,
                )
                return _extract_judge_score(resp.choices[0].message.content.strip())
            except Exception as e2:
                return None, f"[{provider}:{model} unavailable: {e2}]"
        return None, f"[{provider}:{model} unavailable: {e}]"


def judge_answer_confidence(query: str, answer: str, context_items: list[dict]) -> JudgeResult:
    cache_key = _hash_key("judge", query.strip().lower(), answer.strip(), _context_fingerprint(context_items))
    cached = _judge_cache.get(cache_key)
    if cached is not None:
        print("    [judge] Cache hit — reusing judge result from earlier this session")
        return cached

    passages_text = "\n".join(
        f"[Context {i}] {item['sentence_chunk']}"
        for i, item in enumerate(context_items, 1)
    )

    judge_prompt = f"""You are a fair but careful regulatory QA grader. Assess whether the ANSWER's
        claims are supported by the CONTEXT passages. Judge support only -- not writing quality,
        not completeness beyond what was asked. Be accurate, not harsh: a well-grounded answer
        should score high, and minor wording differences are not faults.

        CONTEXT:
        {passages_text}

        QUERY: {query}

        ANSWER:
        {answer}

        ## STEP 1 -- Claim-by-claim audit
        Break the ANSWER into its distinct factual claims (each key requirement, specific fact,
        number, threshold, citation, and any asserted cause/rationale). Label EACH claim:
          [S]  supported by a CONTEXT passage. Paraphrase, reordering, or summarising the source
               in different words still counts as [S] -- the meaning must match, not the wording.
          [P]  mostly right but slightly imprecise, softened, or missing a minor qualifier.
          [U]  materially unsupported: a fact/number/scope not in the context, a rationale the
               context never gives, or something that contradicts the context.
        Guidance: only mark [U] when the claim would actually mislead an SME. Do NOT mark a claim
        down merely because the wording differs from the source, because it is a reasonable
        restatement, or because the answer is shorter/longer than you'd expect.

        ## STEP 2 -- Score
        Let S, P, U be the counts and N = S + P + U (N>=1).
        confidence = (S + 0.85*P + 0*U) / N, rounded to 2 decimals.
        Interpretation: an answer fully grounded in the context SHOULD score ~0.95-1.0. A single
        minor imprecision ([P]) barely moves it. Each materially unsupported claim ([U]) is what
        meaningfully lowers the score. Report 1.0 only when every claim is a clean [S].
        Do not default to 1.0 or 0.5 out of habit -- compute from your labels.

        ## OUTPUT
        Respond with ONLY a JSON object, no other text, no markdown fences, in this exact form:
        {{"confidence": <float 0.0-1.0>, "reasoning": "<one sentence with counts like 'S=4 P=1 U=0' and, if any, which claim was unsupported>"}}
        """

    # --- Multi-judge consensus: run each model in the panel ------------------
    per_model_scores  = {}
    per_model_reasons = {}

    for provider, model in JUDGE_PANEL:
        conf, reason = _run_single_judge(provider, model, judge_prompt)
        key = f"{provider}:{model}"
        per_model_reasons[key] = reason
        if conf is not None:
            per_model_scores[key] = round(conf, 4)
        print(f"    [judge] {key}: "
              f"{'%.2f' % conf if conf is not None else 'FAILED'} — {reason}")

    valid_scores = list(per_model_scores.values())

    # Pull out each panel member's OWN confidence + reasoning into explicit,
    # provider-named fields (in addition to the generic per_model_* dicts)
    # so "why did this judge say that" survives into every export even when
    # the panel composition or dict ordering changes. Matched by provider
    # prefix on the "provider:model" key (e.g. "deepseek:deepseek-chat").
    def _pick(provider: str):
        for key, reason in per_model_reasons.items():
            if key.startswith(f"{provider}:"):
                return per_model_scores.get(key), reason
        return None, None

    ds_conf, ds_reason     = _pick("deepseek")
    groq_conf, groq_reason = _pick("groq")

    if not valid_scores:
        # Whole panel failed
        result = JudgeResult(
            judge_confidence=None,
            judge_reasoning="[all judges unavailable]",
            judge_scores=per_model_reasons,
            judge_reasons=per_model_reasons,
            judge_deepseek_confidence=ds_conf,
            judge_deepseek_reasoning=ds_reason,
            judge_groq_confidence=groq_conf,
            judge_groq_reasoning=groq_reason,
        )
        _judge_cache[cache_key] = result
        return result

    j_mean   = round(mean(valid_scores), 4)
    j_median = round(median(valid_scores), 4)
    j_lower  = round(min(valid_scores), 4)
    j_upper  = round(max(valid_scores), 4)
    j_var    = round(pstdev(valid_scores), 4) if len(valid_scores) > 1 else 0.0

    # Consensus (mean) is the headline groundedness score used downstream.
    # Reasoning notes the panel size and disagreement spread for SME triage.
    consensus_reason = (
        f"consensus of {len(valid_scores)}/{len(JUDGE_PANEL)} judges "
        f"(mean={j_mean:.2f}, median={j_median:.2f}, range={j_lower:.2f}-{j_upper:.2f}, "
        f"std={j_var:.2f})"
    )

    result = JudgeResult(
        judge_confidence=j_mean,
        judge_reasoning=consensus_reason,
        judge_mean=j_mean,
        judge_median=j_median,
        judge_lower=j_lower,
        judge_upper=j_upper,
        judge_variance=j_var,
        judge_scores=per_model_scores,
        judge_reasons=per_model_reasons,
        judge_deepseek_confidence=ds_conf,
        judge_deepseek_reasoning=ds_reason,
        judge_groq_confidence=groq_conf,
        judge_groq_reasoning=groq_reason,
    )
    _judge_cache[cache_key] = result
    return result


# =============================================================================
# Grader 2 — Semantic match vs ground truth (code-based, deterministic)
# =============================================================================

def semantic_match_score(answer: str, context_items: list[dict]) -> Optional[float]:
    if not answer.strip() or not context_items:
        return None

    context_text = " ".join(c["sentence_chunk"] for c in context_items).strip()
    if not context_text:
        return None

    try:
        emb = embedding_model.encode(
            [answer, context_text],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)
        # Embeddings are normalised, so the dot product is cosine similarity.
        cosine = float(np.dot(emb[0], emb[1]))
        # Clamp into [0, 1] (cosine can be slightly negative for unrelated text).
        return round(max(0.0, min(1.0, cosine)), 4)
    except Exception as e:
        print(f"    [semantic match] Failed to compute score ({e})")
        return None


def review_priority(context_items: list[dict], judge_confidence: float | None = None,
                    judge_variance: float | None = None) -> tuple[float, bool]:
    if not context_items:
        return 0.0, True

    scores     = [c["confidence_score"] for c in context_items]
    mean_score = sum(scores) / len(scores)

    if judge_confidence is not None:
        priority = (0.5 * mean_score) + (0.5 * judge_confidence)
    else:
        priority = mean_score

    # Judge disagreement lowers the priority score (lower = more scrutiny):
    # when the panel disagrees, the answer needs a closer human look.
    if judge_variance is not None:
        priority -= judge_variance

    priority = max(0.0, min(1.0, priority))
    return priority, False


# =============================================================================
# EVAL SECTION -- DeepEval Metrics + Claim Decomposition  [ADDED]
# =============================================================================

class AzureDeepEvalWrapper(DeepEvalBaseLLM):
    """
    Routes DeepEval metric computation through the existing Azure OpenAI
    client instead of a standard OpenAI key. Only Graders 3 and 4 use this.
    Claim decomposition (Grader 5) calls azure_client directly.
    """

    def __init__(self, azure_client, deployment_name: str):
        self.client          = azure_client
        self.deployment_name = deployment_name

    def load_model(self):
        # Nothing to initialise -- azure_client is already set up.
        return self.client

    def generate(self, prompt: str) -> str:
        # Sync path used by DeepEval's metric.measure() calls.
        response = self.client.chat.completions.create(
            model=self.deployment_name,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=1024,
            temperature=0,
        )
        return response.choices[0].message.content.strip()

    async def a_generate(self, prompt: str) -> str:
        # Required by DeepEval's interface even though we don't use async.
        return self.generate(prompt)

    def get_model_name(self) -> str:
        return self.deployment_name

_DEEPEVAL_THRESHOLD = 0.5   # score below this counts as a failing test

def run_deepeval_metrics(
    query: str,
    raw_answer: str,
    context_items: list[dict],
) -> dict:
    """
    Grader 4: DeepEval Answer Relevancy.

    Args:
        query         : original user question
        raw_answer    : plain answer text only (no key_requirements / source footer)
                        — passing the full formatted string penalises the score
                          because DeepEval treats the citation footer as an
                          irrelevant statement, deflating relevancy unfairly.
        context_items : list of retrieved chunk dicts with 'sentence_chunk' key

    Returns dict with:
        answer_relevancy_score / reason / passed
    """

    # DeepEval expects retrieval_context as a flat list of strings.
    retrieval_context = [c["sentence_chunk"] for c in context_items]

    test_case = LLMTestCase(
        input=query,
        actual_output=raw_answer,
        retrieval_context=retrieval_context,
    )

    results = {
        "answer_relevancy_score" : None,
        "answer_relevancy_reason": "",
        "answer_relevancy_passed": False,
    }

    # Grader 4: Answer Relevancy -- does the answer address the question?
    try:
        _answer_relevancy_metric.measure(test_case)
        results["answer_relevancy_score"]  = round(_answer_relevancy_metric.score, 4)
        results["answer_relevancy_reason"] = _answer_relevancy_metric.reason or ""
        results["answer_relevancy_passed"] = _answer_relevancy_metric.is_successful()
        print(f"    [Grader 4 Answer Relevancy] {results['answer_relevancy_score']:.2f} "
              f"({'PASS' if results['answer_relevancy_passed'] else 'FAIL'}) "
              f"-- {results['answer_relevancy_reason']}")
    except Exception as e:
        results["answer_relevancy_reason"] = f"[unavailable: {e}]"
        print(f"    [Grader 4 Answer Relevancy] unavailable -- {e}")

    return results


# --- Grader 5: Claim Decomposition -------------------------------------------

def decompose_answer_into_claims(answer: str, azure_client, model: str) -> list[str]:
    """
    Step A: split the answer into a flat list of atomic factual claims.
    One verifiable statement per item, self-contained, no compound sentences.
    Returns a list of strings, or [] if the model call fails.
    """

    decompose_prompt = f"""You are a precise text analyser. Break the following answer into a flat
        list of atomic factual claims. Each claim must be:
        - A single verifiable statement (one fact per item)
        - Self-contained (understandable without reading the others)
        - Taken verbatim or minimally paraphrased from the answer text
        - NOT a heading, instruction, or meta-comment about the answer
        
        Return ONLY a JSON array of strings. No preamble, no markdown fences.
        Example: ["Claim 1.", "Claim 2.", "Claim 3."]
        
        ANSWER TO DECOMPOSE:
        {answer}
        """

    try:
        response = azure_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": decompose_prompt}],
            max_completion_tokens=800,
            temperature=0,
        )
        raw    = response.choices[0].message.content.strip()
        raw    = raw.replace("```json", "").replace("```", "").strip()
        claims = json.loads(raw)
        if isinstance(claims, list):
            return [str(c).strip() for c in claims if str(c).strip()]
        return []
    except Exception as e:
        print(f"    [Grader 5 Claim Decomposition] Decompose step failed: {e}")
        return []


def is_meta_claim(claim: str) -> bool:
    """
    Detect 'meta' claims: statements ABOUT the context or the answer itself,
    rather than factual assertions about the regulatory domain. Examples:

        "The provided context does not contain enough information..."
        "It does not state that X can be used to..."
        "This cannot be determined from the retrieved passages."

    These are honest hedges or refusals. They are not verifiable against the
    context the way a domain claim is -- asking a fact-checker whether the
    context 'contradicts' the statement "the context is insufficient" is a
    category error that previously produced phantom contradiction flags. Such
    claims are bucketed as 'meta' and excluded from the grounding ratio and the
    contradiction count, rather than being verified.

    Detection is deterministic and conservative: it keys on phrasings that refer
    to the context/answer's own sufficiency or silence. A claim that names a
    regulatory fact (even a negative one about the domain) is NOT treated as
    meta, so genuine domain claims are still verified normally.
    """
    c = " ".join((claim or "").lower().split())
    meta_patterns = (
        "context does not contain",
        "does not contain enough information",
        "not enough information",
        "insufficient information",
        "context does not provide",
        "context does not specify",
        "context does not state",
        "context does not mention",
        "does not state that",          # "it does not state that X..."
        "cannot be determined from",
        "cannot be answered from",
        "is not addressed in the",
        "provided context does not",
        "retrieved passages do not",
        "no information is provided",
        "the context is silent",
    )
    return any(p in c for p in meta_patterns)


def verify_claim_against_context(
    claim: str,
    context_items: list[dict],
    azure_client,
    model: str,
) -> dict:
    """
    Step B: verify one atomic claim against the retrieved chunks.
    Returns verdict: supported / unsupported / contradicted, plus a reason.
    """

    context_text = "\n".join(
        f"[Chunk {i+1}] {c['sentence_chunk']}"
        for i, c in enumerate(context_items)
    )

    verify_prompt = f"""You are a strict fact-checker for regulatory AI outputs.

        RETRIEVED CONTEXT (source of truth):
        {context_text}
        
        CLAIM TO VERIFY:
        {claim}
        
        Is this claim supported, unsupported, or contradicted by the retrieved context?
        
        Definitions:
        - supported    : the context explicitly states or clearly implies this claim
        - unsupported  : the context does not mention this claim (cannot tell if
                         true or false from context alone)
        - contradicted : the context explicitly disagrees with this claim
        
        Respond with ONLY a JSON object, no markdown fences:
        {{"verdict": "supported" | "unsupported" | "contradicted", "reason": "<one short sentence>"}}
        """

    try:
        response = azure_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": verify_prompt}],
            max_completion_tokens=150,
            temperature=0,
        )
        raw    = response.choices[0].message.content.strip()
        raw    = raw.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(raw)
        verdict = parsed.get("verdict", "error").lower().strip()
        if verdict not in ("supported", "unsupported", "contradicted"):
            verdict = "error"
        return {
            "claim"  : claim,
            "verdict": verdict,
            "reason" : str(parsed.get("reason", "")),
        }
    except Exception as e:
        return {"claim": claim, "verdict": "error", "reason": f"verification failed: {e}"}


def run_claim_decomposition(
    answer: str,
    context_items: list[dict],
    azure_client,
    model: str,
) -> dict:
    """
    Grader 5: full claim decomposition pipeline -- decompose then verify each.

    Returns dict with:
        claims_total / supported / unsupported / contradicted
        grounding_ratio     : float 0-1 (supported / total), None if no claims
        unsupported_claims  : list of claim strings not found in context
        contradicted_claims : list of claim strings the context disagrees with
        per_claim_results   : full per-claim breakdown (kept in memory, not CSV)
        decomposition_error : True if decompose step returned nothing
    """

    print("    [Grader 5 Claim Decomposition] Decomposing answer into atomic claims...")
    claims = decompose_answer_into_claims(answer, azure_client, model)

    if not claims:
        print("    [Grader 5 Claim Decomposition] No claims extracted -- skipping.")
        return {
            "claims_total"       : 0,
            "claims_supported"   : 0,
            "claims_unsupported" : 0,
            "claims_contradicted": 0,
            "claims_meta"        : 0,
            "claims_verifiable"  : 0,
            "grounding_ratio"    : None,
            "unsupported_claims" : [],
            "contradicted_claims": [],
            "meta_claims"        : [],
            "per_claim_results"  : [],
            "decomposition_error": True,
        }

    print(f"    [Grader 5 Claim Decomposition] {len(claims)} claims. Verifying each...")

    per_claim_results = []
    for i, claim in enumerate(claims):
        if is_meta_claim(claim):
            # Statement about the context/answer itself -- not verifiable as a
            # domain fact. Bucket as 'meta' without an LLM call (also saves cost).
            result = {"claim": claim, "verdict": "meta",
                      "reason": "meta-statement about the context/answer; excluded from grounding"}
        else:
            result = verify_claim_against_context(claim, context_items, azure_client, model)
        per_claim_results.append(result)
        symbol = {"supported": "v", "unsupported": "?",
                  "contradicted": "x", "meta": "\u2298"}.get(result["verdict"], "!")
        print(f"      [{i+1}/{len(claims)}] {symbol} {result['verdict'].upper():12s} "
              f"-- {claim[:70]}{'...' if len(claim) > 70 else ''}")

    n_supported    = sum(1 for r in per_claim_results if r["verdict"] == "supported")
    n_unsupported  = sum(1 for r in per_claim_results if r["verdict"] == "unsupported")
    n_contradicted = sum(1 for r in per_claim_results if r["verdict"] == "contradicted")
    n_meta         = sum(1 for r in per_claim_results if r["verdict"] == "meta")
    n_total        = len(per_claim_results)
    # Grounding is over VERIFIABLE claims only: meta-statements leave the
    # denominator so an honest refusal is not penalised as ungrounded.
    n_verifiable    = n_total - n_meta
    grounding_ratio = round(n_supported / n_verifiable, 4) if n_verifiable > 0 else None

    print(f"    [Grader 5 Claim Decomposition] {n_supported} supported | "
          f"{n_unsupported} unsupported | {n_contradicted} contradicted | "
          f"{n_meta} meta"
          + (f" | grounding ratio: {grounding_ratio:.2f} (of {n_verifiable} verifiable)"
             if grounding_ratio is not None else ""))

    return {
        "claims_total"       : n_total,
        "claims_supported"   : n_supported,
        "claims_unsupported" : n_unsupported,
        "claims_contradicted": n_contradicted,
        "claims_meta"        : n_meta,
        "claims_verifiable"  : n_verifiable,
        "grounding_ratio"    : grounding_ratio,
        "unsupported_claims" : [r["claim"] for r in per_claim_results if r["verdict"] == "unsupported"],
        "contradicted_claims": [r["claim"] for r in per_claim_results if r["verdict"] == "contradicted"],
        "meta_claims"        : [r["claim"] for r in per_claim_results if r["verdict"] == "meta"],
        "per_claim_results"  : per_claim_results,
        "decomposition_error": False,
    }


# =============================================================================
# Grader 6 — Transcript / response metrics (code-based, zero LLM cost)
# =============================================================================

def compute_transcript_metrics(
    raw_answer: str,
    context_items: list[dict],
    llm_answer,
    sub_queries: list[str],
) -> dict:
    """
    Grader 6: deterministic transcript and response-level metrics.

    Args:
        raw_answer    : plain answer text (llm_answer.answer)
        context_items : retrieved chunks with confidence_score
        llm_answer    : parsed LLMAnswer pydantic object
        sub_queries   : list returned by expand_query for this turn

    Returns dict with:
        answer_len_chars / answer_len_tokens (approx)
    """

    answer_len_chars  = len(raw_answer)
    # Approximate token count (same heuristic used elsewhere in the pipeline).
    answer_len_tokens = round(answer_len_chars / 4)

    print(f"[Grader 6 Transcript Metrics] answer={answer_len_tokens} tokens")

    return {
        "answer_len_chars" : answer_len_chars,
        "answer_len_tokens": answer_len_tokens,
    }


# =============================================================================
# Grader 7 — Coverage check: key_requirements vs answer prose (code-based)
# =============================================================================

COVERAGE_STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "of", "in", "to", "and",
    "or", "for", "that", "this", "it", "its", "be", "by", "with", "as",
    "at", "on", "not", "no", "if", "from",
}

def _key_tokens(text: str) -> set[str]:
    """Return lowercase alphabetic tokens ≥4 chars, excluding stop words."""
    return {
        w for w in re.findall(r"[a-zA-Z]{4,}", text.lower())
        if w not in COVERAGE_STOP_WORDS
    }

def compute_coverage_check(llm_answer) -> dict:
    """
    Grader 7: check that each key_requirement is represented in the answer prose.

    A requirement is considered 'covered' if at least 2 of its significant
    keyword tokens (≥4 chars, non-stop-word) appear in the answer text.
    This threshold is intentionally lenient to allow paraphrasing.

    Returns dict with:
        requirements_covered    : int
        requirements_uncovered  : int
        uncovered_requirements  : list of requirement strings not found in prose
    """

    reqs = llm_answer.key_requirements or []
    if not reqs:
        print("[Grader 7 Coverage Check] No key_requirements to check.")
        return {
            "requirements_covered"  : 0,
            "requirements_uncovered": 0,
            "uncovered_requirements": [],
        }

    answer_tokens = _key_tokens(llm_answer.answer)
    covered   = []
    uncovered = []

    for req in reqs:
        req_tokens = _key_tokens(req)
        overlap    = req_tokens & answer_tokens
        # Covered if ≥2 significant tokens overlap, or the requirement is very
        # short (only 1 significant token) and that token is present.
        min_overlap = 1 if len(req_tokens) <= 2 else 2
        if len(overlap) >= min_overlap:
            covered.append(req)
        else:
            uncovered.append(req)

    n_total     = len(reqs)
    n_covered   = len(covered)
    n_uncovered = len(uncovered)

    print(f"[Grader 7 Coverage Check] "
          f"{n_covered}/{n_total} requirements covered in answer prose")
    if uncovered:
        for r in uncovered:
            print(f"    ! NOT COVERED: {r[:80]}{'...' if len(r) > 80 else ''}")

    return {
        "requirements_covered"  : n_covered,
        "requirements_uncovered": n_uncovered,
        "uncovered_requirements": uncovered,
    }


# =============================================================================
# Grader 8 — Retrieval quality distribution (code-based, zero LLM cost)
# =============================================================================

WEAK_CHUNK_THRESHOLD = 0.40   # chunks below this are considered low-signal

def compute_retrieval_quality(context_items: list[dict]) -> dict:
    """
    Grader 8: distribution-level retrieval quality metrics.

    Args:
        context_items : retrieved chunks, each with a 'confidence_score' key

    Returns dict with:
        retrieval_score_max / min / mean / spread
        weak_chunks_ratio   : weak_chunks / total
    """

    if not context_items:
        return {
            "retrieval_score_max"   : None,
            "retrieval_score_min"   : None,
            "retrieval_score_mean"  : None,
            "retrieval_score_spread": None,
            "weak_chunks_ratio"     : None,
        }

    scores = [c["confidence_score"] for c in context_items]
    s_max  = round(max(scores), 4)
    s_min  = round(min(scores), 4)
    s_mean = round(sum(scores) / len(scores), 4)
    spread = round(s_max - s_min, 4)

    weak_count = sum(1 for s in scores if s < WEAK_CHUNK_THRESHOLD)
    weak_ratio = round(weak_count / len(scores), 4)

    print(f"[Grader 8 Retrieval Quality] "
          f"max={s_max:.3f} min={s_min:.3f} mean={s_mean:.3f} spread={spread:.3f} | "
          f"weak ratio={weak_ratio:.2f}")

    return {
        "retrieval_score_max"   : s_max,
        "retrieval_score_min"   : s_min,
        "retrieval_score_mean"  : s_mean,
        "retrieval_score_spread": spread,
        "weak_chunks_ratio"     : weak_ratio,
    }


# =============================================================================
# Grader 9 — Sub-query agreement (code-based, zero LLM cost)
# =============================================================================

def compute_subquery_agreement(context_items: list[dict],
                               retrieval_diagnostics: dict | None) -> dict:
    """
    Grader 9: agreement between sub-queries over the final context chunks.

    Returns dict with:
        subquery_count           : number of sub-queries used (incl. original)
        subquery_agreement_ratio : fraction of final chunks retrieved by >1
                                   sub-query (None if diagnostics unavailable)
        subquery_mean_hits       : mean number of sub-queries that retrieved
                                   each final chunk (None if unavailable)
    """
    diag = retrieval_diagnostics or {}
    per_sub: dict = diag.get("per_sub_query_ids") or {}
    n_sub = diag.get("subquery_count") or (len(per_sub) if per_sub else 0)

    final_ids = [c.get("chunk_id") for c in context_items if c.get("chunk_id") is not None]

    if not per_sub or not final_ids:
        print("[Grader 9 Sub-query Agreement] diagnostics unavailable — skipping")
        return {
            "subquery_count"          : n_sub or None,
            "subquery_agreement_ratio": None,
            "subquery_mean_hits"      : None,
        }

    # For each final chunk, count how many sub-queries retrieved it.
    hits = []
    for cid in final_ids:
        h = sum(1 for ids in per_sub.values() if cid in ids)
        hits.append(h)

    multi = sum(1 for h in hits if h > 1)
    agreement = round(multi / len(hits), 4) if hits else None
    mean_hits = round(sum(hits) / len(hits), 3) if hits else None

    print(f"[Grader 9 Sub-query Agreement] {multi}/{len(hits)} final chunks "
          f"retrieved by >1 of {n_sub} sub-queries (agreement={agreement})")

    return {
        "subquery_count"          : n_sub,
        "subquery_agreement_ratio": agreement,
        "subquery_mean_hits"      : mean_hits,
    }




# =============================================================================
# Grader 11 — Evidence alignment rate (code-based, reuses align_evidence)
# =============================================================================

EVIDENCE_ALIGN_THRESHOLD = 80.0   # align_evidence score (0-100) to count as a hit

def compute_evidence_alignment(llm_answer, context_items: list[dict]) -> dict:
    """
    Grader 11: fraction of the model's source_evidence quotes that align to the
    retrieved context.

    Returns dict with:
        evidence_quotes_total    : number of source_evidence strings
        evidence_quotes_aligned  : how many aligned above threshold
        evidence_alignment_rate  : aligned / total (None if no quotes)
        unaligned_evidence       : list of quotes that failed to align
    """
    quotes = list(getattr(llm_answer, "source_evidence", []) or [])
    if not quotes:
        print("[Grader 11 Evidence Alignment] no source_evidence quotes to check")
        return {
            "evidence_quotes_total"  : 0,
            "evidence_quotes_aligned": 0,
            "evidence_alignment_rate": None,
            "unaligned_evidence"     : [],
        }

    haystack = " ".join(c.get("sentence_chunk", "") for c in context_items)
    aligned, unaligned = 0, []
    for q in quotes:
        res = align_evidence(q, haystack)
        if res["score"] >= EVIDENCE_ALIGN_THRESHOLD:
            aligned += 1
        else:
            unaligned.append(q)

    rate = round(aligned / len(quotes), 4)
    print(f"[Grader 11 Evidence Alignment] {aligned}/{len(quotes)} quotes align "
          f"to context (rate={rate})")
    for q in unaligned:
        print(f"    ! UNALIGNED: {q[:80]}{'...' if len(q) > 80 else ''}")

    return {
        "evidence_quotes_total"  : len(quotes),
        "evidence_quotes_aligned": aligned,
        "evidence_alignment_rate": rate,
        "unaligned_evidence"     : unaligned,
    }


# =============================================================================
# Grader 12 — Context utilisation (code-based, reuses align_evidence)
# =============================================================================

def compute_context_utilisation(llm_answer, context_items: list[dict]) -> dict:
    """
    Grader 12: fraction of retrieved chunks that the answer's evidence drew on.

    Returns dict with:
        chunks_total       : number of retrieved chunks
        chunks_used        : chunks that matched >=1 evidence quote
        context_utilisation: chunks_used / chunks_total (None if no chunks)
    """
    n = len(context_items)
    if n == 0:
        return {"chunks_total": 0, "chunks_used": 0, "context_utilisation": None}

    quotes = list(getattr(llm_answer, "source_evidence", []) or [])
    if not quotes:
        print("[Grader 12 Context Utilisation] no evidence quotes — utilisation not scored")
        return {"chunks_total": n, "chunks_used": 0, "context_utilisation": None}

    used = 0
    for c in context_items:
        chunk_text = c.get("sentence_chunk", "")
        if any(align_evidence(q, chunk_text)["score"] >= EVIDENCE_ALIGN_THRESHOLD
               for q in quotes):
            used += 1

    util = round(used / n, 4)
    print(f"[Grader 12 Context Utilisation] {used}/{n} retrieved chunks used "
          f"(utilisation={util})")

    return {"chunks_total": n, "chunks_used": used, "context_utilisation": util}


# =============================================================================
# Grader 13 — Numeric consistency (code-based, zero LLM cost)
# =============================================================================

_NUM_RE = re.compile(r"\b\d+(?:\.\d+)?\b")

def compute_numeric_consistency(answer: str, context_items: list[dict]) -> dict:
    """
    Grader 13: check that numbers stated in the answer appear in the context.

    Returns dict with:
        numbers_total       : distinct numeric tokens in the answer
        numbers_grounded    : how many also appear in the retrieved context
        numeric_consistency : grounded / total (None if the answer has no numbers)
        ungrounded_numbers  : list of numeric tokens not found in the context
    """
    ans_nums = set(_NUM_RE.findall(answer or ""))
    if not ans_nums:
        return {
            "numbers_total"      : 0,
            "numbers_grounded"   : 0,
            "numeric_consistency": None,
            "ungrounded_numbers" : [],
        }

    ctx_text = " ".join(c.get("sentence_chunk", "") for c in context_items)
    ctx_nums = set(_NUM_RE.findall(ctx_text))

    grounded   = [n for n in ans_nums if n in ctx_nums]
    ungrounded = sorted(ans_nums - ctx_nums, key=lambda x: (len(x), x))
    ratio = round(len(grounded) / len(ans_nums), 4)

    print(f"[Grader 13 Numeric Consistency] {len(grounded)}/{len(ans_nums)} "
          f"answer numbers found in context (consistency={ratio})")
    if ungrounded:
        print(f"    ! NOT IN CONTEXT: {', '.join(ungrounded[:10])}")

    return {
        "numbers_total"      : len(ans_nums),
        "numbers_grounded"   : len(grounded),
        "numeric_consistency": ratio,
        "ungrounded_numbers" : ungrounded,
    }


def run_eval_suite(
    query: str,
    answer: str,
    context_items: list[dict],
    azure_client,
    model: str,
    raw_answer: str = "",
    llm_answer=None,
    sub_queries: list[str] | None = None,
    retrieval_diagnostics: dict | None = None,
) -> dict:
    """
    Combined runner for Graders 4-8.
    Call in the chat loop after the judge panel and semantic match.
    Returns a flat dict ready to merge into result_row.

    Args:
        query       : original user question
        answer      : full formatted answer string (used by claim decomposition)
        raw_answer  : plain answer text only, passed to DeepEval Answer Relevancy
                      to avoid the citation footer deflating the relevancy score.
                      Falls back to `answer` if not supplied.
        llm_answer  : parsed LLMAnswer pydantic object (Graders 6 and 7)
        sub_queries : list of sub-queries used during retrieval (Grader 6)
    """

    print("\n[Running eval suite (Graders 4-8: DeepEval + Claim Decomposition + Code-based)...]")
    deepeval_results  = run_deepeval_metrics(query, raw_answer or answer, context_items)
    claim_results     = run_claim_decomposition(answer, context_items, azure_client, model)
    transcript_metrics = compute_transcript_metrics(
        raw_answer or answer, context_items, llm_answer, sub_queries or []
    )
    coverage_results  = compute_coverage_check(llm_answer)
    retrieval_quality = compute_retrieval_quality(context_items)
    subquery_agreement = compute_subquery_agreement(context_items, retrieval_diagnostics)
    numeric_results    = compute_numeric_consistency(raw_answer or answer, context_items)
    # Graders 11-12 need the parsed answer's verbatim quotes; skip if absent.
    if llm_answer is not None:
        evidence_alignment = compute_evidence_alignment(llm_answer, context_items)
        context_util       = compute_context_utilisation(llm_answer, context_items)
    else:
        evidence_alignment = {"evidence_quotes_total": 0, "evidence_quotes_aligned": 0,
                              "evidence_alignment_rate": None, "unaligned_evidence": []}
        context_util       = {"chunks_total": len(context_items), "chunks_used": 0,
                              "context_utilisation": None}

    return {
        # Grader 4 -- DeepEval Answer Relevancy
        "deepeval_answer_relevancy_score" : deepeval_results["answer_relevancy_score"],
        "deepeval_answer_relevancy_reason": deepeval_results["answer_relevancy_reason"],
        "deepeval_answer_relevancy_passed": deepeval_results["answer_relevancy_passed"],
        # Grader 5 -- Claim Decomposition
        "claim_grounding_ratio"           : claim_results["grounding_ratio"],
        "claims_total"                    : claim_results["claims_total"],
        "claims_supported"                : claim_results["claims_supported"],
        "claims_unsupported"              : claim_results["claims_unsupported"],
        "claims_contradicted"             : claim_results["claims_contradicted"],
        "claims_meta"                     : claim_results["claims_meta"],
        "claims_verifiable"               : claim_results["claims_verifiable"],
        "unsupported_claims"              : claim_results["unsupported_claims"],
        "contradicted_claims"             : claim_results["contradicted_claims"],
        "meta_claims"                     : claim_results["meta_claims"],
        # Kept in memory only -- too verbose for CSV
        "_per_claim_results"              : claim_results["per_claim_results"],
        "decomposition_error"             : claim_results["decomposition_error"],
        # Grader 6 -- Transcript / response metrics
        "answer_len_chars"                : transcript_metrics["answer_len_chars"],
        "answer_len_tokens"               : transcript_metrics["answer_len_tokens"],
        # Grader 7 -- Coverage check: key_requirements vs answer prose
        "requirements_covered"            : coverage_results["requirements_covered"],
        "requirements_uncovered"          : coverage_results["requirements_uncovered"],
        "uncovered_requirements"          : coverage_results["uncovered_requirements"],
        # Grader 8 -- Retrieval quality distribution
        "retrieval_score_max"             : retrieval_quality["retrieval_score_max"],
        "retrieval_score_min"             : retrieval_quality["retrieval_score_min"],
        "retrieval_score_mean"            : retrieval_quality["retrieval_score_mean"],
        "retrieval_score_spread"          : retrieval_quality["retrieval_score_spread"],
        "weak_chunks_ratio"               : retrieval_quality["weak_chunks_ratio"],
        # Grader 9 -- Sub-query agreement
        "subquery_count"                  : subquery_agreement["subquery_count"],
        "subquery_agreement_ratio"        : subquery_agreement["subquery_agreement_ratio"],
        "subquery_mean_hits"              : subquery_agreement["subquery_mean_hits"],
        # Grader 11 -- Evidence alignment rate
        "evidence_quotes_total"           : evidence_alignment["evidence_quotes_total"],
        "evidence_quotes_aligned"         : evidence_alignment["evidence_quotes_aligned"],
        "evidence_alignment_rate"         : evidence_alignment["evidence_alignment_rate"],
        "unaligned_evidence"              : evidence_alignment["unaligned_evidence"],
        # Grader 12 -- Context utilisation
        "chunks_total"                    : context_util["chunks_total"],
        "chunks_used"                     : context_util["chunks_used"],
        "context_utilisation"             : context_util["context_utilisation"],
        # Grader 13 -- Numeric consistency
        "numbers_total"                   : numeric_results["numbers_total"],
        "numbers_grounded"                : numeric_results["numbers_grounded"],
        "numeric_consistency"             : numeric_results["numeric_consistency"],
        "ungrounded_numbers"              : numeric_results["ungrounded_numbers"],
    }


def print_eval_summary(eval_results: dict) -> None:
    """
    Print a compact summary of Graders 3-5 for the SME reviewer,
    shown just before the HITL approve/reject prompt.
    """

    print("\n--- EVAL SCORES (Graders 4-5) ---")

    ar_score = eval_results.get("deepeval_answer_relevancy_score")

    if ar_score is not None:
        print(f"  Grader 4 Answer Relevancy: {ar_score:.2f}  "
              f"[{'PASS' if eval_results.get('deepeval_answer_relevancy_passed') else 'FAIL'}]")
        print(f"    {eval_results.get('deepeval_answer_relevancy_reason', '')}")
    else:
        print("  Grader 4 Answer Relevancy: unavailable")

    grounding = eval_results.get("claim_grounding_ratio")
    n_total   = eval_results.get("claims_total", 0)

    if grounding is not None:
        print(f"  Grader 5 Claim Grounding:  {grounding:.2f}  "
              f"({eval_results.get('claims_supported', 0)}/{n_total} claims supported)")
        unsupported  = eval_results.get("unsupported_claims", [])
        contradicted = eval_results.get("contradicted_claims", [])
        if unsupported:
            print(f"\n  UNSUPPORTED ({len(unsupported)}) -- not in retrieved context:")
            for c in unsupported:
                print(f"    ? {c}")
        if contradicted:
            print(f"\n  CONTRADICTED ({len(contradicted)}) -- context disagrees:")
            for c in contradicted:
                print(f"    x {c}")
        if not unsupported and not contradicted:
            print("  All claims grounded in the retrieved context.")
    elif eval_results.get("decomposition_error"):
        print("  Grader 5 Claim Grounding:  unavailable (decompose step failed)")

    # Grader 6 -- Transcript metrics
    print(f"  Grader 6 Transcript:       "
          f"{eval_results.get('answer_len_tokens', '?')} tokens")

    # Grader 7 -- Coverage check
    uncovered_reqs = eval_results.get("uncovered_requirements", [])
    n_covered   = eval_results.get("requirements_covered", 0)
    n_uncovered = eval_results.get("requirements_uncovered", 0)
    n_total_reqs = n_covered + n_uncovered
    if n_total_reqs > 0:
        print(f"  Grader 7 Coverage:         "
              f"{n_covered}/{n_total_reqs} requirements in prose")
        if uncovered_reqs:
            print(f"    ! Requirements missing from prose ({len(uncovered_reqs)}):")
            for r in uncovered_reqs:
                print(f"      - {r[:80]}{'...' if len(r) > 80 else ''}")
    else:
        print("  Grader 7 Coverage:         n/a (no key_requirements)")

    # Grader 8 -- Retrieval quality distribution
    spread = eval_results.get("retrieval_score_spread")
    if spread is not None:
        print(f"  Grader 8 Retrieval Dist:   "
              f"mean={eval_results.get('retrieval_score_mean', '?'):.3f} "
              f"spread={spread:.3f} | "
              f"weak ratio={eval_results.get('weak_chunks_ratio', '?')}")

    print("---------------------------------\n")


# =============================================================================
# Error tagging
#   retrieval_failure / partial_retrieval / missing_context /
#   misinterpretation / hallucination / contradiction / coverage_gap
# =============================================================================

MISSING_CONTEXT_PHRASE         = "does not contain enough information"

# --- retrieval family thresholds --------------------------------------------
RETRIEVAL_FAIL_GROUNDING_MAX   = 0.34   # grounding at/below this = needle not found
PARTIAL_RETRIEVAL_GROUNDING_MAX = 0.80  # grounded, but evidence incomplete
PARTIAL_RETRIEVAL_WEAK_RATIO   = 0.50   # >= half the retrieved chunks are weak

# --- content family thresholds ----------------------------------------------
MISINTERPRETATION_GROUND_MIN   = 0.60   # answer is well grounded overall...
MISINTERPRETATION_JUDGE_MAX    = 0.70   # ...yet the judge still scores it low
HALLUCINATION_JUDGE_THRESHOLD  = 0.5


def classify_error_tags(answer: str, context_items: list[dict],
                         judge_result: JudgeResult,
                         eval_results: dict | None = None) -> tuple[str, str]:
    """
    Error tagging for the QA export, aligned to the human reviewer's typology.

    Returns (tags, reasons):
      - tags    : comma-separated string of zero or more of 'retrieval_failure',
                  'partial_retrieval', 'missing_context', 'misinterpretation',
                  'hallucination', 'contradiction', 'coverage_gap'. '' if clean.
      - reasons : human-readable ' | '-joined string naming which specific
                  signal(s) triggered each tag.

    ``eval_results`` is the dict assembled by run_eval_suite (containing
    claim_grounding_ratio, claims_unsupported, claims_contradicted,
    requirements_uncovered and weak_chunks_ratio). It is optional for backward
    compatibility: without it, the classifier falls back to the judge score and
    the explicit missing-context phrase alone, and the claim-based content tags
    are skipped.

    Signals used per tag
    --------------------
      retrieval_failure : nothing retrieved, or claim grounding collapsed
                          (<= RETRIEVAL_FAIL_GROUNDING_MAX) -- the answer could
                          not be supported from what was retrieved.
      partial_retrieval : grounded but incomplete -- grounding below
                          PARTIAL_RETRIEVAL_GROUNDING_MAX, or a large share of
                          retrieved chunks are weak, or an unsupported claim
                          sits alongside supported ones.
      missing_context   : the model itself declared the context insufficient
                          (lowest-priority retrieval label).
      misinterpretation : retrieval and grounding are adequate, yet the judge
                          still scores the answer poorly -- the source was read
                          wrongly rather than being missing.
      hallucination     : the answer contains unsupported claims with no
                          retrieval excuse and a low judge score -- fabricated
                          content on otherwise adequate retrieval.
      contradiction     : one or more claims are directly contradicted by the
                          retrieved context (the answer states the opposite of
                          the source). The most serious content failure.
      coverage_gap      : a key requirement the model listed is not actually
                          carried into the answer prose, so a reader of the
                          prose alone would miss it.
    """
    eval_results = eval_results or {}
    tags    = []
    reasons = []

    top_score        = context_items[0]["confidence_score"] if context_items else 0.0
    judge_confidence = judge_result.judge_confidence

    grounding    = eval_results.get("claim_grounding_ratio")
    unsupported  = eval_results.get("claims_unsupported", 0) or 0
    contradicted = eval_results.get("claims_contradicted", 0) or 0
    uncovered    = eval_results.get("requirements_uncovered", 0) or 0
    weak_ratio   = eval_results.get("weak_chunks_ratio")

    explicit_gap = MISSING_CONTEXT_PHRASE in answer.lower()

    # =====================================================================
    # RETRIEVAL FAMILY -- emit at most one, most severe first
    # =====================================================================
    hard_fail = (not context_items) or (
        grounding is not None and grounding <= RETRIEVAL_FAIL_GROUNDING_MAX
    )

    partial = (
        not hard_fail and (
            (grounding is not None and grounding < PARTIAL_RETRIEVAL_GROUNDING_MAX)
            or (weak_ratio is not None and weak_ratio >= PARTIAL_RETRIEVAL_WEAK_RATIO)
            or unsupported > 0
        )
    )

    if hard_fail:
        tags.append("retrieval_failure")
        if not context_items:
            reasons.append("no chunks retrieved")
        else:
            reasons.append(
                f"claim grounding {grounding:.2f} at/below {RETRIEVAL_FAIL_GROUNDING_MAX} "
                f"(top retrieval score {top_score:.3f})"
            )
    elif partial:
        tags.append("partial_retrieval")
        why = []
        if grounding is not None and grounding < PARTIAL_RETRIEVAL_GROUNDING_MAX:
            why.append(f"grounding {grounding:.2f} incomplete")
        if weak_ratio is not None and weak_ratio >= PARTIAL_RETRIEVAL_WEAK_RATIO:
            why.append(f"{weak_ratio:.0%} of chunks weak")
        if unsupported > 0:
            why.append(f"{unsupported} unsupported claim(s)")
        reasons.append("incomplete evidence: " + ", ".join(why))
    elif explicit_gap:
        tags.append("missing_context")
        reasons.append("model explicitly stated the retrieved context was insufficient")

    retrieval_flagged = bool(tags)

    # =====================================================================
    # CONTENT FAMILY -- independent signals, may co-occur
    # =====================================================================

    # ---- contradiction (most serious; always reported if present) -----------
    if contradicted > 0:
        tags.append("contradiction")
        reasons.append(f"{contradicted} claim(s) directly contradicted by the retrieved context")

    # ---- hallucination ------------------------------------------------------
    # Fabricated content: unsupported claims with no retrieval excuse and a low
    # judge score. Distinct from contradiction (which opposes the source) and
    # from partial_retrieval (which is an evidence gap, not fabrication).
    if (not retrieval_flagged and contradicted == 0 and unsupported > 0
            and judge_confidence is not None
            and judge_confidence < HALLUCINATION_JUDGE_THRESHOLD):
        tags.append("hallucination")
        reasons.append(
            f"{unsupported} unsupported claim(s) with judge groundedness "
            f"{judge_confidence:.2f} below {HALLUCINATION_JUDGE_THRESHOLD}"
        )

    # ---- misinterpretation --------------------------------------------------
    # Retrieval and grounding look adequate, yet the judge is unhappy: the
    # source was likely read or applied incorrectly rather than being absent.
    if (not retrieval_flagged
            and grounding is not None and grounding >= MISINTERPRETATION_GROUND_MIN
            and judge_confidence is not None
            and judge_confidence < MISINTERPRETATION_JUDGE_MAX):
        tags.append("misinterpretation")
        reasons.append(
            f"grounding {grounding:.2f} adequate but judge score "
            f"{judge_confidence:.2f} below {MISINTERPRETATION_JUDGE_MAX}"
        )

    # ---- coverage_gap -------------------------------------------------------
    # A listed key requirement never made it into the answer prose, so a
    # reviewer reading the prose alone would miss it.
    if uncovered > 0:
        tags.append("coverage_gap")
        reasons.append(f"{uncovered} key requirement(s) missing from the answer prose")

    return ", ".join(tags), " | ".join(reasons)


# =============================================================================
# Human-in-the-loop
# =============================================================================

def human_review(query: str, answer: str, context_items: list[dict], judge_result: JudgeResult | None = None,
                  error_tags: str = "", error_tag_reasons: str = "") -> dict:
    judge_result   = judge_result or JudgeResult()
    priority_score, no_context = review_priority(context_items, judge_result.judge_confidence,
                                                 judge_result.judge_variance)

    print("\n" + "="*70)
    print("HUMAN-IN-THE-LOOP VALIDATION")
    print("="*70)
    priority_note = "  [no context retrieved]" if no_context else ""
    print(f"Review priority score: {priority_score:.4f}  (lower = needs more scrutiny){priority_note}")
    if judge_result.judge_confidence is not None:
        print(f"LLM judge confidence:  {judge_result.judge_confidence:.2f}  — {judge_result.judge_reasoning}")
    else:
        print(f"LLM judge confidence:  unavailable — {judge_result.judge_reasoning}")
    if judge_result.judge_deepseek_reasoning:
        ds_c = judge_result.judge_deepseek_confidence
        print(f"  ↳ deepseek: {'%.2f' % ds_c if ds_c is not None else 'FAILED'} — {judge_result.judge_deepseek_reasoning}")
    if judge_result.judge_groq_reasoning:
        gq_c = judge_result.judge_groq_confidence
        print(f"  ↳ groq:     {'%.2f' % gq_c if gq_c is not None else 'FAILED'} — {judge_result.judge_groq_reasoning}")
    if error_tags:
        print(f"⚠ Flagged error tags:  {error_tags}")
        if error_tag_reasons:
            print(f"   Reasons: {error_tag_reasons}")
    print("-"*70)
    print(f"QUERY:\n{query}")
    print("-"*70)
    print(f"AI ANSWER:\n{answer}")
    print("-"*70)
    print("Top supporting passages:")
    for item in context_items[:3]:
        print(f"  [{item['confidence_score']:.4f}] {item['source_document']} "
              f"p.{item['page_number']} — {item['sentence_chunk'][:80]}...")
    print("="*70)

    decision = input("\nValidate this answer — [a]pprove / [r]eject / [e]dit / [s]kip for now: ").strip().lower()

    validation = {
        "validation_status": "pending",
        "reviewer"          : None,
        "reviewer_comment"  : None,
        "corrected_answer"  : None,
        "review_priority"   : round(priority_score, 4),
        "no_context"        : no_context,
        "review_timestamp"  : None,
        "judge_confidence"  : judge_result.judge_confidence,
        "judge_reasoning"   : judge_result.judge_reasoning,
        "judge_deepseek_confidence": judge_result.judge_deepseek_confidence,
        "judge_deepseek_reasoning" : judge_result.judge_deepseek_reasoning,
        "judge_groq_confidence"    : judge_result.judge_groq_confidence,
        "judge_groq_reasoning"     : judge_result.judge_groq_reasoning,
    }

    if decision not in ("a", "r", "e"):
        return validation

    reviewer = input("Reviewer initials/ID: ").strip()
    validation["reviewer"]        = reviewer
    validation["review_timestamp"] = datetime.datetime.now().isoformat()

    if decision == "a":
        validation["validation_status"] = "approved"

    elif decision == "r":
        validation["validation_status"] = "rejected"
        validation["reviewer_comment"]  = input("Reason for rejection: ").strip()

    elif decision == "e":
        validation["validation_status"] = "edited"
        print("Enter corrected answer (finish with a blank line):")
        lines = []
        while True:
            line = input()
            if not line:
                break
            lines.append(line)
        validation["corrected_answer"] = "\n".join(lines)
        validation["reviewer_comment"] = input("Comment on correction: ").strip()

    return validation


def save_session(session_log: list[dict], results: list[dict]) -> None:
    timestamp  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("Runs", timestamp)
    os.makedirs(output_dir, exist_ok=True)
    pd.DataFrame(session_log).to_csv(os.path.join(output_dir, "session_transcript.csv"), index=False)

    if results:
        COLUMN_ORDER = [
            "document_title",
            "document_date",
            "query",
            "answer",
            "key_requirements",
            "source_passages_used",
            "confidence_score",
            "source_references",
            "num_chunks_used",
            "error_tags",
            "error_tag_reasons",
            "validation_status",
            "reviewer",
            "reviewer_comment",
            "corrected_answer",
            "review_priority",
            "no_context",
            "review_timestamp",
            "judge_confidence",
            "judge_reasoning",
            "judge_mean",
            "judge_median",
            "judge_lower",
            "judge_upper",
            "judge_variance",
            # Grader 1 -- per-judge breakdown, captured separately for each
            # panel member (see JUDGE_PANEL) so the SME can see WHY each
            # judge scored the way it did, not just the pooled consensus.
            "judge_deepseek_confidence",
            "judge_deepseek_reasoning",
            "judge_groq_confidence",
            "judge_groq_reasoning",
            "semantic_match",
            # Grader 4 -- DeepEval Answer Relevancy  [ADDED]
            "deepeval_answer_relevancy_score",
            "deepeval_answer_relevancy_reason",
            "deepeval_answer_relevancy_passed",
            # Grader 5 -- Claim Decomposition  [ADDED]
            "claim_grounding_ratio",
            "claims_total",
            "claims_supported",
            "claims_unsupported",
            "claims_contradicted",
            "unsupported_claims",
            "contradicted_claims",
            "decomposition_error",
            # Grader 6 -- Transcript / response metrics
            "answer_len_chars",
            "answer_len_tokens",
            # Grader 7 -- Coverage check
            "requirements_covered",
            "requirements_uncovered",
            "uncovered_requirements",
            # Grader 8 -- Retrieval quality distribution
            "retrieval_score_max",
            "retrieval_score_min",
            "retrieval_score_mean",
            "retrieval_score_spread",
            "weak_chunks_ratio",
        ]

        exportable = []
        for r in results:
            row = {k: v for k, v in r.items() if k != "_context_items"}

            if isinstance(row.get("key_requirements"), list):
                row["key_requirements"] = "\n".join(f"- {req}" for req in row["key_requirements"])

            ordered_row = {col: row[col] for col in COLUMN_ORDER if col in row}
            ordered_row.update({k: v for k, v in row.items() if k not in ordered_row})
            exportable.append(ordered_row)

        pd.DataFrame(exportable).to_csv(os.path.join(output_dir, "rag_qa_results.csv"), index=False)

    print(f"\n    Session saved to '{output_dir}'")


# =============================================================================
# Clients (module-level so this file can be imported by app.py without launching
# the interactive session)
# =============================================================================

groq_client = Groq(api_key=groq_api_key)

azure_client = AzureOpenAI(
    api_version=AZURE_OPENAI_API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
)

deepseek_client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
)

_deepeval_model          = AzureDeepEvalWrapper(azure_client, AZURE_OPENAI_DEPLOYMENT)
_answer_relevancy_metric = AnswerRelevancyMetric(threshold=_DEEPEVAL_THRESHOLD, model=_deepeval_model, include_reason=True)

gemini_client = genai.Client(api_key=GOOGLE_API_KEY)


# =============================================================================
# Interactive chat loop  (only runs when this file is executed directly)
# =============================================================================

def run_interactive_session():
    print("################ Starting interactive chat session ################")

    chat_history      = []
    session_log       = []
    results           = []
    hitl_enabled      = True
    active_doc_filter = None

    print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║        FDA Oncology Regulatory Guidance — AI Assistant               ║
╠══════════════════════════════════════════════════════════════════════╣
║  Commands:  'quit'     — end session and save results                ║
║             'history'  — show conversation so far                    ║
║             'save'     — save results to CSV                         ║
║             'clear'    — clear conversation history                  ║
║             'pending'  — review any answers still marked pending     ║
║             'bypass'   — turn OFF human review (auto-pass answers)   ║
║             'validate' — turn human review back ON                   ║
║             'docs'     — list loaded documents and their versions    ║
║             'use <id or name>' — restrict retrieval to one document  ║
║             'all docs' — clear the document filter, search everything║
╚══════════════════════════════════════════════════════════════════════╝

Documents loaded:
""" + "\n".join(f"  • [{e['doc_id']}] {e['source_document']}" for e in DOC_REGISTRY) + """

Ask me anything about the uploaded documents!
""")

    while True:

        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n[INFO] Session interrupted.")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit"):
            print("\n[INFO] Ending session...")
            save_session(session_log, results)
            break

        if user_input.lower() == "history":
            if not chat_history:
                print("\n[INFO] No conversation history yet.\n")
            else:
                print("\n" + "-"*60)
                print("CONVERSATION HISTORY")
                print("-"*60)
                for turn in chat_history:
                    role = "You" if turn["role"] == "user" else "Assistant"
                    print(f"\n{role}:\n{turn['content']}")
                print("-"*60 + "\n")
            continue

        if user_input.lower() == "save":
            save_session(session_log, results)
            continue

        if user_input.lower() == "clear":
            chat_history = []
            print("\n[INFO] Conversation history cleared. Starting fresh.\n")
            continue

        if user_input.lower() == "pending":
            pending_items = [r for r in results if r["validation_status"] == "pending"]
            if not pending_items:
                print("\n[INFO] No pending items awaiting review.\n")
            else:
                print(f"\n[INFO] {len(pending_items)} item(s) pending review.\n")
                pending_items.sort(key=lambda r: r["review_priority"])
                for item in pending_items:
                    judge_result = JudgeResult(
                        judge_confidence=item["judge_confidence"],
                        judge_reasoning=item["judge_reasoning"],
                        judge_variance=item.get("judge_variance"),
                        judge_deepseek_confidence=item.get("judge_deepseek_confidence"),
                        judge_deepseek_reasoning=item.get("judge_deepseek_reasoning"),
                        judge_groq_confidence=item.get("judge_groq_confidence"),
                        judge_groq_reasoning=item.get("judge_groq_reasoning"),
                    )
                    item_llm_answer  = LLMAnswer(**{field: item[field] for field in LLMAnswer.model_fields})
                    item_answer_text = format_answer_for_judge_and_history(item_llm_answer)
                    validation = human_review(item["query"], item_answer_text, item["_context_items"], judge_result,
                                               error_tags=item.get("error_tags", ""),
                                               error_tag_reasons=item.get("error_tag_reasons", ""))
                    item.update(validation)
            continue

        if user_input.lower() == "bypass":
            hitl_enabled = False
            print("\n[INFO] Human review is now OFF. Answers will auto-pass as 'not_reviewed'.\n")
            continue

        if user_input.lower() == "validate":
            hitl_enabled = True
            print("\n[INFO] Human review is now ON.\n")
            continue

        if user_input.lower() == "docs":
            print_doc_registry()
            if active_doc_filter:
                print(f"[INFO] Active filter: searching ONLY '{active_doc_filter}'. Use 'all docs' to clear.\n")
            else:
                print("[INFO] No active filter — searching across all documents.\n")
            continue

        if user_input.lower() in ("all docs", "clear filter", "clear docs"):
            active_doc_filter = None
            print("\n[INFO] Document filter cleared — now searching across all documents.\n")
            continue

        if user_input.lower().startswith("use "):
            target  = user_input[4:].strip()
            matches = find_doc_matches(target)
            if len(matches) == 1:
                active_doc_filter = matches[0]
                print(f"\n[INFO] Now restricting retrieval to: '{active_doc_filter}'. Use 'all docs' to clear.\n")
            elif len(matches) == 0:
                print(f"\n[INFO] No document matched '{target}'. Use 'docs' to see what's loaded.\n")
            else:
                print(f"\n[INFO] '{target}' matched multiple documents — be more specific, or use the doc_id:")
                for m in matches:
                    print(f"    - {m}")
                print()
            continue

        # --- Determine which document(s) to search this turn ---------------------
        doc_filter_this_turn = active_doc_filter
        inferred_note        = None

        if doc_filter_this_turn is None:
            inferred = infer_doc_filter_from_query(user_input)
            if len(inferred) == 1:
                doc_filter_this_turn = inferred[0]
                inferred_note = (f"[INFO] Detected this question likely refers to '{doc_filter_this_turn}' "
                                 f"— searching only that document. (Use 'use <name>' to make this explicit, "
                                 f"or 'all docs' to search everything.)")
            elif len(inferred) > 1:
                inferred_note = (
                    "[INFO] This question could refer to more than one loaded document:\n"
                    + "\n".join(f"    - {m}" for m in inferred)
                    + "\n    Searching across all documents for now — use 'use <id or name>' to narrow it down."
                )

        if inferred_note:
            print(f"\n{inferred_note}\n")

        # --- Retrieve with hybrid query expansion --------------------------------
        print("\n[Retrieving relevant passages with hybrid query expansion...]")
        context_items, retrieval_diagnostics = retrieve_with_confidence(
            user_input, azure_client, AZURE_OPENAI_DEPLOYMENT,
            doc_filter=doc_filter_this_turn, return_diagnostics=True,
        )

        print(f"  Top {len(context_items)} chunks retrieved:")
        for item in context_items:
            print(f"    Score: {item['confidence_score']:.4f} | "
                  f"{item['source_document']} p.{item['page_number']} | "
                  f"{item['sentence_chunk'][:65]}...")

        # --- Build prompt with history -------------------------------------------
        prompt = build_prompt(user_input, context_items, chat_history[-6:])

        # --- Generate answer (main model = Azure OpenAI gpt-5.4-mini) ------------
        print(f"\n[Generating answer with Azure OpenAI ({model_name})...]\n")

        response = azure_client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            messages=[
                {
                    "role"   : "system",
                    "content": "You are a helpful assistant specialising in FDA oncology regulatory guidance documents.",
                },
                {
                    "role"   : "user",
                    "content": prompt,
                },
            ],
            max_completion_tokens=4096,
        )
        raw_content = response.choices[0].message.content.strip()

        # --- Parse the structured JSON output into an LLMAnswer (pydantic) -------
        try:
            llm_answer = parse_llm_answer(raw_content)
        except (json.JSONDecodeError, ValidationError) as e:
            print(f"[WARNING] Failed to parse structured model output as JSON ({e}). "
                  f"Falling back to a placeholder record — raw output preserved in the answer field.")
            llm_answer = LLMAnswer(
                document_title="PARSE_ERROR",
                document_date="unknown",
                answer=(f"The model's response could not be parsed into the expected structured format "
                        f"({e}).\n\nRaw output:\n{raw_content}"),
                key_requirements=[],
                source_passages_used="",
                confidence_score=0.0,
            )

        answer_text = format_answer_for_judge_and_history(llm_answer)

        # --- Print answer --------------------------------------------------------
        print("─"*70)
        print("Assistant:")
        print_llm_answer(llm_answer)
        print("─"*70 + "\n")

        # --- Grader 1: multi-judge consensus -------------------------------------
        print("[Scoring answer confidence with judge panel...]")
        judge_result = judge_answer_confidence(user_input, answer_text, context_items)
        if judge_result.judge_confidence is None:
            print(f"[WARNING] Judge panel failed: {judge_result.judge_reasoning}")
        else:
            print(f"[Judge consensus: mean={judge_result.judge_mean:.2f} "
                  f"median={judge_result.judge_median:.2f} "
                  f"bounds=[{judge_result.judge_lower:.2f}, {judge_result.judge_upper:.2f}] "
                  f"std={judge_result.judge_variance:.2f}]")
            if judge_result.judge_variance is not None and judge_result.judge_variance >= 0.2:
                print(f"    ⚠ High judge disagreement (std={judge_result.judge_variance:.2f}) "
                      f"— flag for SME review.")

        # --- Grader 2: semantic match vs ground-truth context --------------------
        semantic_match = semantic_match_score(llm_answer.answer, context_items)
        if semantic_match is not None:
            print(f"[Semantic match vs source context: {semantic_match:.2f} "
                  f"(embedding cosine, code-based)]")

        # --- Graders 3-5: DeepEval + Claim Decomposition  [ADDED] ---------------
        # Runs after Graders 1 and 2 so all scoring is done before HITL.
        # Results are merged into result_row and printed to the SME before review.
        # Retrieve the sub-queries used this turn from the expansion cache for Grader 6.
        _expansion_key = _hash_key("expand", user_input.strip().lower())
        _sub_queries   = _query_expansion_cache.get(_expansion_key, [user_input])
        eval_results = run_eval_suite(
            user_input, answer_text, context_items, azure_client, model_name,
            raw_answer=llm_answer.answer,
            llm_answer=llm_answer,
            sub_queries=_sub_queries,
            retrieval_diagnostics=retrieval_diagnostics,
        )

        # --- Holistic error tagging ----------------------------------------------
        error_tags, error_tag_reasons = classify_error_tags(
            answer_text, context_items, judge_result, eval_results)
        if error_tags:
            print(f"[Error tag check] Flagged: {error_tags}")
            print(f"    Reasons: {error_tag_reasons}")

        # --- Human-in-the-loop validation ----------------------------------------
        # Show Graders 3-5 summary before the reviewer decides.  [ADDED]
        print_eval_summary(eval_results)
        if hitl_enabled:
            validation = human_review(user_input, answer_text, context_items, judge_result,
                                       error_tags=error_tags, error_tag_reasons=error_tag_reasons)
        else:
            priority_score, no_context = review_priority(context_items, judge_result.judge_confidence,
                                                         judge_result.judge_variance)
            validation = {
                "validation_status": "not_reviewed",
                "reviewer"          : None,
                "reviewer_comment"  : None,
                "corrected_answer"  : None,
                "review_priority"   : round(priority_score, 4),
                "no_context"        : no_context,
                "review_timestamp"  : None,
                "judge_confidence"  : judge_result.judge_confidence,
                "judge_reasoning"   : judge_result.judge_reasoning,
                "judge_deepseek_confidence": judge_result.judge_deepseek_confidence,
                "judge_deepseek_reasoning" : judge_result.judge_deepseek_reasoning,
                "judge_groq_confidence"    : judge_result.judge_groq_confidence,
                "judge_groq_reasoning"     : judge_result.judge_groq_reasoning,
            }

        # --- Update conversation history -----------------------------------------
        history_answer = validation["corrected_answer"] or answer_text
        chat_history.append({"role": "user",      "content": user_input})
        chat_history.append({"role": "assistant", "content": history_answer})

        # --- Log for export ------------------------------------------------------
        source_refs = " | ".join(
            f"{c['source_document']} p.{c['page_number']} (conf: {c['confidence_score']:.4f})"
            for c in context_items
        )

        session_log.append({
            "turn"   : len(session_log) // 2 + 1,
            "role"   : "user",
            "content": user_input,
        })
        session_log.append({
            "turn"              : len(session_log) // 2,
            "role"              : "assistant",
            "content"           : answer_text,
            "validation_status" : validation["validation_status"],
            "error_tags"        : error_tags,
            "error_tag_reasons" : error_tag_reasons,
        })

        # --- Assemble the final structured QA record (pydantic) ------------------
        qa_record = QAResultRecord(
            **llm_answer.model_dump(),
            query             = user_input,
            source_references = source_refs,
            num_chunks_used   = len(context_items),
            reviewer          = validation["reviewer"],
            review_priority   = validation["review_priority"],
            review_timestamp  = validation["review_timestamp"],
            judge_confidence  = validation["judge_confidence"],
            judge_reasoning   = validation["judge_reasoning"],
            judge_mean        = judge_result.judge_mean,
            judge_median      = judge_result.judge_median,
            judge_lower       = judge_result.judge_lower,
            judge_upper       = judge_result.judge_upper,
            judge_variance    = judge_result.judge_variance,
            judge_deepseek_confidence = judge_result.judge_deepseek_confidence,
            judge_deepseek_reasoning  = judge_result.judge_deepseek_reasoning,
            judge_groq_confidence     = judge_result.judge_groq_confidence,
            judge_groq_reasoning      = judge_result.judge_groq_reasoning,
            semantic_match    = semantic_match,
        )

        result_row = qa_record.model_dump()
        result_row.update({
            "error_tags"       : error_tags,
            "error_tag_reasons": error_tag_reasons,
            "no_context"       : validation.get("no_context", False),
            "validation_status": validation["validation_status"],
            "reviewer_comment" : validation["reviewer_comment"],
            "corrected_answer" : validation["corrected_answer"],
            "_context_items"   : context_items,
        })
        # Merge Graders 3-5 results -- skip keys starting with _ (per-claim
        # breakdown is kept in memory only, too verbose for CSV export).  [ADDED]
        result_row.update({k: v for k, v in eval_results.items() if not k.startswith("_")})
        results.append(result_row)


if __name__ == "__main__":
    run_interactive_session()