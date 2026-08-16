from __future__ import annotations

import io
import os
import csv
import json
import argparse
import datetime
import tempfile
import traceback
import re

import numpy as np
import faiss
from flask import Flask, request, jsonify, Response

import Guidance_Pipeline as M

app = Flask(__name__)

SESSION_LOG: list[dict] = []
_ENTRY_SEQ = 0
UPLOAD_DIR = os.path.join(tempfile.gettempdir(), "studio_uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

_DOC_ID_SEQ = 0


# =============================================================================
# Document registration helper (used for uploaded PDFs)
# =============================================================================

def register_pdf(pdf_path: str) -> dict:
    """
    Process a PDF through Main.process_document and splice its chunks +
    embeddings into Main's in-memory index structures (global FAISS index,
    per-document sub-index, chunk list, registry). Returns the new registry
    entry. Mirrors the wiring Main does at startup.
    """
    chunks, embeddings, version_info = M.process_document(
        pdf_path, M.chunk_size, M.min_tokens, M.embedding_model
    )

    base_name = os.path.basename(pdf_path).replace(".pdf", "")

    # Build a unique display label (append version / hash if the name clashes)
    existing_labels = {e["source_document"] for e in M.DOC_REGISTRY}
    final_label = base_name
    if final_label in existing_labels:
        final_label = f"{base_name} [{version_info['version_label']}]"
    if final_label in existing_labels:
        import hashlib
        short = hashlib.sha256(pdf_path.encode()).hexdigest()[:6]
        final_label = f"{final_label} ({short})"

    # Append chunks to the global list, recording their global indices
    start_idx = len(M.all_chunks)
    for c in chunks:
        c["source_document"]    = final_label
        c["doc_version_label"]  = version_info["version_label"]
        c["doc_version_source"] = version_info["version_source"]
        M.all_chunks.append(c)
    end_idx = len(M.all_chunks)
    global_indices = list(range(start_idx, end_idx))

    # Extend the global FAISS index
    if embeddings.shape[0] > 0:
        if M.faiss_index.d != embeddings.shape[1]:
            raise ValueError("Embedding dim mismatch with existing index.")
        M.faiss_index.add(embeddings)
        # keep the concatenated matrix in sync (used to build sub-indices)
        M.embeddings_np = (np.concatenate([M.embeddings_np, embeddings], axis=0)
                           if M.embeddings_np.shape[0] else embeddings)

    # Build a per-document sub-index (so this doc can be searched in isolation)
    sub_index = faiss.IndexFlatIP(M.embedding_dim)
    if embeddings.shape[0] > 0:
        sub_index.add(embeddings)
    M.DOC_FAISS_INDEX[final_label]     = sub_index
    M.DOC_LOCAL_TO_GLOBAL[final_label] = global_indices
    M.DOC_CHUNK_INDICES[final_label]   = global_indices

    # Monotonic doc_id that never collides, even if a doc was removed earlier.
    global _DOC_ID_SEQ
    _DOC_ID_SEQ = max(_DOC_ID_SEQ, len(M.DOC_REGISTRY)) + 1
    doc_id = f"d{_DOC_ID_SEQ}"
    entry = {
        "doc_id"         : doc_id,
        "source_document": final_label,
        "base_name"      : base_name,
        "doc_title"      : version_info.get("doc_title") or base_name,
        "display_date"   : version_info.get("display_date"),
        "version_label"  : version_info["version_label"],
        "version_source" : version_info["version_source"],
        "path"           : pdf_path,
        "num_chunks"     : len(chunks),
    }
    M.DOC_REGISTRY.append(entry)
    return entry


def full_document_text(label: str) -> str:
    """
    Reconstruct a readable full-text for a document from its chunks (the
    pipeline doesn't keep raw page text around, so we stitch the chunks). The
    highlighter locates retrieved spans inside this same text.

    Because chunks overlap (overlap=3), naively concatenating them duplicates
    sentences. We de-duplicate at the SENTENCE level using each chunk's
    `chunk_sentences` list (added in Main), rebuilding clean, non-repeating page
    text. All text passes through M.normalize_ws so that the exact same string
    form is used here and in the stored chunk text -- this is what makes exact
    substring span-location work reliably.
    """
    idxs = M.DOC_CHUNK_INDICES.get(label, [])
    seen_pages: dict[int, list[str]] = {}

    for gi in idxs:
        c = M.all_chunks[gi]
        pg = c["page_number"]
        # Prefer the per-sentence list (clean, de-dupable); fall back to the
        # whole chunk text for any older cached chunk without it.
        sents = c.get("chunk_sentences")
        if not sents:
            sents = [M.normalize_ws(c["sentence_chunk"])]
        bucket = seen_pages.setdefault(pg, [])
        for s in sents:
            s = M.normalize_ws(s)
            if s and s not in bucket:
                bucket.append(s)

    parts = []
    for page in sorted(seen_pages):
        parts.append(f"[Page {page + 1}]")
        parts.append(" ".join(seen_pages[page]))
    return "\n\n".join(parts)


def doc_label_for_id(doc_id: str) -> str | None:
    for e in M.DOC_REGISTRY:
        if e["doc_id"] == doc_id:
            return e["source_document"]
    return None


def _title_for_label(label: str) -> str:
    """Map an internal retrieval label back to the human-readable document title
    (with date if available), for display in the source pane header."""
    for e in M.DOC_REGISTRY:
        if e["source_document"] == label:
            t = e.get("doc_title") or label
            d = e.get("display_date")
            return f"{t} ({d})" if d else t
    return label


def _titles_for_labels(labels: list[str]) -> str:
    """Comma-joined real document titles for the given internal labels, for the
    report header. Falls back to empty string if none resolve."""
    titles = []
    for lbl in labels:
        for e in M.DOC_REGISTRY:
            if e["source_document"] == lbl:
                t = e.get("doc_title") or lbl
                if t not in titles:
                    titles.append(t)
                break
    return "; ".join(titles)


def _dates_for_labels(labels: list[str]) -> str:
    """Comma-joined document dates for the given labels (deduplicated)."""
    dates = []
    for lbl in labels:
        for e in M.DOC_REGISTRY:
            if e["source_document"] == lbl:
                d = e.get("display_date")
                if d and d not in dates:
                    dates.append(d)
                break
    return "; ".join(dates)


# =============================================================================
# Core: answer a question over a chosen set of documents
# =============================================================================

def retrieve_over_docs(question: str, labels: list[str]) -> tuple[list[dict], dict]:
    """Retrieve context for a question over the selected documents.

    Returns (context_items, retrieval_diagnostics). The diagnostics carry the
    per-sub-query chunk-id sets needed by Grader 9 (sub-query agreement). For a
    multi-document search the sets are unioned across the per-label calls, so a
    chunk counts as retrieved by a sub-query if that sub-query surfaced it under
    any selected document.
    """
    all_labels = [e["source_document"] for e in M.DOC_REGISTRY]
    use_global = (not labels) or set(labels) == set(all_labels)

    if use_global:
        items, diagnostics = M.retrieve_with_confidence(
            question, M.azure_client, M.AZURE_OPENAI_DEPLOYMENT, doc_filter=None,
            apply_precision=False, return_diagnostics=True,
        )
        return items, diagnostics

    merged: list[dict] = []
    seen = set()
    merged_per_sub: dict[str, set] = {}
    subquery_count = 0
    for label in labels:
        items, diag = M.retrieve_with_confidence(
            question, M.azure_client, M.AZURE_OPENAI_DEPLOYMENT, doc_filter=label,
            apply_precision=False, return_diagnostics=True,
        )
        subquery_count = max(subquery_count, diag.get("subquery_count", 0))
        # Union each sub-query's retrieved chunk ids across the per-label calls.
        for sq, ids in (diag.get("per_sub_query_ids") or {}).items():
            merged_per_sub.setdefault(sq, set()).update(ids)
        for it in items:
            key = (it["source_document"], it["page_number"], it["sentence_chunk"][:60])
            if key not in seen:
                seen.add(key)
                merged.append(it)
    merged.sort(key=lambda x: x["confidence_score"], reverse=True)
    merged = merged[: M.top_k]

    diagnostics = {
        "subquery_count"   : subquery_count or (len(merged_per_sub) if merged_per_sub else 0),
        "per_sub_query_ids": merged_per_sub,
    }
    return merged, diagnostics


# -----------------------------------------------------------------------------
# Sentence-level highlight refinement  [ADDED]
# -----------------------------------------------------------------------------

EVIDENCE_FALLBACK_ENABLED = True


def _best_doc_for_quote(quote: str, doc_texts: dict[str, str],
                        preferred_labels: list[str]) -> tuple[str | None, dict]:
    """
    Align one verbatim quote against every contributing document and return
    (best_label, alignment). Documents that actually contributed retrieved
    context (preferred_labels) win ties, so a quote is attributed to the
    document it came from rather than a coincidental match elsewhere.
    """
    best_label, best = None, {"start": None, "end": None, "score": 0.0}
    for label, text in doc_texts.items():
        al = M.align_evidence(quote, text)
        if al["start"] is None:
            continue
        # Prefer contributing docs on (near-)ties by nudging their score.
        adj = al["score"] + (1.5 if label in preferred_labels else 0.0)
        best_adj = best["score"] + (1.5 if best_label in preferred_labels else 0.0)
        if adj > best_adj:
            best_label, best = label, al
    return best_label, best


def _split_on_list_markers(sentence: str) -> list[str]:
    """
    PDF extraction sometimes glues a list item onto the previous sentence across
    a bullet marker (e.g. '...relationships.) ‒ In general, ...'). spaCy keeps
    these together, which makes a snapped highlight bleed into the prior item.
    Split such a sentence at internal list markers (‒ – — • *) so each list item
    becomes its own boundary. The leading marker is dropped from each piece.
    """
    # Split BEFORE an internal list marker that is followed by a space + word.
    parts = re.split(r"\s+(?=[‒–—\u2022•\*]\s+\S)", sentence)
    out = []
    for p in parts:
        p = re.sub(r"^\s*[‒–—\u2022•\*]\s+", "", p).strip()
        if p:
            out.append(p)
    return out or [sentence]


def _build_sentence_index(label: str, doc_text: str) -> list[tuple[int, int]]:
    """
    Locate every real (spaCy-produced) sentence from this document's chunks
    inside the reconstructed doc_text, returning sorted, de-duplicated
    (start, end) character spans. These are the ground-truth sentence
    boundaries we snap highlights to -- far more reliable than walking raw
    punctuation, which bleeds across PDF-extracted footnotes and list markers.
    """
    spans: list[tuple[int, int]] = []
    idxs = M.DOC_CHUNK_INDICES.get(label, [])
    seen = set()
    for gi in idxs:
        c = M.all_chunks[gi]
        for raw in c.get("chunk_sentences", []) or []:
            # Break glued list items into separate boundary units.
            for s in _split_on_list_markers(M.normalize_ws(raw)):
                s = M.normalize_ws(s)
                if not s or s in seen:
                    continue
                seen.add(s)
                al = M.align_evidence(s, doc_text)
                if al["start"] is not None:
                    spans.append((al["start"], al["end"]))
    spans.sort()
    # merge exact-duplicate / nested spans
    merged: list[tuple[int, int]] = []
    for s, e in spans:
        if merged and s <= merged[-1][0] and e <= merged[-1][1]:
            continue
        merged.append((s, e))
    return merged


def _snap_to_sentences(sentence_spans: list[tuple[int, int]],
                       start: int, end: int) -> tuple[int, int]:
    """
    Expand [start, end) outward to cover every known sentence span it overlaps,
    giving a clean full-stop-to-full-stop highlight bounded by REAL sentences
    (so it never bleeds into a footnote or the next paragraph).
    Falls back to the original span if no sentence overlaps (defensive).
    """
    lo, hi = None, None
    for s, e in sentence_spans:
        # overlap test between [start,end) and [s,e)
        if s < end and start < e:
            lo = s if lo is None else min(lo, s)
            hi = e if hi is None else max(hi, e)
    if lo is None:
        return start, end
    return min(lo, start), max(hi, end)


def locate_spans(context_items: list[dict], doc_texts: dict[str, str],
                 answer_text: str = "", query_text: str = "",
                 source_evidence: list[str] | None = None) -> list[dict]:
    """
    Build highlight spans from the model's verbatim `source_evidence` quotes.

    For each quote we find its exact (or near-exact) character span in the
    correct document's full text and emit a single tight span. This replaces
    whole-chunk / whole-sentence highlighting -- only the operative wording the
    model actually cited is marked, which is what the SME needs to verify fast.
    """
    source_evidence = source_evidence or []

    # Documents that contributed retrieved context (for tie-breaking + fallback).
    contributing = []
    for it in context_items:
        if it["source_document"] not in contributing:
            contributing.append(it["source_document"])

    spans = []
    seen = set()  # de-dupe identical spans if two quotes overlap
    n = 0

    sentence_index: dict[str, list[tuple[int, int]]] = {}
    def _sent_idx(label: str) -> list[tuple[int, int]]:
        if label not in sentence_index:
            sentence_index[label] = _build_sentence_index(label, doc_texts.get(label, ""))
        return sentence_index[label]

    for quote in source_evidence:
        q = M.normalize_ws(quote)
        if len(q) < M.EVIDENCE_MIN_CHARS:
            continue
        label, al = _best_doc_for_quote(q, doc_texts, contributing)
        if label is None or al["start"] is None:
            continue
        # Widen the tight match out to the full sentence(s) it overlaps, so the
        # SME sees complete sentences rather than a clipped fragment.
        s_start, s_end = _snap_to_sentences(_sent_idx(label), al["start"], al["end"])
        key = (label, s_start, s_end)
        if key in seen:
            continue
        seen.add(key)

        page = _page_for_offset(label, s_start)
        spans.append({
            "id"        : f"ev{n}",
            "item"      : f"ev{n}",
            "doc"       : label,
            "start"     : s_start,
            "end"       : s_end,
            "score"     : round(al["score"] / 100.0, 4),   # 0-1 for the UI
            "match_score": al["score"],                    # 0-100 raw
            "grounding" : M.evidence_grounding_label(al["score"]),
            "quote"     : quote,
            "page"      : page,
        })
        n += 1

    # ---- Fallback: no usable quotes -> mark best sentence of top chunk ------
    if not spans and EVIDENCE_FALLBACK_ENABLED and context_items:
        reference = (answer_text or "").strip() or (query_text or "").strip()
        top = context_items[0]
        label = top["source_document"]
        text = doc_texts.get(label, "")
        sentences = top.get("chunk_sentences") or [M.normalize_ws(top["sentence_chunk"])]
        sentences = [M.normalize_ws(s) for s in sentences if M.normalize_ws(s)]
        if text and sentences:
            # pick the sentence most similar to the answer/query
            if reference:
                sims = _sentence_relevances(sentences, reference)
                best_idx = max(range(len(sentences)), key=lambda k: sims[k])
            else:
                best_idx = 0
            al = M.align_evidence(sentences[best_idx], text)
            if al["start"] is not None:
                s_start, s_end = _snap_to_sentences(
                    _build_sentence_index(label, text), al["start"], al["end"])
                spans.append({
                    "id"        : "ev0",
                    "item"      : "ev0",
                    "doc"       : label,
                    "start"     : s_start,
                    "end"       : s_end,
                    "score"     : round(al["score"] / 100.0, 4),
                    "match_score": al["score"],
                    "grounding" : M.evidence_grounding_label(al["score"]),
                    "quote"     : sentences[best_idx],
                    "page"      : _page_for_offset(label, s_start),
                    "fallback"  : True,
                })

    return spans


def _page_for_offset(label: str, offset: int) -> int:
    """Best-effort page number for a character offset in the reconstructed
    document text, using the '[Page N]' markers full_document_text inserts."""
    text = _DOC_TEXT_CACHE.get(label)
    if not text:
        return 1
    page = 1
    for m in re.finditer(r"\[Page (\d+)\]", text):
        if m.start() <= offset:
            page = int(m.group(1))
        else:
            break
    return page


# Cache of the reconstructed per-document text for the current request, so
# _page_for_offset can map offsets to pages without recomputing.
_DOC_TEXT_CACHE: dict[str, str] = {}


def _sentence_relevances(sentences: list[str], reference: str) -> list[float]:
    """Cosine similarity of each sentence to the reference text (answer/query),
    used only by the no-evidence fallback path."""
    if not sentences or not reference.strip():
        return [0.0] * len(sentences)
    try:
        texts = sentences + [reference]
        emb = M.embedding_model.encode(
            texts, normalize_embeddings=True, convert_to_numpy=True
        ).astype(np.float32)
        ref_vec = emb[-1]
        sent_vecs = emb[:-1]
        sims = sent_vecs @ ref_vec
        return [float(s) for s in sims]
    except Exception:
        return [0.0] * len(sentences)


def answer_question(question: str, doc_ids: list[str], run_judge: bool = True) -> dict:
    labels = [lbl for lbl in (doc_label_for_id(d) for d in doc_ids) if lbl]

    context_items, retrieval_diagnostics = retrieve_over_docs(question, labels)

    # Prepend a synthetic metadata passage so questions like "what is the
    # document title / date?" can be answered from the real PDF title (which is
    # otherwise not part of the body text that got chunked). Only the documents
    # that actually contributed context are described.
    meta_lines = []
    seen_meta = set()
    for it in context_items:
        lbl = it["source_document"]
        if lbl in seen_meta:
            continue
        seen_meta.add(lbl)
        for e in M.DOC_REGISTRY:
            if e["source_document"] == lbl:
                t = e.get("doc_title") or lbl
                d = e.get("display_date")
                meta_lines.append(f"Document title: \"{t}\""
                                  + (f"; document date: {d}." if d else "."))
                break
    if meta_lines:
        meta_chunk = {
            "source_document" : context_items[0]["source_document"],
            "page_number"     : 0,
            "sentence_chunk"  : "Document metadata. " + " ".join(meta_lines),
            "chunk_sentences" : meta_lines,
            "confidence_score": 1.0,
            "_synthetic_meta" : True,   # excluded from grading so it can't inflate scores
        }
        context_items = [meta_chunk] + context_items

    # Build the prompt + generate the structured answer (Azure main model)
    prompt = M.build_prompt(question, context_items, [])
    response = M.azure_client.chat.completions.create(
        model=M.AZURE_OPENAI_DEPLOYMENT,
        messages=[
            {"role": "system",
             "content": "You are a helpful assistant specialising in FDA oncology regulatory guidance documents."},
            {"role": "user", "content": prompt},
        ],
        max_completion_tokens=4096,
    )
    raw_content = response.choices[0].message.content.strip()

    try:
        llm_answer = M.parse_llm_answer(raw_content)
    except Exception as e:
        llm_answer = M.LLMAnswer(
            document_title="PARSE_ERROR", document_date="unknown",
            answer=f"The model output could not be parsed ({e}).\n\nRaw:\n{raw_content}",
            key_requirements=[], source_passages_used="", confidence_score=0.0,
        )

    answer_text = M.format_answer_for_judge_and_history(llm_answer)

    # Grading must run against the REAL retrieved passages only. The synthetic
    # document-metadata chunk (added so title questions are answerable) is not
    # part of the source and would otherwise inflate grounding scores, so we
    # strip it out before every grader.
    grading_context = [c for c in context_items if not c.get("_synthetic_meta")]

    # Grader 1: multi-judge consensus
    judge = M.JudgeResult()
    if run_judge:
        judge = M.judge_answer_confidence(question, answer_text, grading_context)

    # Grader 2: semantic match vs source context
    semantic = M.semantic_match_score(llm_answer.answer, grading_context)

    # Graders 4-8: DeepEval + Claim Decomposition + Code-based metrics
    _expansion_key = M._hash_key("expand", question.strip().lower())
    _sub_queries   = M._query_expansion_cache.get(_expansion_key, [question])
    eval_results = M.run_eval_suite(
        question, answer_text, grading_context, M.azure_client, M.model_name,
        raw_answer=llm_answer.answer,
        llm_answer=llm_answer,
        sub_queries=_sub_queries,
        # Grader 9 receives real per-sub-query diagnostics from the app's
        # retrieval path, so sub-query agreement is computed live.
        retrieval_diagnostics=retrieval_diagnostics,
    )

    # Error tags + review priority
    error_tags, error_reasons = M.classify_error_tags(answer_text, grading_context, judge, eval_results)
    priority, no_context = M.review_priority(
        context_items, judge.judge_confidence, judge.judge_variance
    )

    # Full text of every document that contributed context (for the left pane)
    contributing = []
    for it in context_items:
        if it["source_document"] not in contributing:
            contributing.append(it["source_document"])
    # if nothing retrieved, still show the selected docs
    for lbl in (labels or [e["source_document"] for e in M.DOC_REGISTRY]):
        if lbl not in contributing:
            contributing.append(lbl)

    doc_texts = {lbl: full_document_text(lbl) for lbl in contributing}
    # Populate the module-level cache so span->page mapping works.
    _DOC_TEXT_CACHE.clear()
    _DOC_TEXT_CACHE.update(doc_texts)
    # Highlight ONLY the model's verbatim evidence quotes, aligned to the source
    # text. This is the tight, SME-grade highlight -- just the operative wording
    # the model cited, not the whole retrieved chunk. Falls back to the best
    # sentence of the top chunk only if the model emitted no usable quotes.
    spans = locate_spans(
        context_items, doc_texts,
        answer_text=llm_answer.answer,
        query_text=question,
        source_evidence=getattr(llm_answer, "source_evidence", []) or [],
    )

    return {
        "question"      : question,
        "documents"     : [
            {"label": lbl, "title": _title_for_label(lbl), "text": doc_texts[lbl]}
            for lbl in contributing
        ],
        "spans"         : spans,
        "extraction"    : {
            # Real document title/date from the PDF (registry), not the model's
            # guess -- so the report header always shows the correct title.
            "document_title"      : _titles_for_labels(contributing) or llm_answer.document_title,
            "document_date"       : _dates_for_labels(contributing) or llm_answer.document_date,
            "model_document_title": llm_answer.document_title,
            "answer"              : llm_answer.answer,
            "key_requirements"    : llm_answer.key_requirements,
            "source_evidence"     : getattr(llm_answer, "source_evidence", []) or [],
            "source_passages_used": llm_answer.source_passages_used,
            "model_confidence"    : llm_answer.confidence_score,
        },
        "context_items" : [
            {"id": f"sp{i}", "source_document": it["source_document"],
             "page": it["page_number"] + 1, "score": it["confidence_score"],
             "text": it["sentence_chunk"]}
            for i, it in enumerate(context_items)
        ],
        "grading"       : {
            # Grader 1 — consensus
            "judge_confidence": judge.judge_confidence,
            "judge_mean"      : judge.judge_mean,
            "judge_lower"     : judge.judge_lower,
            "judge_upper"     : judge.judge_upper,
            "judge_variance"  : judge.judge_variance,
            "judge_reasoning" : judge.judge_reasoning,
            "judge_scores"    : judge.judge_scores,
            "judge_reasons"   : judge.judge_reasons,
            # Grader 1 — per-judge breakdown, captured separately for deepseek
            # and groq so the SME can see WHY each judge scored as it did.
            "judge_deepseek_confidence": judge.judge_deepseek_confidence,
            "judge_deepseek_reasoning" : judge.judge_deepseek_reasoning,
            "judge_groq_confidence"    : judge.judge_groq_confidence,
            "judge_groq_reasoning"     : judge.judge_groq_reasoning,
            # Grader 2 — semantic match
            "semantic_match"  : semantic,
            # Graders 4-8 — eval suite
            "deepeval_answer_relevancy_score" : eval_results.get("deepeval_answer_relevancy_score"),
            "deepeval_answer_relevancy_reason": eval_results.get("deepeval_answer_relevancy_reason"),
            "deepeval_answer_relevancy_passed": eval_results.get("deepeval_answer_relevancy_passed"),
            "claim_grounding_ratio"           : eval_results.get("claim_grounding_ratio"),
            "claims_total"                    : eval_results.get("claims_total"),
            "claims_supported"                : eval_results.get("claims_supported"),
            "claims_unsupported"              : eval_results.get("claims_unsupported"),
            "claims_contradicted"             : eval_results.get("claims_contradicted"),
            "claims_meta"                     : eval_results.get("claims_meta"),
            "claims_verifiable"               : eval_results.get("claims_verifiable"),
            "unsupported_claims"              : eval_results.get("unsupported_claims"),
            "contradicted_claims"             : eval_results.get("contradicted_claims"),
            "meta_claims"                     : eval_results.get("meta_claims"),
            "decomposition_error"             : eval_results.get("decomposition_error"),
            "answer_len_chars"                : eval_results.get("answer_len_chars"),
            "answer_len_tokens"               : eval_results.get("answer_len_tokens"),
            "requirements_covered"            : eval_results.get("requirements_covered"),
            "requirements_uncovered"          : eval_results.get("requirements_uncovered"),
            "uncovered_requirements"          : eval_results.get("uncovered_requirements"),
            "retrieval_score_max"             : eval_results.get("retrieval_score_max"),
            "retrieval_score_min"             : eval_results.get("retrieval_score_min"),
            "retrieval_score_mean"            : eval_results.get("retrieval_score_mean"),
            "retrieval_score_spread"          : eval_results.get("retrieval_score_spread"),
            "weak_chunks_ratio"               : eval_results.get("weak_chunks_ratio"),
            # Grader 9 — sub-query agreement
            "subquery_count"                  : eval_results.get("subquery_count"),
            "subquery_agreement_ratio"        : eval_results.get("subquery_agreement_ratio"),
            "subquery_mean_hits"              : eval_results.get("subquery_mean_hits"),
            # Grader 11 — evidence alignment
            "evidence_quotes_total"           : eval_results.get("evidence_quotes_total"),
            "evidence_quotes_aligned"         : eval_results.get("evidence_quotes_aligned"),
            "evidence_alignment_rate"         : eval_results.get("evidence_alignment_rate"),
            "unaligned_evidence"              : eval_results.get("unaligned_evidence"),
            # Grader 12 — context utilisation
            "chunks_total"                    : eval_results.get("chunks_total"),
            "chunks_used"                     : eval_results.get("chunks_used"),
            "context_utilisation"             : eval_results.get("context_utilisation"),
            # Grader 13 — numeric consistency
            "numbers_total"                   : eval_results.get("numbers_total"),
            "numbers_grounded"                : eval_results.get("numbers_grounded"),
            "numeric_consistency"             : eval_results.get("numeric_consistency"),
            "ungrounded_numbers"              : eval_results.get("ungrounded_numbers"),
        },
        "review"        : {
            "priority"    : round(priority, 4),
            "no_context"  : no_context,
            "error_tags"  : error_tags,
            "error_reasons": error_reasons,
        },
    }


# =============================================================================
# Interface (single embedded HTML page)
# =============================================================================

STUDIO_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>FDA Oncology Transformation Studio</title>
<style>
  :root{
    --ink:#0f2b28; --muted:#64748b; --line:#e2e8f0; --line-soft:#eef2f7;
    --bg:#ffffff; --panel:#ffffff; --wash:#f6faf9;
    /* ICON plc brand palette: primary teal (Pantone 3282 C) + white */
    --accent:#008579; --accent-dark:#006b62; --accent-soft:#e4f3f0;
    --mark:#fff3c4; --mark-focus:#ffe27a; --rule:#e11d48;
    --ok:#0f9d58; --warn:#b45309; --bad:#be123c;
    --mono:Calibri,"Segoe UI",Candara,Optima,sans-serif;
    --sans:Calibri,"Segoe UI",Candara,Optima,-apple-system,BlinkMacSystemFont,Arial,sans-serif;
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans);font-size:15px;line-height:1.5}
  .wrap{max-width:1560px;margin:0 auto;padding:26px 28px 40px}
  .eyebrow{font-size:12px;letter-spacing:.12em;font-weight:700;color:var(--accent)}
  h1{font-size:27px;margin:0 0 4px;font-weight:650;letter-spacing:-.01em}
  .sub{color:var(--muted);margin:0 0 18px}
  .topline{display:flex;justify-content:space-between;align-items:center;gap:24px;flex-wrap:wrap;margin-bottom:6px}
  .topctrl{display:flex;align-items:center;gap:14px;flex-wrap:wrap}
  .toggle{display:flex;align-items:center;gap:7px;font-size:12.5px;color:#334155;cursor:pointer;white-space:nowrap}
  .toggle input{margin:0}
  .docchip{display:flex;align-items:center;gap:8px;border:1px solid var(--line);border-radius:999px;
           padding:7px 14px;background:var(--wash);font-size:12.5px;color:#334155}
  .dot{width:8px;height:8px;border-radius:50%;background:var(--ok)}

  .stats{display:grid;grid-template-columns:repeat(5,1fr);border:1px solid var(--line);
         border-radius:10px;overflow:hidden;margin:0 0 16px}
  .stat{padding:16px 18px;display:flex;flex-direction:column;align-items:center;gap:6px;
        border-right:1px solid var(--line);background:var(--panel);text-align:center}
  .stat:last-child{border-right:0}
  .stat b{font-size:24px;font-weight:650;letter-spacing:-.02em;color:var(--ink)}
  .stat span{color:var(--muted);font-size:12.5px}

  /* ---- docs / scope bar ---- */
  /* ---- shell: collapsible left sidebar + main area ---- */
  .shell{display:grid;grid-template-columns:288px 1fr;gap:20px;align-items:start;transition:grid-template-columns .18s ease}
  .shell.collapsed{grid-template-columns:44px 1fr}
  .sidebar{border:1px solid var(--line);border-radius:10px;background:var(--panel);overflow:hidden;
           position:sticky;top:16px;align-self:start}
  .sidebar-head{display:flex;align-items:center;justify-content:space-between;gap:8px;
                padding:12px 14px;border-bottom:1px solid var(--line-soft)}
  .sidebar-head h3{margin:0;font-size:13.5px;font-weight:650;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
  .collapse-btn{padding:4px 9px;border:1px solid var(--line);border-radius:7px;background:#fff;color:var(--muted);
                font-size:12px;line-height:1;cursor:pointer;flex:0 0 auto}
  .collapse-btn:hover{background:var(--accent-soft);color:var(--accent);border-color:var(--accent)}
  .sidebar-body{padding:14px}
  /* When collapsed: hide title + body, rotate the chevron to point right */
  .shell.collapsed .sidebar-head h3{display:none}
  .shell.collapsed .sidebar-body{display:none}
  .shell.collapsed .sidebar-head{justify-content:center;padding:12px 6px}
  .shell.collapsed .collapse-btn{transform:rotate(180deg)}

  .doclist{display:flex;flex-direction:column;gap:8px}
  .docpill{display:flex;align-items:center;gap:7px;border:1px solid var(--line);background:var(--wash);
           border-radius:8px;padding:8px 11px;font-size:12.5px;color:#334155;cursor:pointer;user-select:none}
  .docpill.on{border-color:var(--accent);background:var(--accent-soft);color:var(--accent);font-weight:600}
  .docpill input{margin:0;flex:0 0 auto}
  .docpill span{white-space:normal;line-height:1.35}
  .scopeacts{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:12px}

  .ask{display:flex;gap:10px;margin-bottom:16px;flex-wrap:wrap}
  .ask input[type=text]{flex:1;min-width:320px;padding:12px 14px;border:1px solid var(--line);
        border-radius:8px;font-size:14px;font-family:var(--sans)}
  .ask input[type=text]:focus{outline:2px solid var(--accent-soft);border-color:var(--accent)}
  button{font-family:var(--sans);font-size:13.5px;padding:11px 18px;border-radius:8px;border:1px solid var(--accent);
         background:var(--accent);color:#fff;cursor:pointer;font-weight:550}
  button.ghost{background:#fff;color:var(--ink);border-color:var(--line)}
  button.small{padding:7px 12px;font-size:12.5px}
  button:disabled{opacity:.55;cursor:not-allowed}

  .cols{display:grid;grid-template-columns:1fr 1fr;gap:22px}
  @media(max-width:1180px){.cols{grid-template-columns:1fr}}
  .panelhead{display:flex;align-items:center;justify-content:space-between;margin-bottom:10px;gap:12px;min-height:38px}
  .panelhead h2{font-size:16px;margin:0;font-weight:600}
  .num{font-variant-numeric:tabular-nums;color:var(--muted);font-size:12.5px;margin-right:8px;font-weight:600}
  .hlnav{display:flex;align-items:center;gap:8px}
  .hllabel{font-size:12px;color:var(--muted);font-weight:600;margin-right:2px}
  .hlbtn{padding:4px 12px;border:1px solid var(--line);border-radius:7px;background:#fff;color:var(--accent);
         font-size:15px;font-weight:600;line-height:1;cursor:pointer;min-width:34px}
  .hlbtn:hover:not(:disabled){background:var(--accent-soft);border-color:var(--accent)}
  .hlbtn:disabled{color:#cbd5e1;cursor:default}
  .hlcount{font-size:12.5px;color:var(--muted);min-width:44px;text-align:center;font-variant-numeric:tabular-nums}
  .panel{border:1px solid var(--line);border-radius:10px;background:var(--panel);height:660px;overflow:auto}
  .panel .inner{padding:18px 20px}

  #source{font-family:var(--mono);font-size:14px;line-height:1.72;white-space:pre-wrap;word-break:break-word;color:#1e293b}
  #source .docdiv{color:var(--accent);font-weight:700;display:block;margin:18px 0 8px;font-family:var(--sans);font-size:13.5px}
  #source .docdiv:first-child{margin-top:0}
  mark{background:var(--mark);border-radius:2px;padding:0 1px;cursor:pointer}
  mark.focus{background:var(--mark-focus);text-decoration:underline;text-decoration-color:var(--rule);
             text-decoration-thickness:2px;text-underline-offset:3px}
  /* Grounding-tinted highlights: exact/verbatim = green, fuzzy/partial = amber. */
  mark.grounded{background:#bbf7d0}
  mark.partial{background:#fde68a}
  mark.ungrounded{background:#fecaca}
  /* Clickable verbatim-evidence chips in the report tab. */
  ul.evlist li.evq{cursor:pointer;padding:6px 9px;border-radius:6px;border:1px solid var(--line);
                   list-style:none;margin-left:-18px;margin-bottom:6px;font-family:var(--mono);
                   font-size:13px;line-height:1.5;background:#f8fafc}
  ul.evlist li.evq:hover{border-color:var(--accent);background:var(--accent-soft)}
  ul.evlist li.evq.grounded{border-left:3px solid #16a34a}
  ul.evlist li.evq.partial{border-left:3px solid #d97706}
  ul.evlist li.evq.ungrounded{border-left:3px solid #dc2626}

  .tabs{display:inline-flex;border:1px solid var(--line);border-radius:8px;overflow:hidden}
  .tabs button{border:0;background:#fff;color:#334155;padding:8px 14px;border-radius:0;font-weight:500}
  .tabs button.on{background:var(--accent-soft);color:var(--accent);font-weight:650}

  .xtitle{font-size:18px;font-weight:700;letter-spacing:.01em;margin:0 0 4px}
  .xdate{color:var(--muted);font-size:12.5px;margin-bottom:16px}
  .xhead{color:var(--accent);font-size:13.5px;margin:18px 0 8px;font-weight:650;letter-spacing:.01em}
  .answer{font-size:14px;line-height:1.7;white-space:pre-wrap}
  .confrow{display:flex;align-items:center;justify-content:space-between;gap:12px;
           border:1px solid var(--line);border-radius:8px;padding:9px 14px;margin:0 0 8px}
  .confrow-label{font-size:12.5px;color:var(--muted);font-weight:600}
  .confrow-val{font-size:19px;font-weight:700;font-variant-numeric:tabular-nums}
  .confrow.cb-hi{background:#ecfdf3;border-color:#a7f3d0}.confrow.cb-hi .confrow-val{color:#0f9d58}
  .confrow.cb-mid{background:#fffbeb;border-color:#fde68a}.confrow.cb-mid .confrow-val{color:#b45309}
  .confrow.cb-lo{background:#fef2f2;border-color:#fecaca}.confrow.cb-lo .confrow-val{color:#be123c}
  ul.points{margin:8px 0 0;padding-left:18px}
  ul.points li{margin-bottom:6px}
  .item{display:flex;gap:10px;padding:8px 10px;border-radius:7px;cursor:pointer;border:1px solid transparent}
  .item:hover{background:var(--wash)}
  .item.on{background:var(--accent-soft);border-color:#c7d2fe}
  .item .bullet{color:var(--muted);flex:0 0 30px;text-align:right;font-family:var(--mono);font-size:12px;padding-top:3px}
  .item .body{font-family:var(--mono);font-size:13.5px;line-height:1.6;color:#1e293b}
  .item .meta{margin-top:4px;font-family:var(--sans);font-size:11.5px;color:var(--muted)}
  pre.json{font-family:var(--mono);font-size:12.5px;white-space:pre-wrap;word-break:break-word;margin:0;color:#334155}

  .grades{display:grid;grid-template-columns:repeat(2,1fr);gap:10px;margin:6px 0 4px}
  .grade{border:1px solid var(--line);border-radius:8px;padding:10px 12px;background:var(--wash);text-align:center;position:relative}
  .grade .k{font-size:11.5px;color:var(--muted);letter-spacing:.01em;padding:0 22px}
  .grade .v{font-size:20px;font-weight:650;letter-spacing:-.01em}
  .grade .r{font-size:11.5px;color:#475569;margin-top:3px}
  .barwrap{height:6px;background:#e2e8f0;border-radius:3px;margin-top:7px;overflow:hidden}
  .bar{height:100%;background:var(--accent);border-radius:3px}

  .footer{margin-top:18px;border:1px solid var(--line);border-radius:10px;background:var(--wash);
          padding:12px 18px;display:flex;gap:22px;align-items:center;flex-wrap:wrap;font-size:12.5px;color:#475569}
  .footer label{display:flex;align-items:center;gap:7px;cursor:pointer}
  .diag{font-family:var(--mono);font-size:12.5px;color:var(--muted)}
  .empty{color:var(--muted);padding:40px 4px;text-align:center;font-size:13px}
  .spin{display:inline-block;width:13px;height:13px;border:2px solid #c7d2fe;border-top-color:var(--accent);
        border-radius:50%;animation:s .7s linear infinite;vertical-align:-2px;margin-right:8px}
  @keyframes s{to{transform:rotate(360deg)}}
  .note{background:#fffbeb;border:1px solid #fde68a;color:#92400e;border-radius:8px;padding:9px 12px;margin-bottom:14px;font-size:12.5px}

  /* ---- ICON plc brand bar ---- */
  .brandbar{background:var(--accent);color:#fff}
  .brandbar-inner{max-width:1560px;margin:0 auto;padding:11px 28px;display:flex;align-items:center;gap:14px}
  .brandlogo{display:flex;align-items:center;line-height:0}
  .brandlogo svg{height:30px;width:auto;display:block}
  .brandbar-div{width:1px;height:20px;background:rgba(255,255,255,.4)}
  .brandbar-tag{font-size:12.5px;letter-spacing:.02em;opacity:.95;font-weight:500}
  h1{color:var(--ink)}

  /* Branded touches: section numbers + stat strip pick up the ICON teal. */
  .num{color:var(--accent)!important}
  .stats{border-top:3px solid var(--accent)}
  #go:hover{background:var(--accent-dark);border-color:var(--accent-dark)}

  /* Consensus-judge toggle sitting inline in the ask row. */
  .ask .toggle{border:1px solid var(--line);border-radius:8px;padding:11px 14px;background:#fff;
               font-size:13px;color:#334155;white-space:nowrap}

  /* ---- Export dropdown ---- */
  .exportwrap{position:relative;display:inline-block}
  .export-btn{display:inline-flex;align-items:center;gap:7px}
  .export-btn svg{width:15px;height:15px;stroke:currentColor}
  .export-menu{position:absolute;right:0;top:calc(100% + 6px);background:#fff;border:1px solid var(--line);
    border-radius:10px;box-shadow:0 10px 28px rgba(15,43,40,.14);padding:6px;min-width:184px;z-index:50;display:none}
  .export-menu.open{display:block}
  .export-menu button{display:flex;align-items:center;justify-content:space-between;width:100%;background:#fff;
    color:var(--ink);border:0;border-radius:7px;padding:9px 11px;font-size:13px;text-align:left;cursor:pointer;font-weight:500}
  .export-menu button:hover{background:var(--accent-soft);color:var(--accent-dark)}
  .export-menu .fmt-ext{color:var(--muted);font-size:11.5px;font-variant-numeric:tabular-nums}

  /* ---- Human-in-the-loop validation card ---- */
  .review-card{margin-top:18px;border:1px solid var(--line);border-top:3px solid var(--accent);
               border-radius:10px;background:var(--panel);overflow:hidden}
  .review-head{display:flex;align-items:center;justify-content:space-between;gap:12px;padding:14px 18px;
               border-bottom:1px solid var(--line-soft);background:var(--wash)}
  .review-head h2{font-size:16px;margin:0;font-weight:600}
  .review-status{font-size:12px;font-weight:650;padding:5px 12px;border-radius:999px;
                 background:#eef2f7;color:#475569;border:1px solid var(--line)}
  .review-status.s-approved{background:#e6f6ee;color:#0f7a43;border-color:#a7e0c0}
  .review-status.s-accepted{background:var(--accent-soft);color:var(--accent-dark);border-color:#9fd8cf}
  .review-status.s-rejected{background:#fdeaec;color:#b0233a;border-color:#f2b9c1}
  .review-status.s-skipped{background:#eef2f7;color:#475569;border-color:#dbe3ec}
  .review-body{padding:16px 18px;display:flex;flex-direction:column;gap:12px}
  .review-fields{display:grid;grid-template-columns:230px 1fr;gap:12px}
  @media(max-width:720px){.review-fields{grid-template-columns:1fr}}
  .review-fields input,#correctedAnswer{padding:10px 12px;border:1px solid var(--line);border-radius:8px;
    font-size:13.5px;font-family:var(--sans);width:100%;color:var(--ink)}
  .review-fields input:focus,#correctedAnswer:focus{outline:2px solid var(--accent-soft);border-color:var(--accent)}
  #correctedAnswer{min-height:92px;line-height:1.6;resize:vertical}
  .review-hint{font-size:11.5px;color:var(--muted);margin-top:-4px}
  .review-actions{display:flex;gap:10px;flex-wrap:wrap}
  .rv{border-radius:8px;font-size:13.5px;font-weight:600;padding:10px 18px;cursor:pointer;border:1px solid}
  .rv-approve{background:#0f9d58;border-color:#0f9d58;color:#fff}
  .rv-approve:hover{background:#0c8249;border-color:#0c8249}
  .rv-accept{background:var(--accent);border-color:var(--accent);color:#fff}
  .rv-accept:hover{background:var(--accent-dark);border-color:var(--accent-dark)}
  .rv-reject{background:#e11d48;border-color:#e11d48;color:#fff}
  .rv-reject:hover{background:#be123c;border-color:#be123c}
  .rv-skip{background:#fff;border-color:var(--line);color:#334155}
  .rv-skip:hover{background:var(--wash)}

  /* ---- Prominent / highlighted controls ---- */
  /* Sidebar expand/collapse chevron: brand-filled so it's easy to spot. */
  .collapse-btn{background:var(--accent);color:#fff;border-color:var(--accent);font-size:13px;padding:5px 10px;
                box-shadow:0 1px 3px rgba(0,133,121,.25)}
  .collapse-btn:hover{background:var(--accent-dark);color:#fff;border-color:var(--accent-dark)}

  /* Question box: bolder brand border + soft glow so it draws the eye. */
  .q-highlight{border:2px solid var(--accent)!important;background:#fbfffe;
               box-shadow:0 0 0 3px rgba(0,133,121,.06)}
  .q-highlight:focus{outline:3px solid var(--accent-soft);border-color:var(--accent-dark)!important}

  /* Upload button: primary teal action inside the document library. */
  .upload-btn{display:flex;align-items:center;justify-content:center;gap:8px;width:100%;
    background:var(--accent);color:#fff;border:1px solid var(--accent);border-radius:8px;
    padding:11px 14px;font-size:13.5px;font-weight:650;cursor:pointer;margin-bottom:12px;
    box-shadow:0 2px 7px rgba(0,133,121,.24)}
  .upload-btn:hover{background:var(--accent-dark);border-color:var(--accent-dark)}
  .upload-btn svg{width:16px;height:16px;flex:0 0 auto}

  /* Per-judge scores as equal-width horizontal chips (wrap, never scroll). */
  .judgechips{display:flex;gap:10px;flex-wrap:wrap;margin-top:8px}
  .judgechip{flex:1 1 0;min-width:150px;border:1px solid var(--line);border-radius:8px;background:var(--wash);
    padding:9px 12px;display:flex;flex-direction:column;align-items:center;text-align:center;gap:3px}
  .judgechip .jk{font-size:11px;color:var(--muted);font-family:var(--mono);word-break:break-word;line-height:1.3}
  .judgechip .jv{font-size:19px;font-weight:650;color:var(--ink);font-variant-numeric:tabular-nums}

  /* ---- Info "(i)" buttons + hover tooltip (tooltip rendered at document level) ---- */
  /* Pinned to the top-right corner of each grade box for a consistent position. */
  .help{position:absolute;top:8px;right:8px;display:inline-flex;align-items:center;justify-content:center;
    width:17px;height:17px;border-radius:50%;border:1px solid var(--line);color:var(--muted);
    font-size:11px;font-weight:700;font-style:italic;font-family:Georgia,"Times New Roman",serif;
    line-height:1;cursor:help;background:#fff}
  .help:hover{background:var(--accent-soft);color:var(--accent-dark);border-color:var(--accent)}
  .help-tip{position:fixed;background:var(--ink);color:#fff;padding:7px 10px;border-radius:7px;font-size:11.5px;
    font-weight:500;line-height:1.4;max-width:240px;text-align:left;z-index:200;pointer-events:none;
    box-shadow:0 8px 22px rgba(15,43,40,.25);opacity:0;transition:opacity .12s}
  .help-tip.show{opacity:1}

  /* ---- Report tab: compact review action bar + small confidence chip ---- */
  .report-actionbar{display:flex;align-items:center;justify-content:flex-end;gap:12px;flex-wrap:wrap;margin-bottom:12px}
  .mini-review{display:flex;gap:6px;flex-wrap:wrap;margin-right:auto}
  .rv-mini{border-radius:7px;font-size:12px;font-weight:600;padding:6px 11px;cursor:pointer;border:1px solid;line-height:1}
  .rv-mini.m-approve{background:#0f9d58;border-color:#0f9d58;color:#fff}
  .rv-mini.m-approve:hover{background:#0c8249;border-color:#0c8249}
  .rv-mini.m-accept{background:var(--accent);border-color:var(--accent);color:#fff}
  .rv-mini.m-accept:hover{background:var(--accent-dark);border-color:var(--accent-dark)}
  .rv-mini.m-reject{background:#e11d48;border-color:#e11d48;color:#fff}
  .rv-mini.m-reject:hover{background:#be123c;border-color:#be123c}
  .rv-mini.m-skip{background:#fff;border-color:var(--line);color:#334155}
  .rv-mini.m-skip:hover{background:var(--wash)}
  .rv-mini:disabled{opacity:.55;cursor:not-allowed}
  /* Selected decision: dark ring on both the mini and full buttons. */
  .rv.selected,.rv-mini.selected{outline:3px solid var(--ink);outline-offset:2px}

  .conf-chip{display:inline-flex;align-items:center;gap:10px;border:1px solid var(--line);border-radius:8px;
             padding:6px 12px;background:var(--wash);font-variant-numeric:tabular-nums;white-space:nowrap}
  .conf-chip-label{font-size:11.5px;color:var(--muted);font-weight:600}
  .conf-chip-val{font-size:16px;font-weight:700}
  .conf-chip.cb-hi{background:#ecfdf3;border-color:#a7f3d0}.conf-chip.cb-hi .conf-chip-val{color:#0f9d58}
  .conf-chip.cb-mid{background:#fffbeb;border-color:#fde68a}.conf-chip.cb-mid .conf-chip-val{color:#b45309}
  .conf-chip.cb-lo{background:#fef2f2;border-color:#fecaca}.conf-chip.cb-lo .conf-chip-val{color:#be123c}

  /* ---- Error logs tab ---- */
  .tabbadge{display:none;margin-left:6px;min-width:16px;height:16px;line-height:16px;text-align:center;
    border-radius:999px;background:var(--bad);color:#fff;font-size:10.5px;font-weight:700;padding:0 4px;
    vertical-align:middle}
  .tabbadge.on{display:inline-block}
  .errlist{display:flex;flex-direction:column;gap:10px}
  .errcard{border:1px solid var(--line);border-left-width:4px;border-radius:8px;padding:11px 14px;background:var(--wash)}
  .errcard.err-bad{border-left-color:var(--bad);background:#fef2f2}
  .errcard.err-warn{border-left-color:var(--warn);background:#fffbeb}
  .errcard-head{display:flex;align-items:center;gap:8px;font-weight:650;font-size:14px;color:var(--ink)}
  .errcard.err-bad .errcard-head{color:#b0233a}
  .errcard.err-warn .errcard-head{color:#92400e}
  .errdot{width:9px;height:9px;border-radius:50%;background:currentColor;flex:0 0 auto}
  .errtag{margin-left:auto;font-family:var(--mono);font-size:11px;font-weight:600;color:var(--muted);
    background:#fff;border:1px solid var(--line);border-radius:5px;padding:2px 7px}
  .errcard-desc{font-size:12.5px;color:#475569;margin-top:5px;line-height:1.5}
  .errcard-reason{font-size:12px;color:#64748b;margin-top:5px;font-family:var(--mono);word-break:break-word}
  .err-ok{border:1px solid #a7e0c0;background:#e6f6ee;color:#0f7a43;border-radius:8px;padding:14px 16px;
    font-size:13.5px;font-weight:500}

  /* ---- Batch question modal ---- */
  .modal-overlay{display:none;position:fixed;inset:0;background:rgba(15,43,40,.45);z-index:50;
    align-items:flex-start;justify-content:center;padding:5vh 20px}
  .modal-overlay.open{display:flex}
  .modal{background:#fff;border-radius:12px;width:100%;max-width:720px;max-height:90vh;
    display:flex;flex-direction:column;box-shadow:0 20px 50px rgba(0,0,0,.25)}
  .modal-head{display:flex;align-items:center;justify-content:space-between;padding:16px 20px;
    border-bottom:1px solid var(--line)}
  .modal-head h3{margin:0;font-size:16px;font-weight:650}
  .modal-close{background:none;border:none;color:var(--muted);font-size:22px;line-height:1;
    padding:2px 6px;cursor:pointer}
  .modal-close:hover{color:var(--ink)}
  .modal-body{padding:16px 20px;overflow-y:auto}
  .modal-hint{margin:0 0 10px;font-size:12.5px;color:var(--muted)}
  #batchText{width:100%;min-height:120px;padding:10px 12px;border:1px solid var(--line);
    border-radius:8px;font-size:13.5px;font-family:var(--sans);resize:vertical;box-sizing:border-box}
  #batchText:focus{outline:2px solid var(--accent-soft);border-color:var(--accent)}
  .batch-filerow{display:flex;align-items:center;gap:10px;margin-top:10px}
  .upload-btn.small{width:auto;margin-bottom:0;padding:8px 14px;font-size:12.5px;font-weight:600;
    box-shadow:none}
  .batch-filename{font-size:12px;color:var(--muted)}
  .batch-progress{display:flex;align-items:center;gap:10px;margin-top:14px}
  .batch-progress-bar{flex:1;height:8px;border-radius:999px;background:var(--wash);
    border:1px solid var(--line);overflow:hidden}
  .batch-progress-bar > div{height:100%;background:var(--accent);width:2%;transition:width .25s ease}
  .batch-progress span{font-size:12px;color:var(--muted);white-space:nowrap;font-variant-numeric:tabular-nums}
  .batch-results{margin-top:16px}
  .batch-table{width:100%;border-collapse:collapse;font-size:12.5px}
  .batch-table th{text-align:left;color:var(--muted);font-weight:600;font-size:11.5px;
    text-transform:uppercase;letter-spacing:.03em;padding:6px 8px;border-bottom:1px solid var(--line)}
  .batch-table td{padding:8px;border-bottom:1px solid var(--line-soft);vertical-align:top}
  .batch-table tr.batch-err td{color:var(--bad)}
  .batch-table td:nth-child(1){color:var(--muted);font-variant-numeric:tabular-nums}
  .batch-table td:nth-child(3){color:#475569}
  .modal-foot{display:flex;justify-content:flex-end;gap:10px;padding:14px 20px;
    border-top:1px solid var(--line)}
</style>
</head>
<body>
<div class="brandbar">
  <div class="brandbar-inner">
    <span class="brandlogo" aria-label="ICON plc">
      <svg width="138" height="38" viewBox="0 0 138 38" xmlns="http://www.w3.org/2000/svg">
        <g transform="translate(0.000000, -6.000000)">
          <g transform="translate(0.000000, 6.000000)">
            <path d="M119.142405,0.379586207 C111.826326,0.379586207 105.512148,4.59475862 102.43488,10.7244138 C99.3576129,4.59475862 93.0420584,0.379586207 85.7273554,0.379586207 C78.3383356,0.379586207 71.9691079,4.68303448 68.9276228,10.9147586 C65.8875139,4.68303448 59.5169099,0.379586207 52.1292663,0.379586207 C44.8131871,0.379586207 38.4962564,4.59751724 35.4189891,10.7285517 C32.343098,4.59751724 26.0261673,0.379586207 18.7087119,0.379586207 C8.37729604,0.379586207 0.000137623762,8.77544828 0.000137623762,19.1313103 C0.000137623762,29.4871724 8.37729604,37.8830345 18.7087119,37.8830345 C26.0261673,37.8830345 32.343098,33.6664828 35.4189891,27.534069 C38.4962564,33.6664828 44.8131871,37.8830345 52.1292663,37.8830345 C59.5169099,37.8830345 65.8875139,33.5809655 68.9276228,27.3478621 C71.9691079,33.5809655 78.3383356,37.8830345 85.7273554,37.8830345 C93.0420584,37.8830345 99.3576129,33.6678621 102.43488,27.5382069 C105.512148,33.6678621 111.826326,37.8830345 119.142405,37.8830345 C129.476573,37.8830345 137.852355,29.4871724 137.852355,19.1313103 C137.852355,8.77544828 129.476573,0.379586207 119.142405,0.379586207" fill="#FFFFFF"></path>
            <path d="M85.8009842,25.2907586 C82.3314891,25.2852414 79.5294693,22.4742069 79.5225881,19.0011034 C79.5294693,15.528 82.3314891,12.7197241 85.8009842,12.7114483 C89.2622218,12.7197241 92.0669941,15.528 92.072499,19.0011034 C92.0669941,22.4742069 89.2622218,25.2852414 85.8009842,25.2907586 M85.7996079,8.08524138 C79.782697,8.08524138 74.9053109,12.9721379 74.9053109,19.0011034 C74.9053109,25.030069 79.782697,29.9169655 85.7996079,29.9169655 C91.8096376,29.9169655 96.6911525,25.030069 96.6911525,19.0011034 C96.6911525,12.9721379 91.8096376,8.08524138 85.7996079,8.08524138" fill="#008579"></path>
            <path d="M110.591978,29.9943448 L110.399305,29.9943448 L110.399305,10.5653793 C110.403434,8.9902069 111.454879,8.0577931 112.815978,8.05503448 C113.468315,8.06744828 114.390394,8.51434483 114.834919,9.11296552 L123.948364,21.2977931 L123.948364,10.5653793 C123.949741,8.9902069 125.001186,8.0577931 126.363661,8.05503448 C127.723384,8.0577931 128.773453,8.9902069 128.776206,10.5653793 L128.776206,27.4384828 C128.773453,29.0136552 127.723384,29.946069 126.363661,29.9474483 C125.708572,29.9377931 124.814018,29.484 124.343345,28.8922759 L115.231275,16.8508966 L115.231275,29.9943448 L110.591978,29.9943448 Z" fill="#008579"></path>
            <path d="M59.5893,24.2757241 C59.1502802,23.6012414 58.4139931,23.2164138 57.6611911,23.2164138 C57.2524485,23.2136552 56.8340723,23.3295172 56.4652406,23.5708966 L56.7157158,23.9557241 L56.4088149,23.6136552 C55.2926861,24.6164138 53.8132307,25.2274483 52.2016564,25.2274483 C48.7335376,25.221931 45.9315178,22.4122759 45.9246366,18.9391724 C45.9315178,15.4674483 48.7335376,12.6564138 52.2016564,12.6495172 C54.5247455,12.6481379 56.6386465,13.884 57.7189931,15.746069 L58.0217653,16.2729655 L61.2352802,12.9515862 L61.0068248,12.6398621 C59.0291713,9.91848276 55.8197851,8.14606897 52.2016564,8.14606897 C46.1861218,8.14606897 41.3073594,13.0343448 41.3073594,19.0633103 C41.3073594,25.0922759 46.1861218,29.9777931 52.2016564,29.9777931 C54.7958644,29.9777931 57.1905178,29.0702069 59.0677059,27.5502069 C59.6938941,27.141931 60.0599733,26.4136552 60.0599733,25.6715862 C60.0599733,25.2495172 59.9429931,24.8205517 59.6938941,24.4398621 L59.5893,24.2757241 Z" fill="#008579"></path>
            <path d="M18.6690762,7.72606897 L18.4750267,7.72606897 C17.2240267,7.72606897 16.2111158,8.74262069 16.2111158,9.99641379 L16.2111158,30.2612414 L20.9329871,30.2612414 L20.9329871,9.99641379 C20.9329871,8.74262069 19.9187,7.72606897 18.6690762,7.72606897" fill="#008579"></path>
          </g>
        </g>
      </svg>
    </span>
    <span class="brandbar-div"></span>
    <span class="brandbar-tag">AI Quality Assurance &middot; Oncology Regulatory Guidance</span>
  </div>
</div>
<div class="wrap">

  <div class="topline">
    <div>
      <div class="eyebrow">TRANSFORMATION STUDIO</div>
      <h1>FDA Document Studio</h1>
    </div>
    <div class="topctrl">
      <div class="exportwrap">
        <button class="ghost export-btn" id="exportBtn" title="Export the QA audit trail">
          <svg viewBox="0 0 24 24" fill="none" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M12 3v12"></path><path d="M8 11l4 4 4-4"></path><path d="M4 17v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2"></path>
          </svg>Export
        </button>
        <div class="export-menu" id="exportMenu">
          <button data-fmt="csv">CSV <span class="fmt-ext">.csv</span></button>
          <button data-fmt="xlsx">Excel <span class="fmt-ext">.xlsx</span></button>
          <button data-fmt="json">JSON <span class="fmt-ext">.json</span></button>
        </div>
      </div>
      <div class="docchip"><span class="dot"></span><span id="modelchip">loading...</span></div>
    </div>
  </div>

  <div class="shell" id="shell">
    <aside class="sidebar" id="sidebar">
      <div class="sidebar-head">
        <h3 id="sidebarTitle">Document library</h3>
        <button class="collapse-btn" id="collapseBtn" title="Collapse panel" aria-label="Collapse document library">&#10094;</button>
      </div>
      <div class="sidebar-body" id="sidebarBody">
        <label class="upload-btn" title="Upload one or more PDF documents">
          <input type="file" id="file" accept="application/pdf" style="display:none" multiple>
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
            <path d="M12 16V4"></path><path d="M8 8l4-4 4 4"></path><path d="M4 16v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2"></path>
          </svg>
          <span id="uplabel">Upload PDF(s)</span>
        </label>
        <div class="scopeacts">
          <button class="ghost small" id="selAll">Select all</button>
          <button class="ghost small" id="selNone">Clear</button>
        </div>
        <div class="doclist" id="doclist"><span class="empty" style="padding:6px">No documents yet. Upload PDFs to begin.</span></div>
      </div>
    </aside>

    <div class="main">
      <div class="stats">
        <div class="stat"><b id="s-items">0</b><span>Context passages</span></div>
        <div class="stat"><b id="s-judge">-</b><span>Judge consensus</span></div>
        <div class="stat"><b id="s-sem">-</b><span>Semantic match</span></div>
        <div class="stat"><b id="s-relevancy">-</b><span>Answer relevancy</span></div>
        <div class="stat"><b id="s-grounding">-</b><span>Claim grounding</span></div>
      </div>

      <div class="ask">
        <input type="text" id="q" class="q-highlight" placeholder="Ask a question, e.g. What are the eligibility criteria for RTOR?" autocomplete="off">
        <button id="go">Ask</button>
        <button type="button" class="ghost" id="batchBtn" title="Ask multiple questions at once">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" style="width:15px;height:15px;vertical-align:-2px;margin-right:5px">
            <path d="M8 6h13"></path><path d="M8 12h13"></path><path d="M8 18h13"></path>
            <path d="M3 6h.01"></path><path d="M3 12h.01"></path><path d="M3 18h.01"></path>
          </svg>Batch
        </button>
        <label class="toggle" title="Run the multi-judge consensus grader on each answer">
          <input type="checkbox" id="useJudge" checked> Run consensus judge
        </label>
        <label class="toggle" title="Show the human-in-the-loop validation panel for each answer">
          <input type="checkbox" id="useHitl" checked> Human-in-the-loop
        </label>
      </div>

      <div id="note-slot"></div>

      <div class="cols">
        <section>
          <div class="panelhead">
            <h2><span class="num">01</span>Source document(s)</h2>
            <div class="hlnav" id="hlnav">
              <span class="hllabel">Highlights</span>
              <button type="button" id="hlprev" class="hlbtn" title="Previous highlight" aria-label="Previous highlight" disabled>&#8592;</button>
              <span id="hlcount" class="hlcount">0 / 0</span>
              <button type="button" id="hlnext" class="hlbtn" title="Next highlight" aria-label="Next highlight" disabled>&#8594;</button>
            </div>
          </div>
          <div class="panel"><div class="inner"><div id="source"><div class="empty">Upload documents and ask a question. The source text with highlighted spans appears here.</div></div></div></div>
        </section>

        <section>
          <div class="panelhead">
            <h2><span class="num">02</span>Structured extraction</h2>
            <div class="tabs">
              <button class="on" data-tab="report">Report</button>
              <button data-tab="grading">Grading</button>
              <button data-tab="passages">Passages</button>
              <button data-tab="errors">Error logs<span class="tabbadge" id="errBadge"></span></button>
            </div>
          </div>
          <div class="panel"><div class="inner">
            <div id="tab-report"><div class="empty">The structured answer appears here.</div></div>
            <div id="tab-passages" style="display:none"></div>
            <div id="tab-grading" style="display:none"></div>
            <div id="tab-errors" style="display:none"></div>
          </div></div>
        </section>
      </div>

      <div class="review-card" id="reviewCard" style="display:none">
        <div class="review-head">
          <h2><span class="num">03</span>Human-in-the-loop validation</h2>
          <span class="review-status" id="reviewStatus">Awaiting review</span>
        </div>
        <div class="review-body">
          <div class="review-fields">
            <input type="text" id="reviewer" placeholder="Reviewer initials / ID" autocomplete="off">
            <input type="text" id="reviewComment" placeholder="Comment (optional, e.g. reason for rejection)" autocomplete="off">
          </div>
          <textarea id="correctedAnswer" placeholder="Corrected answer: edit here before choosing Edit"></textarea>
          <div class="review-hint">Approve to sign off as-is &middot; Edit to save your corrected answer &middot; Reject to flag for rework &middot; Skip to defer. Your decision is written to the audit trail and included in every export.</div>
          <div class="review-actions">
            <button class="rv rv-approve" data-status="approved">&#10003; Approve</button>
            <button class="rv rv-accept"  data-status="accepted">&#9998; Edit</button>
            <button class="rv rv-reject"  data-status="rejected">&#10007; Reject</button>
            <button class="rv rv-skip"     data-status="skipped">&#8631; Skip</button>
          </div>
        </div>
      </div>
    </div>
  </div>

  <div class="footer">
    <span>Main model: <b id="f-model">-</b></span>
    <span>Judge panel: <b id="f-judge">-</b></span>
    <span class="diag" id="f-diag"></span>
  </div>
</div>

<div class="modal-overlay" id="batchOverlay">
  <div class="modal">
    <div class="modal-head">
      <h3>Ask multiple questions</h3>
      <button type="button" class="modal-close" id="batchClose" aria-label="Close">&times;</button>
    </div>
    <div class="modal-body">
      <p class="modal-hint">One question per line. Each runs against the same document scope as the main panel <span id="batchDocCount"></span> and is logged to the audit trail / export just like a single question.</p>
      <textarea id="batchText" placeholder="What are the eligibility criteria for RTOR?&#10;What is the review timeline?&#10;What safety data is required at submission?"></textarea>
      <div class="batch-filerow">
        <label class="upload-btn small" title="Upload a .txt or .csv file with one question per line">
          <input type="file" id="batchFile" accept=".txt,.csv" style="display:none">
          <span>Upload .txt / .csv</span>
        </label>
        <span class="batch-filename" id="batchFileName"></span>
      </div>
      <div class="batch-progress" id="batchProgress" style="display:none">
        <div class="batch-progress-bar"><div id="batchProgressFill"></div></div>
        <span id="batchProgressLabel"></span>
      </div>
      <div class="batch-results" id="batchResults"></div>
    </div>
    <div class="modal-foot">
      <button type="button" class="ghost" id="batchCancel">Close</button>
      <button type="button" id="batchRun">Run batch</button>
    </div>
  </div>
</div>

<script>
const $  = s => document.querySelector(s);
const $$ = s => Array.from(document.querySelectorAll(s));
let PAYLOAD = null, FOCUS = null, DOCS = [], SELECTED = new Set();
let BATCH_RESULTS = [];

/* ---------------- meta + docs ---------------- */
function loadMeta(){
  fetch('/api/docs').then(r => r.json()).then(m => {
    $('#modelchip').textContent =
      `${m.model}${m.llm_available ? '' : ' (no key)'} · ${m.documents.length} document(s)`;
    $('#f-model').textContent = m.model;
    $('#f-judge').textContent = m.judge_panel.join(' + ');
    DOCS = m.documents;
    if (SELECTED.size === 0) DOCS.forEach(d => SELECTED.add(d.doc_id));
    renderDocList();
  });
}

function renderDocList(){
  const box = $('#doclist');
  if (!DOCS.length){
    box.innerHTML = '<span class="empty" style="padding:6px">No documents yet. Upload PDFs to begin.</span>';
    return;
  }
  box.innerHTML = DOCS.map(d => {
    const date = d.display_date ? ` (${esc(d.display_date)})` : '';
    return `
    <label class="docpill ${SELECTED.has(d.doc_id) ? 'on' : ''}" data-id="${d.doc_id}"
           title="${esc(d.title)}${date}">
      <input type="checkbox" ${SELECTED.has(d.doc_id) ? 'checked' : ''} data-id="${d.doc_id}">
      <span>${esc(d.title)}${date}</span>
    </label>`;
  }).join('');
  $$('#doclist input').forEach(cb => cb.onchange = () => {
    cb.checked ? SELECTED.add(cb.dataset.id) : SELECTED.delete(cb.dataset.id);
    renderDocList();
  });
}

$('#selAll').onclick  = () => { DOCS.forEach(d => SELECTED.add(d.doc_id)); renderDocList(); };
$('#selNone').onclick = () => { SELECTED.clear(); renderDocList(); };

/* ---------------- upload ---------------- */
$('#file').onchange = async e => {
  const files = Array.from(e.target.files || []);
  if (!files.length) return;
  $('#uplabel').innerHTML = '<span class="spin"></span>Uploading';
  for (const f of files){
    const fd = new FormData();
    fd.append('file', f);
    try {
      const res = await fetch('/api/upload', {method:'POST', body: fd});
      const j = await res.json();
      if (j.error){ note(j.error); }
      else if (j.doc_id){ SELECTED.add(j.doc_id); }
    } catch(err){ note('Upload failed: ' + err); }
  }
  $('#uplabel').textContent = 'Upload PDF(s)';
  $('#file').value = '';
  loadMeta();
};

/* ---------------- ask ---------------- */
$('#go').onclick = ask;
$('#q').addEventListener('keydown', e => { if (e.key === 'Enter') ask(); });

/* ---------------- export dropdown (CSV / Excel / JSON) ---------------- */
$('#exportBtn').onclick = e => { e.stopPropagation(); $('#exportMenu').classList.toggle('open'); };
$$('#exportMenu button').forEach(b => b.onclick = e => {
  e.stopPropagation();
  location.href = '/api/export?format=' + encodeURIComponent(b.dataset.fmt);
  $('#exportMenu').classList.remove('open');
});
// Close the menu when clicking anywhere else.
document.addEventListener('click', () => $('#exportMenu').classList.remove('open'));
// Collapsible documents sidebar.
$('#collapseBtn').onclick = () => {
  const shell = $('#shell');
  shell.classList.toggle('collapsed');
  const collapsed = shell.classList.contains('collapsed');
  $('#collapseBtn').title = collapsed ? 'Expand panel' : 'Collapse panel';
};
// Arrow keys navigate highlights when not typing in the question box.
document.addEventListener('keydown', e => {
  if (e.target && (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA')) return;
  if (e.key === 'ArrowRight'){ gotoHighlight(1); }
  else if (e.key === 'ArrowLeft'){ gotoHighlight(-1); }
});

function note(msg){ $('#note-slot').innerHTML = `<div class="note">${esc(msg)}</div>`; }

async function ask(){
  const question = $('#q').value.trim();
  if (!question) return;
  if (SELECTED.size === 0){ note('Select at least one document to search.'); return; }
  $('#go').disabled = true;
  $('#go').innerHTML = '<span class="spin"></span>Working';
  $('#note-slot').innerHTML = '';
  try{
    const res = await fetch('/api/ask', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({question, doc_ids:[...SELECTED], judge: $('#useJudge').checked})
    });
    PAYLOAD = await res.json();
    if (PAYLOAD.error){ note(PAYLOAD.error); return; }
    render();
  } catch(err){ note('Request failed: ' + err); }
  finally { $('#go').disabled = false; $('#go').textContent = 'Ask'; }
}

/* ---------------- batch ask ---------------- */
function openBatch(){
  const n = SELECTED.size;
  $('#batchDocCount').textContent = n ? `(${n} document${n === 1 ? '' : 's'} selected)` : '(no documents selected yet)';
  $('#batchOverlay').classList.add('open');
}
function closeBatch(){ $('#batchOverlay').classList.remove('open'); }

$('#batchBtn').onclick = openBatch;
$('#batchClose').onclick = closeBatch;
$('#batchCancel').onclick = closeBatch;
$('#batchOverlay').addEventListener('click', e => { if (e.target === $('#batchOverlay')) closeBatch(); });
document.addEventListener('keydown', e => {
  if (e.key === 'Escape' && $('#batchOverlay').classList.contains('open')) closeBatch();
});

// Load questions from a .txt (one per line) or .csv (first column, optional
// "question" header row skipped) file and append them into the textarea.
$('#batchFile').onchange = async e => {
  const f = e.target.files && e.target.files[0];
  if (!f) return;
  $('#batchFileName').textContent = f.name;
  try {
    const text = await f.text();
    let lines;
    if (f.name.toLowerCase().endsWith('.csv')) {
      lines = text.split(/\r?\n/)
        .map(row => row.split(',')[0].trim().replace(/^"|"$/g, ''))
        .filter(Boolean);
      if (lines.length && /^question$/i.test(lines[0])) lines = lines.slice(1);
    } else {
      lines = text.split(/\r?\n/).map(s => s.trim()).filter(Boolean);
    }
    const existing = $('#batchText').value.trim();
    $('#batchText').value = existing ? existing + '\n' + lines.join('\n') : lines.join('\n');
  } catch(err){ note('Could not read file: ' + err); }
  finally { $('#batchFile').value = ''; }
};

$('#batchRun').onclick = runBatch;

const MAX_BATCH_QUESTIONS = 50; // mirrors the server-side ceiling in /api/ask_batch

async function runBatch(){
  const questions = [...new Set(
    $('#batchText').value.split(/\r?\n/).map(s => s.trim()).filter(Boolean)
  )];
  if (!questions.length){ $('#batchResults').innerHTML = '<div class="note">Enter at least one question, one per line.</div>'; return; }
  if (SELECTED.size === 0){ $('#batchResults').innerHTML = '<div class="note">Select at least one document in the sidebar before running a batch.</div>'; return; }
  if (questions.length > MAX_BATCH_QUESTIONS){
    $('#batchResults').innerHTML = `<div class="note">Max ${MAX_BATCH_QUESTIONS} questions per batch (you have ${questions.length}). Trim the list and try again.</div>`;
    return;
  }

  $('#batchRun').disabled = true;
  $('#batchRun').textContent = 'Running…';
  $('#batchResults').innerHTML = '';
  $('#batchProgress').style.display = 'flex';
  BATCH_RESULTS = [];

  const total = questions.length;
  const docIds = [...SELECTED];
  const judge  = $('#useJudge').checked;

  // Run one question at a time against the normal /api/ask endpoint so the
  // bar reflects real, per-question progress instead of a single opaque
  // request that only resolves once every question is done. Results render
  // into the table as each one lands, not all at once at the end.
  for (let i = 0; i < total; i++){
    const question = questions[i];
    $('#batchProgressLabel').textContent = `${i} / ${total} — asking "${question.length > 40 ? question.slice(0, 40) + '…' : question}"`;
    $('#batchProgressFill').style.width = Math.max(4, Math.round((i / total) * 100)) + '%';

    try {
      const res = await fetch('/api/ask', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({question, doc_ids: docIds, judge})
      });
      const payload = await res.json();
      if (payload.error){
        BATCH_RESULTS.push({index: i, question, ok: false, error: payload.error});
      } else {
        const g = payload.grading || {};
        BATCH_RESULTS.push({
          index: i, question, ok: true,
          entry_id: payload.entry_id,
          answer: (payload.extraction || {}).answer,
          judge_mean: g.judge_mean,
          semantic_match: g.semantic_match,
          payload,
        });
      }
    } catch(err){
      BATCH_RESULTS.push({index: i, question, ok: false, error: String(err)});
    }

    $('#batchProgressFill').style.width = Math.round(((i + 1) / total) * 100) + '%';
    renderBatchResults();
  }

  const okCount = BATCH_RESULTS.filter(r => r.ok).length;
  const errCount = total - okCount;
  $('#batchProgressLabel').textContent = `${okCount} / ${total} answered` + (errCount ? `, ${errCount} failed` : '');
  $('#batchRun').disabled = false;
  $('#batchRun').textContent = 'Run batch';
  loadMeta(); // keep doc/model chip state fresh
}

function renderBatchResults(){
  const box = $('#batchResults');
  if (!BATCH_RESULTS.length){ box.innerHTML = ''; return; }
  box.innerHTML = `
    <table class="batch-table">
      <thead><tr><th>#</th><th>Question</th><th>Answer</th><th>Judge</th><th>Semantic</th><th></th></tr></thead>
      <tbody>
        ${BATCH_RESULTS.map(r => {
          if (!r.ok){
            return `<tr class="batch-err"><td>${r.index + 1}</td><td>${esc(r.question)}</td>
                    <td colspan="4">Failed: ${esc(r.error || 'unknown error')}</td></tr>`;
          }
          const a = r.answer || '';
          const snippet = a.length > 140 ? a.slice(0, 140) + '…' : a;
          return `<tr>
            <td>${r.index + 1}</td>
            <td>${esc(r.question)}</td>
            <td>${esc(snippet)}</td>
            <td>${pct(r.judge_mean)}</td>
            <td>${pct(r.semantic_match)}</td>
            <td><button type="button" class="ghost small batch-view" data-idx="${r.index}">View</button></td>
          </tr>`;
        }).join('')}
      </tbody>
    </table>`;
  $$('.batch-view').forEach(btn => btn.onclick = () => {
    const r = BATCH_RESULTS[Number(btn.dataset.idx)];
    if (!r || !r.ok) return;
    PAYLOAD = r.payload;
    $('#q').value = r.question;
    closeBatch();
    render();
  });
}

/* ---------------- render ---------------- */
function fmt(x){
  if (x == null) return '-';
  const n = Number(x);
  if (!isFinite(n)) return '-';
  if (n === 0) return '0';
  if (n === 1) return '1';
  if (Number.isInteger(n)) return String(n);
  return n.toFixed(2);
}
// Percentage formatter for [0,1] scores: 0.45 -> "45%", 1 -> "100%", null -> "-".
function pct(x){
  if (x == null) return '-';
  const n = Number(x);
  if (!isFinite(n)) return '-';
  return Math.round(n * 100) + '%';
}
// Set a KPI stat to its percentage and colour it: >50% green, 0 black, else red.
function setPctStat(sel, v){
  const el = $(sel);
  if (!el) return;
  el.textContent = pct(v);
  const n = Number(v);
  if (v == null || !isFinite(n)){ el.style.color = ''; return; }
  const p = Math.round(n * 100);
  if (p === 0)      el.style.color = 'var(--ink)';   // black
  else if (p > 50)  el.style.color = 'var(--ok)';    // green
  else              el.style.color = 'var(--bad)';   // red
}

function render(){
  const g = PAYLOAD.grading || {};
  const x = PAYLOAD.extraction || {};
  $('#s-items').textContent     = (PAYLOAD.context_items || []).length;
  // Scores in [0,1] are shown as percentages; context passages stays a count.
  // Colour rule for the percentage KPIs: >50% green, exactly 0 black, else red.
  setPctStat('#s-judge',     g.judge_mean ?? g.judge_confidence);
  setPctStat('#s-sem',       g.semantic_match);
  setPctStat('#s-relevancy', g.deepeval_answer_relevancy_score);
  setPctStat('#s-grounding', g.claim_grounding_ratio);

  const r = PAYLOAD.review || {};
  $('#f-diag').textContent =
    `passages=${(PAYLOAD.context_items||[]).length} · priority=${fmt(r.priority)}` +
    (r.error_tags ? ` · flags: ${r.error_tags}` : '') +
    (g.judge_variance != null && g.judge_variance >= 0.2 ? ' · high judge disagreement' : '') +
    (g.deepeval_answer_relevancy_score != null ? ` · relevancy=${pct(g.deepeval_answer_relevancy_score)}` : '') +
    (g.claim_grounding_ratio != null ? ` · grounding=${pct(g.claim_grounding_ratio)}` : '');

  FOCUS = null;
  renderSource();
  renderReport();
  renderPassages();
  renderGrading();
  renderErrors();
  resetReviewCard();
}

/* Render every contributing document's full text, injecting a divider before
   each and wrapping the retrieved spans in <mark> at their absolute offsets. */
function renderSource(){
  const docs = (PAYLOAD && PAYLOAD.documents) || [];
  if (!docs.length){ $('#source').innerHTML = '<div class="empty">No document loaded.</div>'; return; }
  const spans = (PAYLOAD && PAYLOAD.spans) || [];

  let html = '';
  for (const d of docs){
    html += `<span class="docdiv">${esc(d.title || d.label)}</span>`;
    const mine = spans.filter(s => s.doc === d.label && s.start >= 0)
                      .sort((a,b) => a.start - b.start);
    const text = d.text || '';
    let cursor = 0, last = -1;
    for (const s of mine){
      if (s.start < last) continue;   // skip overlaps
      // `item` links a highlight span back to its evidence quote / passage.
      const item = s.item || s.id;
      const gl = s.grounding ? ' ' + s.grounding : '';
      html += esc(text.slice(cursor, s.start));
      html += `<mark id="m-${s.id}" class="${gl.trim()}" data-id="${s.id}" data-item="${item}" `
            + `title="match ${s.match_score != null ? Math.round(s.match_score) : ''}%">`
            + `${esc(text.slice(s.start, s.end))}</mark>`;
      cursor = s.end; last = s.end;
    }
    html += esc(text.slice(cursor));
  }
  $('#source').innerHTML = html;
  $$('#source mark').forEach(m => m.onclick = () => {
    // clicking a highlight also selects it in the navigator
    const idx = HL_ORDER.indexOf(m.id);
    if (idx !== -1) HL_INDEX = idx;
    focusItem(m.dataset.item, 'source');
    updateHlNav();
  });

  // Build the ordered list of highlight mark ids for prev/next navigation.
  HL_ORDER = $$('#source mark').map(m => m.id);
  HL_INDEX = HL_ORDER.length ? 0 : -1;
  setupHlNav();
}

/* ---------------- highlight navigation ---------------- */
let HL_ORDER = [], HL_INDEX = -1;

function setupHlNav(){
  // The navigator stays visible for consistent panel height and discoverability;
  // its buttons are simply disabled when there are no highlights.
  updateHlNav();
}

function updateHlNav(){
  const count = $('#hlcount');
  const hasHl = HL_ORDER.length > 0;
  if (!hasHl){
    count.textContent = '0 / 0';
    $('#hlprev').disabled = true;
    $('#hlnext').disabled = true;
    return;
  }
  if (HL_INDEX < 0) HL_INDEX = 0;
  count.textContent = (HL_INDEX + 1) + ' / ' + HL_ORDER.length;
  $('#hlprev').disabled = (HL_INDEX <= 0);
  $('#hlnext').disabled = (HL_INDEX >= HL_ORDER.length - 1);
  // focus-highlight the current mark
  $$('#source mark').forEach(m => m.classList.toggle('focus', m.id === HL_ORDER[HL_INDEX]));
}

function gotoHighlight(delta){
  if (!HL_ORDER.length) return;
  HL_INDEX = Math.max(0, Math.min(HL_ORDER.length - 1, HL_INDEX + delta));
  const mark = $('#' + HL_ORDER[HL_INDEX]);
  if (mark){
    mark.scrollIntoView({behavior:'smooth', block:'center'});
    if (mark.dataset.item) focusItem(mark.dataset.item, 'source');
  }
  updateHlNav();
}

$('#hlprev').onclick = () => gotoHighlight(-1);
$('#hlnext').onclick = () => gotoHighlight(1);

function renderReport(){
  const x = PAYLOAD.extraction || {};
  const reqs = (x.key_requirements || []).map(p => `<li>${esc(p)}</li>`).join('');
  const spans = (PAYLOAD && PAYLOAD.spans) || [];
  // Build clickable evidence chips: each verbatim quote links to its highlight.
  const ev = (x.source_evidence || []).map((q, i) => {
    const sp = spans.find(s => (s.quote || '').trim() === (q || '').trim())
             || spans[i];
    const gid = sp ? sp.item : '';
    const gl  = sp ? (sp.grounding || '') : '';
    return `<li class="evq ${esc(gl)}" data-item="${esc(gid)}">${esc(q)}</li>`;
  }).join('');
  // Compact answer-confidence chip (smaller, sits to the right of the action bar).
  const conf = x.model_confidence;
  let confChip = '';
  if (conf != null){
    const p = Math.round(Number(conf) * 100);
    const cls = p >= 80 ? 'cb-hi' : (p >= 55 ? 'cb-mid' : 'cb-lo');
    confChip = `<div class="conf-chip ${cls}">
        <span class="conf-chip-label">Answer confidence</span>
        <span class="conf-chip-val">${p}%</span>
      </div>`;
  }
  // Compact review buttons that mirror the full HITL panel. Clicking one records
  // the decision and scrolls down to the validation card. Hidden when HITL is off.
  const actionBar = `
    <div class="report-actionbar">
      <div class="mini-review" id="miniReview">
        <button class="rv-mini m-approve" data-status="approved" title="Approve this answer">&#10003; Approve</button>
        <button class="rv-mini m-accept"  data-status="accepted" title="Edit the answer">&#9998; Edit</button>
        <button class="rv-mini m-reject"  data-status="rejected" title="Reject this answer">&#10007; Reject</button>
        <button class="rv-mini m-skip"     data-status="skipped"  title="Skip / decide later">&#8631; Skip</button>
      </div>
      ${confChip}
    </div>`;
  $('#tab-report').innerHTML = `
    ${actionBar}
    <div class="xhead" style="margin-top:0">Answer</div>
    <div class="answer">${esc(x.answer || '')}</div>
    ${reqs ? `<div class="xhead">Key requirements</div><ul class="points">${reqs}</ul>` : ''}
    ${ev ? `<div class="xhead">Source evidence (verbatim)</div><ul class="points evlist">${ev}</ul>` : ''}
    <div class="xhead">Source passages used</div>
    <div class="answer" style="color:#475569">${esc(x.source_passages_used || '-')}</div>`;
  // Wire the compact review buttons and reflect any current decision + toggle state.
  $$('#tab-report .rv-mini').forEach(b => b.onclick = () => reviewFromReport(b.dataset.status));
  markSelected(CURRENT_DECISION);
  applyHitlVisibility();
  // Clicking a quote focuses its highlight span in the document pane.
  $$('#tab-report .evq').forEach(el => el.onclick = () => {
    if (el.dataset.item) focusItem(el.dataset.item, 'items');
  });
}

function renderPassages(){
  const items = PAYLOAD.context_items || [];
  if (!items.length){
    $('#tab-passages').innerHTML = '<div class="empty">No passages retrieved.</div>';
    return;
  }
  $('#tab-passages').innerHTML = items.map((it, i) => `
    <div class="item" id="it-${it.id}" data-id="${it.id}">
      <div class="bullet">${(it.score ?? 0).toFixed(2)}</div>
      <div>
        <div class="body">${esc(it.text)}</div>
        <div class="meta">${esc(it.source_document)} · p. ${it.page}</div>
      </div>
    </div>`).join('');
  $$('#tab-passages .item').forEach(el => el.onclick = () => focusItem(el.dataset.id, 'items'));
}

function bar(v){ return `<div class="barwrap"><div class="bar" style="width:${Math.round((v||0)*100)}%"></div></div>`; }

function renderGrading(){
  const g = PAYLOAD.grading || {};
  const scores  = g.judge_scores  || {};
  const reasons = g.judge_reasons || {};
  const perModel = Object.keys(scores).length
    ? `<div class="xhead">Per-judge scores</div><div class="judgechips">` +
      Object.entries(scores).map(([k,v]) =>
        `<div class="judgechip"><span class="jk">${esc(k)}</span><span class="jv">${fmt(v)}</span></div>`
      ).join('') + `</div>` +
      `<div class="judgereasons">` +
      Object.entries(scores).map(([k]) =>
        reasons[k] ? `<div class="r" style="margin-top:4px"><b>${esc(k)}:</b> ${esc(reasons[k])}</div>` : ''
      ).join('') + `</div>`
    : '';

  // Grader 5: unsupported / contradicted / meta claim lists
  const unsupported  = g.unsupported_claims  || [];
  const contradicted = g.contradicted_claims || [];
  const metaClaims   = g.meta_claims || [];
  const metaBlock = metaClaims.length
    ? `<div class="xhead" style="color:#64748b">Meta-statements (${metaClaims.length}) — about the context, not checkable, excluded from grounding</div><ul class="points">${metaClaims.map(c=>`<li>${esc(c)}</li>`).join('')}</ul>`
    : '';
  const claimLists = (unsupported.length || contradicted.length) ? `
    ${unsupported.length  ? `<div class="xhead" style="color:var(--warn)">Unsupported claims (${unsupported.length})</div><ul class="points">${unsupported.map(c=>`<li>${esc(c)}</li>`).join('')}</ul>` : ''}
    ${contradicted.length ? `<div class="xhead" style="color:var(--bad)">Contradicted claims (${contradicted.length})</div><ul class="points">${contradicted.map(c=>`<li>${esc(c)}</li>`).join('')}</ul>` : ''}
    ${metaBlock}
  ` : `<div class="r" style="color:var(--ok);margin-top:6px">✓ All verifiable claims grounded in the retrieved context.</div>${metaBlock}`;

  // Grader 7: uncovered requirements
  const uncovReqs = g.uncovered_requirements || [];
  const coverDetail = uncovReqs.length
    ? `<div class="r" style="color:var(--warn);margin-top:4px">Missing from prose (${uncovReqs.length}):</div><ul class="points">${uncovReqs.map(r=>`<li>${esc(r)}</li>`).join('')}</ul>`
    : `<div class="r" style="color:var(--ok);margin-top:4px">✓ All requirements present in prose.</div>`;

  // Grader 11: quotes that failed to align to the retrieved context
  const unaligned = g.unaligned_evidence || [];
  const unalignedDetail = unaligned.length
    ? `<div class="r" style="color:var(--warn);margin-top:4px">Quotes not found in context (${unaligned.length}):</div><ul class="points">${unaligned.map(q=>`<li>${esc(q)}</li>`).join('')}</ul>`
    : (g.evidence_alignment_rate != null ? `<div class="r" style="color:var(--ok);margin-top:4px">✓ All evidence quotes align to the retrieved context.</div>` : '');

  // Grader 13: numbers in the answer absent from the context
  const ungroundedNums = g.ungrounded_numbers || [];
  const ungroundedNumDetail = ungroundedNums.length
    ? `<div class="r" style="color:var(--warn);margin-top:4px">Numbers not in context (${ungroundedNums.length}): ${ungroundedNums.map(esc).join(', ')}</div>`
    : (g.numeric_consistency != null ? `<div class="r" style="color:var(--ok);margin-top:4px">✓ All numbers in the answer appear in the retrieved context.</div>` : '');

  // Small "?" help icon with a hover tooltip (rendered via a document-level
  // tooltip so the scrollable panel never clips it). Mean and the
  // per-judge chips deliberately get no icon.
  const hk = (label, tip) => `${esc(label)}<span class="help" data-tip="${esc(tip)}" aria-label="More info">i</span>`;

  $('#tab-grading').innerHTML = `
    <div class="xhead" style="margin-top:0">Grader 1: Multi-judge consensus</div>
    <div class="grades">
      <div class="grade" style="grid-column:1 / -1"><div class="k">Mean</div><div class="v">${pct(g.judge_mean)}</div>${bar(g.judge_mean)}</div>
      <div class="grade"><div class="k">${hk('Bounds', 'Lowest to highest judge confidence for this answer.')}</div><div class="v" style="font-size:15px">${pct(g.judge_lower)} to ${pct(g.judge_upper)}</div><div class="r">lower / upper</div></div>
      <div class="grade"><div class="k">${hk('Disagreement (std)', 'Spread of scores across judges; higher means they disagree more.')}</div><div class="v">${fmt(g.judge_variance)}</div><div class="r">${(g.judge_variance != null && g.judge_variance >= 0.2) ? 'flag for SME review' : 'within tolerance'}</div></div>
    </div>
    <div class="r" style="margin-top:6px;color:#475569">${esc(g.judge_reasoning || '')}</div>
    ${perModel}

    <div class="xhead">Grader 2: Semantic match vs source</div>
    <div class="grades">
      <div class="grade" style="grid-column:1 / -1">
        <div class="k">${hk('Embedding cosine (answer vs retrieved context)', 'Cosine similarity between the answer and the retrieved source text.')}</div>
        <div class="v">${pct(g.semantic_match)}</div>${bar(g.semantic_match)}
        <div class="r">Code-based, deterministic. Ground truth = the source document text.</div>
      </div>
    </div>

    <div class="xhead">Grader 3: DeepEval Answer Relevancy</div>
    <div class="grades">
      <div class="grade">
        <div class="k">${hk('Relevancy score', 'How well the answer addresses the question (DeepEval).')}</div>
        <div class="v">${pct(g.deepeval_answer_relevancy_score)}</div>${bar(g.deepeval_answer_relevancy_score)}
        <div class="r">${g.deepeval_answer_relevancy_score != null ? (g.deepeval_answer_relevancy_passed ? 'PASS' : 'FAIL') : 'unavailable'}</div>
      </div>
      <div class="grade">
        <div class="k">${hk('Reason', "DeepEval's explanation for the relevancy score.")}</div>
        <div class="v" style="font-size:13px;font-weight:500;line-height:1.4">${esc(g.deepeval_answer_relevancy_reason || '-')}</div>
      </div>
    </div>

    <div class="xhead">Grader 4: Claim decomposition</div>
    <div class="grades">
      <div class="grade"><div class="k">${hk('Grounding ratio', "Share of the answer's verifiable claims supported by the retrieved passages. Meta-statements about the context are excluded.")}</div><div class="v">${pct(g.claim_grounding_ratio)}</div>${bar(g.claim_grounding_ratio)}<div class="r">${g.claims_verifiable != null ? 'of ' + g.claims_verifiable + ' verifiable claim' + (g.claims_verifiable === 1 ? '' : 's') : ''}</div></div>
      <div class="grade"><div class="k">${hk('Claims', 'Answer claims split into supported, unsupported, contradicted, and meta (statements about the context, not checkable).')}</div><div class="v" style="font-size:16px">${g.claims_supported ?? '-'} supported &nbsp;${g.claims_unsupported ?? '-'} unsupported &nbsp;${g.claims_contradicted ?? '-'} contradicted &nbsp;${g.claims_meta ?? '-'} meta</div><div class="r">supported / unsupported / contradicted / meta</div></div>
    </div>
    ${claimLists}

    <div class="xhead">Grader 5: Response length</div>
    <div class="grades">
      <div class="grade" style="grid-column:1 / -1">
        <div class="k">${hk('Answer length (approx tokens)', 'Approximate length of the answer, in tokens.')}</div>
        <div class="v">${g.answer_len_tokens ?? '–'}</div>
        <div class="r">${g.answer_len_chars != null ? g.answer_len_chars + ' chars' : ''}</div>
      </div>
    </div>

    <div class="xhead">Grader 6: Key requirements coverage</div>
    <div class="grades">
      <div class="grade" style="grid-column:1 / -1">
        <div class="k">${hk('Requirements covered in prose', 'Key requirements that also appear in the written answer.')}</div>
        <div class="v" style="font-size:18px">${g.requirements_covered ?? '–'} / ${(g.requirements_covered ?? 0) + (g.requirements_uncovered ?? 0)}</div>
      </div>
    </div>
    ${coverDetail}

    <div class="xhead">Grader 7: Retrieval quality distribution</div>
    <div class="grades">
      <div class="grade"><div class="k">${hk('Max score', 'Highest retrieval similarity among the passages used.')}</div><div class="v">${pct(g.retrieval_score_max)}</div>${bar(g.retrieval_score_max)}</div>
      <div class="grade"><div class="k">${hk('Mean score', 'Average retrieval similarity across the passages used.')}</div><div class="v">${pct(g.retrieval_score_mean)}</div>${bar(g.retrieval_score_mean)}</div>
      <div class="grade"><div class="k">${hk('Spread (max-min)', 'Gap between the strongest and weakest retrieved passage.')}</div><div class="v">${fmt(g.retrieval_score_spread)}</div><div class="r">${(g.retrieval_score_spread != null && g.retrieval_score_spread > 0.3) ? 'high spread' : 'within range'}</div></div>
      <div class="grade"><div class="k">${hk('Weak chunk ratio', 'Share of retrieved passages below the strong-match threshold.')}</div><div class="v">${pct(g.weak_chunks_ratio)}</div><div class="r">${(g.weak_chunks_ratio != null && g.weak_chunks_ratio > 0.5) ? 'many weak chunks' : 'acceptable'}</div></div>
    </div>

    <div class="xhead">Grader 8: Sub-query agreement</div>
    <div class="grades">
      <div class="grade" style="grid-column:1 / -1">
        <div class="k">${hk('Agreement across sub-queries', 'Share of the final passages that more than one sub-query retrieved. Higher means retrieval was robust to how the question was phrased.')}</div>
        <div class="v">${g.subquery_agreement_ratio != null ? pct(g.subquery_agreement_ratio) : '–'}</div>
        ${g.subquery_agreement_ratio != null ? bar(g.subquery_agreement_ratio) : ''}
        <div class="r">${g.subquery_agreement_ratio != null ? ((g.subquery_count ?? '?') + ' sub-queries · ' + (g.subquery_mean_hits ?? '?') + ' mean hits/chunk') : 'not computed'}</div>
      </div>
    </div>

    <div class="xhead">Grader 9: Evidence alignment</div>
    <div class="grades">
      <div class="grade" style="grid-column:1 / -1">
        <div class="k">${hk('Quotes aligned to context', "Share of the model's verbatim evidence quotes that actually match the retrieved passages. A low rate means quotes were paraphrased or fabricated.")}</div>
        <div class="v">${g.evidence_alignment_rate != null ? pct(g.evidence_alignment_rate) : '–'}</div>
        ${g.evidence_alignment_rate != null ? bar(g.evidence_alignment_rate) : ''}
        <div class="r">${g.evidence_quotes_aligned ?? '–'} / ${g.evidence_quotes_total ?? '–'} quotes aligned</div>
      </div>
    </div>
    ${unalignedDetail}

    <div class="xhead">Grader 10: Context utilisation</div>
    <div class="grades">
      <div class="grade" style="grid-column:1 / -1">
        <div class="k">${hk('Retrieved passages used', 'Share of retrieved passages the answer actually drew on. Low utilisation means retrieval returned unused material.')}</div>
        <div class="v">${g.context_utilisation != null ? pct(g.context_utilisation) : '–'}</div>
        ${g.context_utilisation != null ? bar(g.context_utilisation) : ''}
        <div class="r">${g.chunks_used ?? '–'} / ${g.chunks_total ?? '–'} passages used</div>
      </div>
    </div>

    <div class="xhead">Grader 11: Numeric consistency</div>
    <div class="grades">
      <div class="grade" style="grid-column:1 / -1">
        <div class="k">${hk('Numbers grounded in context', 'Share of numbers stated in the answer that also appear in the retrieved context. Catches fabricated thresholds, doses and dates.')}</div>
        <div class="v">${g.numeric_consistency != null ? pct(g.numeric_consistency) : '–'}</div>
        ${g.numeric_consistency != null ? bar(g.numeric_consistency) : ''}
        <div class="r">${g.numeric_consistency != null ? ((g.numbers_grounded ?? '–') + ' / ' + (g.numbers_total ?? '–') + ' numbers found in context') : 'answer contains no numbers'}</div>
      </div>
    </div>
    ${ungroundedNumDetail}`;
}

function renderErrors(){
  const r = PAYLOAD.review || {};
  const g = PAYLOAD.grading || {};
  const tagStr  = (r.error_tags || '').trim();
  const tags    = tagStr ? tagStr.split(',').map(s => s.trim()).filter(Boolean) : [];
  const reasons = (r.error_reasons || '').split('|').map(s => s.trim());

  // Human-readable metadata for each error tag produced by the engine.
  const META = {
    retrieval_failure: {label:'Retrieval failure',  sev:'bad',  desc:'The answer could not be supported from what was retrieved: claim grounding collapsed or nothing relevant was found.'},
    partial_retrieval: {label:'Partial retrieval',  sev:'warn', desc:'Relevant passages were found but the supporting evidence was incomplete: partial grounding, weak passages, or an unsupported claim.'},
    missing_context:   {label:'Missing context',   sev:'bad',  desc:'The model stated the retrieved context was insufficient to answer the question.'},
    misinterpretation: {label:'Misinterpretation',  sev:'warn', desc:'Retrieval and grounding were adequate, but the judge scored the answer poorly, suggesting the source was read incorrectly rather than missing.'},
    hallucination:     {label:'Hallucination risk', sev:'bad',  desc:'The answer contains unsupported claims with no retrieval excuse and a low judge score: fabricated content on otherwise adequate retrieval.'},
    contradiction:     {label:'Contradiction',      sev:'bad',  desc:'One or more claims are directly contradicted by the retrieved context: the answer states the opposite of the source.'},
    coverage_gap:      {label:'Coverage gap',       sev:'warn', desc:'A key requirement the model listed was not carried into the answer prose, so a reader of the prose alone would miss it.'},
  };

  // Update the tab badge with the number of flags (hidden when there are none).
  const badge = $('#errBadge');
  if (badge){
    if (tags.length){ badge.textContent = tags.length; badge.classList.add('on'); }
    else { badge.textContent = ''; badge.classList.remove('on'); }
  }

  let cards;
  if (tags.length){
    cards = tags.map((t, i) => {
      const m = META[t] || {label:t, sev:'warn', desc:''};
      const reason = reasons[i] || '';
      return `<div class="errcard err-${m.sev}">
        <div class="errcard-head"><span class="errdot"></span>${esc(m.label)}<span class="errtag">${esc(t)}</span></div>
        <div class="errcard-desc">${esc(m.desc)}</div>
        ${reason ? `<div class="errcard-reason">Signal: ${esc(reason)}</div>` : ''}
      </div>`;
    }).join('');
  } else {
    cards = `<div class="err-ok">&#10003; No error flags on this answer. Retrieval, context and grounding checks all passed.</div>`;
  }

  const highDisagree = (g.judge_variance != null && g.judge_variance >= 0.2);
  const signals = `
    <div class="xhead">Review signals</div>
    <div class="grades">
      <div class="grade"><div class="k">Review priority</div><div class="v">${fmt(r.priority)}</div><div class="r">lower = needs more scrutiny</div></div>
      <div class="grade"><div class="k">Context retrieved</div><div class="v">${r.no_context ? 'No' : 'Yes'}</div><div class="r">${r.no_context ? 'nothing retrieved' : 'passages found'}</div></div>
      <div class="grade"><div class="k">Judge disagreement</div><div class="v">${fmt(g.judge_variance)}</div><div class="r">${highDisagree ? 'flag for SME review' : 'within tolerance'}</div></div>
    </div>`;

  $('#tab-errors').innerHTML = `
    <div class="xhead" style="margin-top:0">Error flags</div>
    <div class="errlist">${cards}</div>
    ${signals}`;
}

function focusItem(id, from){
  FOCUS = id;
  // Highlights are keyed by evidence-quote id (e.g. "ev0"). Focus all marks that
  // belong to this item and scroll the first into view.
  const marks = $$('#source mark').filter(m => m.dataset.item === id);
  $$('#source mark').forEach(m => m.classList.toggle('focus', m.dataset.item === id));
  $$('.item').forEach(i => i.classList.toggle('on', i.dataset.id === id));
  let target;
  if (from === 'items'){
    target = marks.length ? marks[0] : $('#m-' + id);
  } else {
    target = $('#it-' + id);
  }
  if (target) target.scrollIntoView({behavior:'smooth', block:'center'});
}

/* ---------------- tabs ---------------- */
$$('.tabs button').forEach(b => b.onclick = () => {
  $$('.tabs button').forEach(x => x.classList.toggle('on', x === b));
  ['report','passages','grading','errors'].forEach(t =>
    $('#tab-' + t).style.display = (t === b.dataset.tab ? '' : 'none'));
});

/* ---------------- human-in-the-loop validation ---------------- */
// The current reviewer decision for the answer on screen (null until decided).
let CURRENT_DECISION = null;

// Show the validation card + the compact report buttons only when HITL is
// enabled AND there is an answer to review.
function applyHitlVisibility(){
  const on = $('#useHitl').checked;
  const hasAnswer = PAYLOAD && !PAYLOAD.error && PAYLOAD.entry_id != null;
  const show = !!(on && hasAnswer);
  const card = $('#reviewCard');
  if (card) card.style.display = show ? '' : 'none';
  const mini = $('#miniReview');
  if (mini) mini.style.display = show ? '' : 'none';
}
$('#useHitl').onchange = applyHitlVisibility;

// Highlight the chosen action on both the full and compact button sets.
function markSelected(status){
  $$('.rv, .rv-mini').forEach(b => b.classList.toggle('selected', b.dataset.status === status));
}

function resetReviewCard(){
  CURRENT_DECISION = null;
  const badge = $('#reviewStatus');
  if (badge){ badge.textContent = 'Awaiting review'; badge.className = 'review-status'; }
  const cm = $('#reviewComment'); if (cm) cm.value = '';
  // Prefill the corrected-answer box with the model's answer so an SME can edit in place.
  const ca = $('#correctedAnswer');
  if (ca) ca.value = (PAYLOAD && PAYLOAD.extraction && PAYLOAD.extraction.answer) || '';
  $$('.rv, .rv-mini').forEach(b => { b.disabled = false; b.classList.remove('selected'); });
  applyHitlVisibility();
  // (reviewer id is intentionally kept across turns)
}

async function submitReview(status){
  if (!PAYLOAD || PAYLOAD.entry_id == null){ note('Ask a question before validating an answer.'); return; }
  const sendCorrected = (status === 'accepted' || status === 'rejected');
  const body = {
    entry_id        : PAYLOAD.entry_id,
    status          : status,
    reviewer        : ($('#reviewer').value || '').trim(),
    comment         : ($('#reviewComment').value || '').trim(),
    corrected_answer: sendCorrected ? ($('#correctedAnswer').value || '').trim() : ''
  };
  $$('.rv, .rv-mini').forEach(b => b.disabled = true);
  try{
    const res = await fetch('/api/validate', {
      method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)
    });
    const j = await res.json();
    if (j.error){ note(j.error); return; }
    CURRENT_DECISION = status;
    let label;
    if (status === 'accepted'){
      const orig  = ((PAYLOAD.extraction && PAYLOAD.extraction.answer) || '').trim();
      const edited = (($('#correctedAnswer').value || '').trim() !== orig);
      label = edited ? 'Edited' : 'Reviewed';
    } else {
      label = {approved:'Approved', rejected:'Rejected', skipped:'Skipped'}[status] || status;
    }
    const badge = $('#reviewStatus');
    badge.textContent = label + (j.reviewer ? ' \u00b7 ' + j.reviewer : '');
    badge.className = 'review-status s-' + status;
    markSelected(status);
  } catch(err){
    note('Validation failed: ' + err);
  } finally {
    // Re-enable so the reviewer can refine (e.g. edit then re-confirm) or change their mind.
    $$('.rv, .rv-mini').forEach(b => b.disabled = false);
  }
}
$$('.rv').forEach(b => b.onclick = () => submitReview(b.dataset.status));

// Triggered by the compact buttons in the Report tab: record the decision, then
// scroll down to the full HITL panel so the reviewer can confirm or refine it.
async function reviewFromReport(status){
  await submitReview(status);
  const card = $('#reviewCard');
  if (!card || card.style.display === 'none') return;
  card.scrollIntoView({behavior:'smooth', block:'center'});
  // Land the cursor in the field most relevant to the chosen action.
  setTimeout(() => {
    if (status === 'accepted'){ const t = $('#correctedAnswer'); if (t) t.focus({preventScroll:true}); }
    else if (status === 'rejected'){ const c = $('#reviewComment'); if (c) c.focus({preventScroll:true}); }
  }, 350);
}

function esc(s){
  return (s ?? '').toString().replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
}

/* ---------------- help "?" tooltips ---------------- */
// One reusable tooltip element, positioned with position:fixed so the
// scrollable panels can never clip it. Uses delegation so dynamically
// rendered help icons work without re-binding.
let TIP_EL = null;
function showTip(el){
  hideTip();
  const tip = document.createElement('div');
  tip.className = 'help-tip';
  tip.textContent = el.dataset.tip || '';
  document.body.appendChild(tip);
  const r  = el.getBoundingClientRect();
  const tr = tip.getBoundingClientRect();
  let left = r.left + r.width / 2 - tr.width / 2;
  left = Math.max(8, Math.min(left, window.innerWidth - tr.width - 8));
  let top = r.top - tr.height - 8;
  if (top < 8) top = r.bottom + 8;   // flip below the icon if there's no room above
  tip.style.left = left + 'px';
  tip.style.top  = top + 'px';
  requestAnimationFrame(() => tip.classList.add('show'));
  TIP_EL = tip;
}
function hideTip(){ if (TIP_EL){ TIP_EL.remove(); TIP_EL = null; } }
document.addEventListener('mouseover', e => {
  const h = e.target.closest && e.target.closest('.help');
  if (h) showTip(h);
});
document.addEventListener('mouseout', e => {
  const h = e.target.closest && e.target.closest('.help');
  if (h) hideTip();
});
// Also clear the tooltip on scroll so it never lingers in the wrong place.
document.addEventListener('scroll', hideTip, true);

loadMeta();
</script>
</body>
</html>
"""


# =============================================================================
# Routes
# =============================================================================

@app.route("/")
def index():
    return Response(STUDIO_HTML, mimetype="text/html")


@app.route("/api/docs")
def docs():
    documents = []
    for e in M.DOC_REGISTRY:
        # page count = distinct page numbers among the doc's chunks
        idxs = M.DOC_CHUNK_INDICES.get(e["source_document"], [])
        pages = {M.all_chunks[i]["page_number"] for i in idxs}
        documents.append({
            "doc_id"      : e["doc_id"],
            "title"       : e.get("doc_title") or e["source_document"],
            "display_date": e.get("display_date"),
            "label"       : e["source_document"],   # internal retrieval key
            "version"     : e["version_label"],
            "pages"       : (max(pages) + 1) if pages else 0,
            "chunks"      : e["num_chunks"],
        })
    return jsonify({
        "documents"    : documents,
        "model"        : M.model_name,
        "judge_panel"  : [f"{p}:{m}" for p, m in M.JUDGE_PANEL],
        "llm_available": True,
    })


@app.route("/api/doctext")
def doctext():
    doc_id = request.args.get("id", "")
    label = doc_label_for_id(doc_id)
    if not label:
        return jsonify({"error": "unknown document"}), 404
    return jsonify({"label": label, "text": full_document_text(label)})


@app.route("/api/upload", methods=["POST"])
def upload():
    try:
        if "file" not in request.files:
            return jsonify({"error": "no file provided"}), 400
        f = request.files["file"]
        if not f.filename:
            return jsonify({"error": "empty filename"}), 400
        if not f.filename.lower().endswith(".pdf"):
            return jsonify({"error": "only PDF files are supported"}), 400

        safe_name = os.path.basename(f.filename)
        dest = os.path.join(UPLOAD_DIR, safe_name)
        try:
            f.save(dest)
        except Exception as e:
            traceback.print_exc()
            return jsonify({"error": f"could not save upload: {e}"}), 500

        # Guard against empty / non-PDF payloads that would crash the parser.
        if not os.path.exists(dest) or os.path.getsize(dest) == 0:
            return jsonify({"error": "uploaded file is empty"}), 400

        entry = register_pdf(dest)
        return jsonify({
            "doc_id"      : entry["doc_id"],
            "title"       : entry.get("doc_title") or entry["source_document"],
            "display_date": entry.get("display_date"),
        })
    except Exception as e:
        # Never let an exception escape as a dropped connection ("Failed to
        # fetch"); always return a JSON error the UI can display.
        traceback.print_exc()
        return jsonify({"error": f"failed to process PDF: {e}"}), 500


def record_answer(question: str, doc_ids: list[str], run_judge: bool = True) -> dict:
    """
    Run answer_question() and append the result to SESSION_LOG, exactly as the
    single-question /api/ask endpoint used to do inline. Shared by /api/ask and
    /api/ask_batch so a batched question is indistinguishable from a single one
    in the audit trail / exports.
    """
    payload = answer_question(question, doc_ids, run_judge=run_judge)

    g = payload["grading"]
    global _ENTRY_SEQ
    _ENTRY_SEQ += 1
    entry_id = _ENTRY_SEQ
    payload["entry_id"] = entry_id
    SESSION_LOG.append({
        "entry_id"                        : entry_id,
        # Human-in-the-loop validation (filled in later via /api/validate)
        "validation_status"               : "not_reviewed",
        "reviewer"                        : None,
        "reviewer_comment"                : None,
        "corrected_answer"                : None,
        "review_timestamp"                : None,
        "timestamp"                       : datetime.datetime.now().isoformat(timespec="seconds"),
        "question"                        : question,
        "documents"                       : " | ".join(d["label"] for d in payload["documents"]),
        "answer"                          : payload["extraction"]["answer"],
        "model_confidence"                : payload["extraction"]["model_confidence"],
        "judge_mean"                      : g.get("judge_mean"),
        "judge_lower"                     : g.get("judge_lower"),
        "judge_upper"                     : g.get("judge_upper"),
        "judge_variance"                  : g.get("judge_variance"),
        # Grader 1 — per-judge breakdown, captured separately for each panel
        # member so exports show WHY deepseek/groq scored as they did, not
        # just the pooled consensus mean/reason.
        "judge_deepseek_confidence"       : g.get("judge_deepseek_confidence"),
        "judge_deepseek_reasoning"        : g.get("judge_deepseek_reasoning"),
        "judge_groq_confidence"           : g.get("judge_groq_confidence"),
        "judge_groq_reasoning"            : g.get("judge_groq_reasoning"),
        "semantic_match"                  : g.get("semantic_match"),
        "review_priority"                 : payload["review"]["priority"],
        "error_tags"                      : payload["review"]["error_tags"],
        # Grader 4 — DeepEval Answer Relevancy
        "deepeval_answer_relevancy_score" : g.get("deepeval_answer_relevancy_score"),
        "deepeval_answer_relevancy_passed": g.get("deepeval_answer_relevancy_passed"),
        "deepeval_answer_relevancy_reason": g.get("deepeval_answer_relevancy_reason"),
        # Grader 5 — Claim Decomposition
        "claim_grounding_ratio"           : g.get("claim_grounding_ratio"),
        "claims_total"                    : g.get("claims_total"),
        "claims_supported"                : g.get("claims_supported"),
        "claims_unsupported"              : g.get("claims_unsupported"),
        "claims_contradicted"             : g.get("claims_contradicted"),
        "claims_meta"                     : g.get("claims_meta"),
        "decomposition_error"             : g.get("decomposition_error"),
        # Grader 6 — Transcript metrics
        "answer_len_tokens"               : g.get("answer_len_tokens"),
        # Grader 7 — Coverage check
        "requirements_covered"            : g.get("requirements_covered"),
        "requirements_uncovered"          : g.get("requirements_uncovered"),
        # Grader 8 — Retrieval quality distribution
        "retrieval_score_max"             : g.get("retrieval_score_max"),
        "retrieval_score_min"             : g.get("retrieval_score_min"),
        "retrieval_score_mean"            : g.get("retrieval_score_mean"),
        "retrieval_score_spread"          : g.get("retrieval_score_spread"),
        "weak_chunks_ratio"               : g.get("weak_chunks_ratio"),
        # Grader 9 — Sub-query agreement
        "subquery_count"                  : g.get("subquery_count"),
        "subquery_agreement_ratio"        : g.get("subquery_agreement_ratio"),
        "subquery_mean_hits"              : g.get("subquery_mean_hits"),
        # Grader 11 — Evidence alignment
        "evidence_quotes_total"           : g.get("evidence_quotes_total"),
        "evidence_quotes_aligned"         : g.get("evidence_quotes_aligned"),
        "evidence_alignment_rate"         : g.get("evidence_alignment_rate"),
        # Grader 12 — Context utilisation
        "chunks_total"                    : g.get("chunks_total"),
        "chunks_used"                     : g.get("chunks_used"),
        "context_utilisation"             : g.get("context_utilisation"),
        # Grader 13 — Numeric consistency
        "numbers_total"                   : g.get("numbers_total"),
        "numbers_grounded"                : g.get("numbers_grounded"),
        "numeric_consistency"             : g.get("numeric_consistency"),
    })
    return payload


@app.route("/api/ask", methods=["POST"])
def ask():
    body = request.get_json(force=True) or {}
    question = (body.get("question") or "").strip()
    doc_ids = body.get("doc_ids") or []
    run_judge = bool(body.get("judge", True))
    if not question:
        return jsonify({"error": "empty question"}), 400
    if not M.DOC_REGISTRY:
        return jsonify({"error": "no documents loaded - upload a PDF first"}), 400
    try:
        payload = record_answer(question, doc_ids, run_judge)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    return jsonify(payload)


# Hard ceiling on questions per batch request -- keeps a mis-pasted 5,000-line
# file from silently spinning the server for an hour on one request.
MAX_BATCH_QUESTIONS = 50


@app.route("/api/ask_batch", methods=["POST"])
def ask_batch():
    """
    Run a list of questions against the same document scope, one after another,
    reusing record_answer() so every successful question is logged to
    SESSION_LOG / exports exactly like a normal single ask. Failures on one
    question don't abort the rest of the batch -- each result reports its own
    ok/error status so the UI can show a per-row outcome.
    """
    body = request.get_json(force=True) or {}
    raw_questions = body.get("questions") or []
    doc_ids = body.get("doc_ids") or []
    run_judge = bool(body.get("judge", True))

    if not isinstance(raw_questions, list):
        return jsonify({"error": "questions must be a list"}), 400

    # Clean, drop blanks, de-duplicate while preserving order.
    seen = set()
    questions = []
    for q in raw_questions:
        q = (q or "").strip() if isinstance(q, str) else ""
        if q and q not in seen:
            seen.add(q)
            questions.append(q)

    if not questions:
        return jsonify({"error": "no questions provided"}), 400
    if not M.DOC_REGISTRY:
        return jsonify({"error": "no documents loaded - upload a PDF first"}), 400
    if len(questions) > MAX_BATCH_QUESTIONS:
        return jsonify({
            "error": f"batch too large - max {MAX_BATCH_QUESTIONS} questions per run "
                     f"(got {len(questions)})"
        }), 400

    results = []
    for i, question in enumerate(questions):
        try:
            payload = record_answer(question, doc_ids, run_judge)
            g = payload["grading"]
            results.append({
                "index"           : i,
                "question"        : question,
                "ok"              : True,
                "entry_id"        : payload["entry_id"],
                "answer"          : payload["extraction"]["answer"],
                "judge_mean"      : g.get("judge_mean"),
                "judge_deepseek_confidence": g.get("judge_deepseek_confidence"),
                "judge_groq_confidence"    : g.get("judge_groq_confidence"),
                "semantic_match"  : g.get("semantic_match"),
                "review_priority" : payload["review"]["priority"],
                "payload"         : payload,
            })
        except Exception as e:
            traceback.print_exc()
            results.append({
                "index"   : i,
                "question": question,
                "ok"      : False,
                "error"   : str(e),
            })

    ok_count = sum(1 for r in results if r["ok"])
    return jsonify({
        "results"    : results,
        "total"      : len(results),
        "ok_count"   : ok_count,
        "error_count": len(results) - ok_count,
    })


# Column order shared by every export format (CSV / Excel / JSON).
EXPORT_COLS = [
    "entry_id", "timestamp", "question", "documents", "answer", "model_confidence",
    # Human-in-the-loop validation
    "validation_status", "reviewer", "reviewer_comment", "corrected_answer", "review_timestamp",
    # Grader 1 — multi-judge consensus
    "judge_mean", "judge_lower", "judge_upper", "judge_variance",
    # Grader 1 — per-judge breakdown, captured separately for deepseek and groq
    "judge_deepseek_confidence", "judge_deepseek_reasoning",
    "judge_groq_confidence", "judge_groq_reasoning",
    # Grader 2 — semantic match
    "semantic_match",
    # Review / error
    "review_priority", "error_tags",
    # Grader 4 — DeepEval Answer Relevancy
    "deepeval_answer_relevancy_score", "deepeval_answer_relevancy_passed",
    "deepeval_answer_relevancy_reason",
    # Grader 5 — Claim Decomposition
    "claim_grounding_ratio", "claims_total", "claims_supported",
    "claims_unsupported", "claims_contradicted", "claims_meta", "decomposition_error",
    # Grader 6 — Transcript metrics
    "answer_len_tokens",
    # Grader 7 — Coverage check
    "requirements_covered", "requirements_uncovered",
    # Grader 8 — Retrieval quality distribution
    "retrieval_score_max", "retrieval_score_min", "retrieval_score_mean",
    "retrieval_score_spread", "weak_chunks_ratio",
    # Grader 9 — Sub-query agreement
    "subquery_count", "subquery_agreement_ratio", "subquery_mean_hits",
    # Grader 11 — Evidence alignment
    "evidence_quotes_total", "evidence_quotes_aligned", "evidence_alignment_rate",
    # Grader 12 — Context utilisation
    "chunks_total", "chunks_used", "context_utilisation",
    # Grader 13 — Numeric consistency
    "numbers_total", "numbers_grounded", "numeric_consistency",
]


@app.route("/api/validate", methods=["POST"])
def validate():
    """Record a human-in-the-loop decision (approve / accept / reject / skip)
    against a previously answered question, writing it into the audit trail so
    it flows through to every export format."""
    body = request.get_json(force=True) or {}
    entry_id = body.get("entry_id")
    status   = (body.get("status") or "").strip().lower()
    allowed  = {"approved", "accepted", "rejected", "skipped"}
    if status not in allowed:
        return jsonify({"error": f"invalid validation status '{status}'"}), 400

    try:
        entry_id = int(entry_id)
    except (TypeError, ValueError):
        return jsonify({"error": "missing or invalid entry_id"}), 400

    target = next((r for r in SESSION_LOG if r.get("entry_id") == entry_id), None)
    if target is None:
        return jsonify({"error": "unknown entry_id — nothing to validate"}), 404

    corrected = (body.get("corrected_answer") or "").strip()
    target["validation_status"] = status
    target["reviewer"]          = (body.get("reviewer") or "").strip() or None
    target["reviewer_comment"]  = (body.get("comment") or "").strip() or None
    target["corrected_answer"]  = corrected or None
    target["review_timestamp"]  = datetime.datetime.now().isoformat(timespec="seconds")

    return jsonify({
        "ok"               : True,
        "entry_id"         : entry_id,
        "validation_status": status,
        "reviewer"         : target["reviewer"],
        "review_timestamp" : target["review_timestamp"],
    })


@app.route("/api/export")
def export():
    fmt  = (request.args.get("format") or "csv").strip().lower()
    cols = EXPORT_COLS
    rows = [{c: row.get(c) for c in cols} for row in SESSION_LOG]

    # ---- JSON ---------------------------------------------------------------
    if fmt == "json":
        return Response(
            json.dumps(rows, indent=2, default=str),
            mimetype="application/json",
            headers={"Content-Disposition": "attachment; filename=oncology_qa_session.json"})

    # ---- Excel (.xlsx) ------------------------------------------------------
    if fmt in ("xlsx", "excel"):
        df = M.pd.DataFrame(rows, columns=cols)
        last_err = None
        for engine in ("openpyxl", "xlsxwriter"):
            try:
                out = io.BytesIO()
                with M.pd.ExcelWriter(out, engine=engine) as writer:
                    df.to_excel(writer, index=False, sheet_name="QA session")
                out.seek(0)
                return Response(
                    out.getvalue(),
                    mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    headers={"Content-Disposition": "attachment; filename=oncology_qa_session.xlsx"})
            except Exception as e:   # engine not installed / write failed -> try next
                last_err = e
        traceback.print_exc()
        return jsonify({
            "error": f"Excel export unavailable ({last_err}). "
                     f"Install an .xlsx writer with:  pip install openpyxl"
        }), 500

    # ---- CSV (default) ------------------------------------------------------
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=cols)
    w.writeheader()
    for row in rows:
        w.writerow(row)
    return Response(buf.getvalue(), mimetype="text/csv", headers={
        "Content-Disposition": "attachment; filename=oncology_qa_session.csv"})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=5000)
    args = ap.parse_args()
    app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024
    print(f"\n  Studio ready -> http://127.0.0.1:{args.port}\n")
    app.run(debug=False, port=args.port, threaded=True)