from __future__ import annotations

import os
import io
import re
import json
import csv
import uuid
import pickle
import hashlib
import datetime
import argparse
from collections import Counter
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple, Any

import bisect

import numpy as np
import fitz  # PyMuPDF

# =============================================================================
# 1. CONFIG
# =============================================================================

# ---------------------------------------------------------------- documents
# Point this at the long protocol PDF(s). Multiple files are supported.
PDF_PATHS = [
    os.environ.get("PROTOCOL_PDF", r"Prot_000 (1).pdf"),
    "Prot_000.pdf",
]

# ---------------------------------------------------------------- LLM setup
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "INSERT_API_KEY")

AZURE_OPENAI_ENDPOINT    = os.environ.get(
    "AZURE_OPENAI_ENDPOINT", "INSERT_ENDPOINT")
AZURE_OPENAI_API_KEY     = os.environ.get("AZURE_OPENAI_API_KEY", "INSERT_API_KEY")
AZURE_OPENAI_API_VERSION = "2024-12-01-preview"
AZURE_OPENAI_DEPLOYMENT  = "gpt-5.4-mini"

MODEL_NAME  = AZURE_OPENAI_DEPLOYMENT              # generation / extraction
JUDGE_MODEL = "llama-3.3-70b-versatile"            # expansion / groundedness


ALLOW_LLM_FALLBACK = True

# ------------------------------------------------------------- embeddings
EMBEDDING_MODEL = "all-mpnet-base-v2"
CROSS_ENCODER   = "cross-encoder/ms-marco-MiniLM-L-6-v2"   # set None to skip
USE_CROSS_ENCODER = True

# ------------------------------------------------------------- chunking
CHUNK_CHARS        = 1100
CHUNK_OVERLAP      = 220
MIN_CHUNK_CHARS    = 120

# ------------------------------------------------------------- retrieval
TOP_K_DENSE        = 40      # per sub-query, before fusion
TOP_K_BM25         = 40
FUSED_CANDIDATES   = 60      # after reciprocal-rank fusion
NEIGHBOUR_WINDOW   = 1       # pull n chunks either side of every hit
CONTEXT_CHAR_BUDGET = 60_000  # how much source text may reach the LLM per turn
SECTION_MATCH_THRESHOLD = 0.62   # similarity of query to a section title path
MAX_SECTION_CHARS   = 120_000    # whole-section guard rail
USE_LLM_ROUTER      = True       # ask the judge model to pick the scope
ROUTER_SHORTLIST    = 18         # candidate sections shown to the router
ROUTER_TOP_HITS     = 25         # strongest fused hits used for routing signals
SECTION_CONCENTRATION = 0.30     # share of top hits inside one section
SECTION_COVERAGE_TRIGGER = 0.30  # share of a section already hit -> take it all
SECTION_HIT_TRIGGER = 3          # or this many distinct chunks hit in it
MIN_LIST_ITEMS      = 4          # enumerated items that make a section "list-like"

# --------------------------------------------------------------- caching
CACHE_DIR = "vector_cache_v2"

# --------------------------------------------------------------- grounding
GROUNDED_THRESHOLD = 92
PARTIAL_THRESHOLD  = 75


# =============================================================================
# 2. DOCUMENT MODEL / PARSING / SECTIONING / CHUNKING
# =============================================================================


# =============================================================================
# Data model
# =============================================================================

@dataclass
class Section:
    sec_id: str            # e.g. "s042"
    number: str            # e.g. "3.3.1"
    title: str             # e.g. "Inclusion Criteria"
    full_title: str        # e.g. "3.3.1 Inclusion Criteria"
    path: str              # e.g. "3 STUDY POPULATION > 3.3 Study Population > 3.3.1 Inclusion Criteria"
    level: int
    page_start: int        # 0-based
    page_end: int
    char_start: int        # offset into doc.text
    char_end: int


@dataclass
class Chunk:
    chunk_id: int
    doc_id: str
    sec_id: str
    section_path: str
    page_start: int
    page_end: int
    char_start: int
    char_end: int
    text: str

    def embed_text(self) -> str:
        """Section path prefixed so an anonymous criterion still carries its
        heading context into the embedding (fixes failure mode 6)."""
        return f"{self.section_path}\n{self.text}"


@dataclass
class ProtocolDoc:
    doc_id: str
    path: str
    title: str
    version_label: str
    n_pages: int
    text: str                                   # full cleaned text
    page_offsets: List[Tuple[int, int]]         # (start, end) per page
    sections: List[Section]
    chunks: List[Chunk]
    line_starts: List[int] = field(default_factory=list)   # char offset per line
    line_x: List[float] = field(default_factory=list)      # left edge of that line
    sec_by_id: Dict[str, Section] = field(default_factory=dict)
    chunks_by_sec: Dict[str, List[int]] = field(default_factory=dict)

    def finalise(self):
        self.sec_by_id = {s.sec_id: s for s in self.sections}
        self.chunks_by_sec = {}
        for c in self.chunks:
            self.chunks_by_sec.setdefault(c.sec_id, []).append(c.chunk_id)
        return self

    def page_of_offset(self, offset: int) -> int:
        for i, (s, e) in enumerate(self.page_offsets):
            if s <= offset < e:
                return i
        return max(0, len(self.page_offsets) - 1)

    def section_text(self, sec_id: str) -> str:
        s = self.sec_by_id[sec_id]
        return self.text[s.char_start:s.char_end]

    def x_at(self, offset: int) -> float:
        """Left edge (in PDF points) of the line containing this offset.
        Indentation is how a protocol encodes nesting -- 72pt for a top-level
        item, 108pt for its children, 126pt for theirs -- and it is the only
        reliable depth signal when the same marker style ('a)') is reused at
        several levels."""
        if not self.line_starts:
            return -1.0
        i = bisect.bisect_right(self.line_starts, offset) - 1
        return self.line_x[i] if 0 <= i < len(self.line_x) else -1.0


# =============================================================================
# 1. Page text extraction + running header/footer removal
# =============================================================================

_WS = re.compile(r"[ \t]+")


def _normalise_line(line: str) -> str:
    return _WS.sub(" ", line).strip()


def detect_boilerplate(pages: List[str], min_ratio: float = 0.25) -> set:
    """
    Lines that repeat on a large fraction of pages are running headers/footers
    (document title, protocol number, drug name, 'Approved 13.0 v', page
    numbers, revision date). They are removed before chunking -- this is
    failure mode 2 above and it alone recovers a lot of embedding signal.
    """
    counts = Counter()
    for text in pages:
        seen = set()
        for raw in text.split("\n"):
            line = _normalise_line(raw)
            if not line or len(line) > 90:
                continue
            # normalise bare page numbers / dates so they collapse together
            key = re.sub(r"\d+", "#", line)
            if key not in seen:
                seen.add(key)
                counts[key] += 1
    threshold = max(3, int(len(pages) * min_ratio))
    return {k for k, v in counts.items() if v >= threshold}


def extract_page_lines(page) -> List[Tuple[str, float]]:
    """
    Lines with their left edge, in reading order. Plain get_text("text") throws
    the x-coordinate away, and with it the document's own nesting information --
    which is why depth had to be guessed from marker style before.
    A blank line is emitted between PDF blocks to preserve paragraph breaks.
    """
    out: List[Tuple[str, float]] = []
    data = page.get_text("dict")
    for block in data.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            text = _normalise_line("".join(s.get("text", "") for s in line.get("spans", [])))
            if text:
                out.append((text, round(float(line["bbox"][0]), 1)))
        if out and out[-1][0]:
            out.append(("", -1.0))
    return out


def clean_page_lines(lines: List[Tuple[str, float]],
                     boilerplate: set) -> List[Tuple[str, float]]:
    out: List[Tuple[str, float]] = []
    for line, x in lines:
        if not line:
            if out and out[-1][0]:
                out.append(("", -1.0))
            continue
        key = re.sub(r"\d+", "#", line)
        if key in boilerplate:
            continue
        if re.fullmatch(r"\d{1,4}", line):      # naked page number
            continue
        out.append((line, x))
    while out and not out[-1][0]:
        out.pop()
    return out


# =============================================================================
# 2. TOC-driven sectioning
# =============================================================================

HEADING_RE = re.compile(r"^(\d{1,2}(?:\.\d{1,2}){0,4})\s+(.{2,120})$")


def _locate_heading(doc_text: str, page_offsets, page_no: int,
                    number: str, title: str, search_from: int) -> Optional[int]:
    """
    Find the char offset where a TOC entry's heading actually starts.
    Tries, in order: '<number>\\n<title>' (how PyMuPDF emits BMS-style
    headings), '<number> <title>', bare '<number>' at line start, then the
    title alone -- all constrained to the bookmark's page +/- 1.
    """
    lo = page_offsets[max(0, page_no - 1)][0]
    hi = page_offsets[min(len(page_offsets) - 1, page_no + 1)][1]
    lo = max(lo, search_from)
    if lo >= hi:
        lo, hi = page_offsets[page_no]
    window = doc_text[lo:hi]
    t = title.strip()
    candidates = []
    if number:
        candidates += [f"{number}\n{t}", f"{number} {t}", f"\n{number}\n"]
    candidates.append(t)
    for cand in candidates:
        idx = window.find(cand)
        if idx != -1:
            return lo + idx
    # last resort: case-insensitive title match
    idx = window.lower().find(t.lower()[:60])
    return lo + idx if idx != -1 else None


def build_sections(doc: fitz.Document, doc_text: str, page_offsets) -> List[Section]:
    toc = doc.get_toc() or []
    entries = []

    for level, raw_title, page_1based in toc:
        page = max(0, min(len(page_offsets) - 1, page_1based - 1))
        raw_title = _normalise_line(raw_title)
        m = HEADING_RE.match(raw_title)
        number, title = (m.group(1), m.group(2).strip()) if m else ("", raw_title)
        entries.append({"level": level, "number": number, "title": title,
                        "full_title": raw_title, "page": page})

    if not entries:
        entries = _sections_from_text(doc_text, page_offsets)

    # resolve start offsets, monotonically increasing
    cursor = 0
    for e in entries:
        off = _locate_heading(doc_text, page_offsets, e["page"],
                              e["number"], e["title"], cursor)
        if off is None or off < cursor:
            off = max(cursor, page_offsets[e["page"]][0])
        e["char_start"] = off
        cursor = off

    # end offset = start of the next entry (any level); last runs to EOF
    sections: List[Section] = []
    stack: List[Tuple[int, str]] = []          # (level, full_title) for path
    for i, e in enumerate(entries):
        char_end = entries[i + 1]["char_start"] if i + 1 < len(entries) else len(doc_text)
        if char_end <= e["char_start"]:
            char_end = min(len(doc_text), e["char_start"] + 1)

        while stack and stack[-1][0] >= e["level"]:
            stack.pop()
        path = " > ".join([t for _, t in stack] + [e["full_title"]])
        stack.append((e["level"], e["full_title"]))

        page_start = e["page"]
        page_end = _page_of(page_offsets, char_end - 1)
        sections.append(Section(
            sec_id=f"s{i:04d}", number=e["number"], title=e["title"],
            full_title=e["full_title"], path=path, level=e["level"],
            page_start=page_start, page_end=page_end,
            char_start=e["char_start"], char_end=char_end,
        ))

    # a leading pre-TOC preamble
    if sections and sections[0].char_start > 200:
        sections.insert(0, Section(
            sec_id="s_pre", number="", title="Front matter",
            full_title="Front matter", path="Front matter", level=1,
            page_start=0, page_end=_page_of(page_offsets, sections[0].char_start - 1),
            char_start=0, char_end=sections[0].char_start))
    return sections


def _page_of(page_offsets, offset: int) -> int:
    for i, (s, e) in enumerate(page_offsets):
        if s <= offset < e:
            return i
    return len(page_offsets) - 1


def _sections_from_text(doc_text: str, page_offsets) -> List[dict]:
    """Fallback for PDFs with no bookmarks: numbered headings on their own line."""
    entries = []
    for m in re.finditer(r"(?m)^(\d{1,2}(?:\.\d{1,2}){0,3})\s+([A-Z][^\n]{3,90})$", doc_text):
        number, title = m.group(1), m.group(2).strip()
        entries.append({"level": number.count(".") + 1, "number": number,
                        "title": title, "full_title": f"{number} {title}",
                        "page": _page_of(page_offsets, m.start())})
    if not entries:
        entries = [{"level": 1, "number": "", "title": "Document",
                    "full_title": "Document", "page": 0}]
    return entries


# =============================================================================
# 3. Enumerated-item segmentation  (the completeness guarantee)
# =============================================================================

MARKER_STYLES = [
    ("paren_num",   re.compile(r"^\s*(\d{1,3})\)(?:\s+|\s*$)")),
    ("paren_alpha", re.compile(r"^\s*([a-z])\)(?:\s+|\s*$)")),
    ("roman",       re.compile(r"^\s*(x{0,3}(?:ix|iv|v?i{1,3}|v))\.(?:\s+|\s*$)", re.I)),
    ("dot_num",     re.compile(r"^\s*(\d{1,3})\.(?:\s+(?=[A-Za-z(])|\s*$)")),
    ("bullet",      re.compile(r"^\s*([\u2022\u25cf\u25aa\u2013\-])(?:\s+|\s*$)")),
]
_ITEM_RE = [(rx, i) for i, (_, rx) in enumerate(MARKER_STYLES)]


def _marker_of(line: str):
    for name, rx in MARKER_STYLES:
        m = rx.match(line)
        if m:
            return name, m.group(1)
    return None, None


_ROMAN = {"i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5, "vi": 6, "vii": 7,
          "viii": 8, "ix": 9, "x": 10, "xi": 11, "xii": 12, "xiii": 13,
          "xiv": 14, "xv": 15, "xvi": 16, "xvii": 17, "xviii": 18,
          "xix": 19, "xx": 20}


def _marker_value(style: str, marker: str) -> Optional[int]:
    """Ordinal value of a list marker, or None for bullets."""
    if style in ("paren_num", "dot_num"):
        return int(marker)
    if style == "paren_alpha":
        return ord(marker.lower()) - 96
    if style == "roman":
        return _ROMAN.get(marker.lower())
    return None


def split_marker(text: str) -> Tuple[str, str]:
    """Separate a leading list marker from the body: 'a) Subjects must...' ->
    ('a)', 'Subjects must...'). The UI shows the marker once, as the bullet."""
    for _, rx in MARKER_STYLES:
        m = rx.match(text)
        if m:
            return text[m.start():m.end()].strip(), text[m.end():].strip()
    return "", text.strip()


def _depth_from_indent(xs: List[float], tol: float = 6.0) -> Optional[List[int]]:
    """Map left-edge positions to nesting depths by clustering them.
    Returns None when the x information is missing (older cache, odd PDF)."""
    usable = [x for x in xs if x is not None and x >= 0]
    if len(usable) < max(2, int(0.8 * len(xs))):
        return None
    levels: List[float] = []
    for x in sorted(set(usable)):
        if not levels or x - levels[-1] > tol:
            levels.append(x)
    def level_of(x: float) -> int:
        if x is None or x < 0:
            return 0
        best, best_d = 0, float("inf")
        for i, lv in enumerate(levels):
            d = abs(x - lv)
            if d < best_d:
                best, best_d = i, d
        return best
    return [level_of(x) for x in xs]


def segment_enumerated_items(text: str, base_offset: int = 0,
                             max_depth: Optional[int] = None,
                             x_lookup=None) -> List[dict]:
    """
    Split a section into its enumerated items *deterministically*, hierarchy
    and char offsets intact. For "give me ALL the inclusion criteria" this is
    the completeness guarantee: the list is produced by structure, not by
    model recall. The LLM is only ever asked to label and normalise items it
    has already been handed.

    Returns a flat list, each entry carrying:
        depth, marker, label (heading line), text (full item), parents (path),
        char_start/char_end (absolute, for source highlighting).
    """
    lines = text.split("\n")
    offsets, cursor = [], 0
    for ln in lines:
        offsets.append(cursor)
        cursor += len(ln) + 1

    marks = []                       # (line_idx, style_name, marker)
    for i, ln in enumerate(lines):
        style, marker = _marker_of(ln)
        if style:
            marks.append((i, style, marker))
    if not marks:
        return []

    # Depth: the document's own indentation when we have it, otherwise the
    # order in which marker styles first appear. Indentation is what
    # distinguishes an 'a)' nested three levels deep from a top-level 'a)'.
    depths: Optional[List[int]] = None
    if x_lookup is not None:
        depths = _depth_from_indent(
            [x_lookup(base_offset + offsets[i]) for i, _, _ in marks])

    first_seen = {}
    for _, style, _ in marks:
        first_seen.setdefault(style, len(first_seen))
    depth_of = (lambda n, style: depths[n]) if depths is not None \
        else (lambda n, style: first_seen[style])

    # ---- reject false markers -------------------------------------------
    validated, last_val, kept_depths = [], {}, []
    for n, (line_idx, style, marker) in enumerate(marks):
        depth = depth_of(n, style)
        val = _marker_value(style, marker)
        if val is None:                                   # bullets
            validated.append((line_idx, style, marker))
            kept_depths.append(depth)
            last_val[depth] = None
            continue
        prev = last_val.get(depth)
        # Monotonic with a small tolerance: a marker may be missed by text
        # extraction (gap of 1-2), but a *decrease* means this is not a list
        # marker at all -- it is a wrapped line that happens to start with
        # something like "1) disease." Recall matters more than perfect
        # nesting here, so tolerate gaps, reject only regressions.
        ok = (prev is None) or (prev < val <= prev + 3) or (depth > 0 and val == 1)
        if ok:
            validated.append((line_idx, style, marker))
            kept_depths.append(depth)
            last_val[depth] = val
            for d in list(last_val):                      # deeper runs reset
                if d > depth:
                    last_val.pop(d)
    marks = validated
    if not marks:
        return []

    # renumber depths so they start at 0 and have no gaps
    used = sorted(set(kept_depths))
    remap = {d: i for i, d in enumerate(used)}

    items, stack = [], []            # stack of (depth, label)
    for n, (line_idx, style, marker) in enumerate(marks):
        depth = remap[kept_depths[n]]
        end_line = marks[n + 1][0] if n + 1 < len(marks) else len(lines)
        block = "\n".join(lines[line_idx:end_line]).rstrip()
        if len(block.strip()) < 5:
            continue

        while stack and stack[-1][0] >= depth:
            stack.pop()
        parents = [lbl for _, lbl in stack]

        label = re.sub(r"\s+", " ", lines[line_idx].strip())
        stack.append((depth, label))
        body = re.sub(r"\s*\n\s*", " ", block).strip()
        marker_text, clean_text = split_marker(body)

        if max_depth is not None and depth > max_depth:
            continue

        s = base_offset + offsets[line_idx]
        items.append({
            "index": len(items) + 1,
            "depth": depth,
            "style": style,
            "marker": marker,
            "label": label,
            "parents": parents,
            "marker": marker_text,
            "text": clean_text or body,
            "raw": block,
            "char_start": s,
            "char_end": s + len(block),
        })
    return items


def group_items(items: List[dict]) -> List[dict]:
    """
    Fold the flat item list into the heading + entries shape the studio renders.

    A depth-0 item becomes a HEADING only if something is actually nested under
    it ("2) Target Population" has children; a bullet in a flat list does not).
    Everything else becomes an entry, at whatever depth it sits, so nothing is
    dropped -- the previous version kept only depth 1 and silently discarded
    deeper criteria, and turned a flat bullet list into headings with no text.
    """
    groups: List[dict] = []
    current: Optional[dict] = None
    current_is_heading = False

    def new_group(heading: str, it: dict) -> dict:
        g = {"heading": heading, "char_start": it["char_start"],
             "char_end": it["char_end"], "items": []}
        groups.append(g)
        return g

    for i, it in enumerate(items):
        has_child = (i + 1 < len(items)) and (items[i + 1]["depth"] > it["depth"])

        if it["depth"] == 0 and has_child:
            current = new_group(it["label"], it)
            current_is_heading = True
            continue

        if current is None or (it["depth"] == 0 and current_is_heading):
            current = new_group("", it)
            current_is_heading = False

        current["items"].append(it)
        current["char_end"] = max(current["char_end"], it["char_end"])

    return [g for g in groups if g["items"]]


# =============================================================================
# 4. Section-aware, offset-preserving chunking
# =============================================================================

def _split_blocks(text: str) -> List[Tuple[int, str]]:
    """Split into (offset, block) at list-item starts and blank lines so a
    chunk boundary never lands mid-criterion."""
    lines = text.split("\n")
    offsets, cursor = [], 0
    for ln in lines:
        offsets.append(cursor)
        cursor += len(ln) + 1

    blocks, buf, buf_start = [], [], 0
    def flush():
        if buf:
            blocks.append((buf_start, "\n".join(buf)))

    for i, ln in enumerate(lines):
        is_item = any(rx.match(ln) for rx, _ in _ITEM_RE)
        if (is_item or not ln.strip()) and buf:
            flush()
            buf, buf_start = [], offsets[i]
        if not buf:
            buf_start = offsets[i]
        buf.append(ln)
    flush()

    # Hard-split any single block bigger than a chunk (dense tables, long
    # narrative paragraphs). Without this, one block becomes one 10k-char
    # chunk and everything past ~384 tokens is dropped at embed time.
    out = []
    for off, b in blocks:
        if not b.strip():
            continue
        if len(b) <= CHUNK_CHARS:
            out.append((off, b))
            continue
        buf, buf_off, local = [], off, 0
        for ln in b.split("\n"):
            if sum(len(x) + 1 for x in buf) + len(ln) > CHUNK_CHARS and buf:
                out.append((buf_off, "\n".join(buf)))
                buf_off = off + local
                buf = []
            buf.append(ln)
            local += len(ln) + 1
        if buf:
            out.append((buf_off, "\n".join(buf)))
    return out


def chunk_section(doc_text: str, section: Section, doc_id: str,
                  start_id: int) -> List[Chunk]:
    raw = doc_text[section.char_start:section.char_end]
    blocks = _split_blocks(raw)
    if not blocks:
        return []

    chunks, cur, cur_start, cur_len = [], [], None, 0

    def emit(next_start=None):
        nonlocal cur, cur_start, cur_len
        if not cur:
            return
        body = "\n".join(b for _, b in cur).strip()
        if len(body) >= MIN_CHUNK_CHARS or not chunks:
            abs_start = section.char_start + cur[0][0]
            abs_end = abs_start + len("\n".join(b for _, b in cur))
            chunks.append(Chunk(
                chunk_id=start_id + len(chunks), doc_id=doc_id,
                sec_id=section.sec_id, section_path=section.path,
                page_start=0, page_end=0,
                char_start=abs_start, char_end=min(abs_end, section.char_end),
                text=body))
        cur, cur_start, cur_len = [], None, 0

    for off, block in blocks:
        blen = len(block)
        if cur_len + blen > CHUNK_CHARS and cur:
            tail = []
            acc = 0
            for item in reversed(cur):              # carry overlap forward
                acc += len(item[1])
                tail.insert(0, item)
                if acc >= CHUNK_OVERLAP:
                    break
            emit()
            cur = list(tail)
            cur_len = sum(len(b) for _, b in cur)
        if not cur:
            cur_start = off
        cur.append((off, block))
        cur_len += blen + 1
    emit()
    return chunks


# =============================================================================
# 5. Build (with caching)
# =============================================================================

def _fingerprint(path: str) -> str:
    st = os.stat(path)
    raw = (f"{path}|{st.st_size}|{st.st_mtime}|{CHUNK_CHARS}"
           f"|{CHUNK_OVERLAP}|v3")
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _doc_version(doc: fitz.Document, path: str) -> str:
    meta = doc.metadata or {}
    for key in ("creationDate", "modDate"):
        m = re.match(r"D:(\d{4})(\d{2})", meta.get(key) or "")
        if m:
            return f"{m.group(1)}-{m.group(2)}"
    return datetime.date.fromtimestamp(os.path.getmtime(path)).strftime("%Y-%m")


def _pack(pdoc: "ProtocolDoc") -> dict:
    """Cache as plain dicts, not pickled class instances -- the cache must load
    identically whether pipeline.py is imported or run as __main__."""
    return {
        "doc_id": pdoc.doc_id, "path": pdoc.path, "title": pdoc.title,
        "version_label": pdoc.version_label, "n_pages": pdoc.n_pages,
        "text": pdoc.text, "page_offsets": pdoc.page_offsets,
        "sections": [asdict(s) for s in pdoc.sections],
        "chunks": [asdict(c) for c in pdoc.chunks],
        "line_starts": pdoc.line_starts, "line_x": pdoc.line_x,
    }


def _unpack(d: dict) -> "ProtocolDoc":
    return ProtocolDoc(
        doc_id=d["doc_id"], path=d["path"], title=d["title"],
        version_label=d["version_label"], n_pages=d["n_pages"],
        text=d["text"], page_offsets=d["page_offsets"],
        sections=[Section(**s) for s in d["sections"]],
        chunks=[Chunk(**c) for c in d["chunks"]],
        line_starts=d.get("line_starts", []), line_x=d.get("line_x", []),
    ).finalise()


def load_protocol(path: str, use_cache: bool = True) -> ProtocolDoc:
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_file = os.path.join(
        CACHE_DIR,
        f"{os.path.basename(path).replace('.pdf', '')}__{_fingerprint(path)}.pkl")

    if use_cache and os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            print(f"    [cache hit] {os.path.basename(path)}")
            return _unpack(pickle.load(f))

    print(f"    [parsing] {os.path.basename(path)}")
    doc = fitz.open(path)
    raw_pages = [extract_page_lines(p) for p in doc]
    boiler = detect_boilerplate(["\n".join(l for l, _ in page) for page in raw_pages])
    print(f"    [clean] removed {len(boiler)} running header/footer line patterns")

    pieces, page_offsets, line_starts, line_x, cursor = [], [], [], [], 0
    for page in raw_pages:
        page_start = cursor
        for line, x in clean_page_lines(page, boiler):
            line_starts.append(cursor)
            line_x.append(x)
            pieces.append(line + "\n")
            cursor += len(line) + 1
        pieces.append("\n")
        cursor += 1
        page_offsets.append((page_start, cursor))
    doc_text = "".join(pieces)

    sections = build_sections(doc, doc_text, page_offsets)
    print(f"    [sections] {len(sections)} sections from "
          f"{'PDF bookmarks' if doc.get_toc() else 'heading regex'}")

    doc_id = os.path.basename(path).replace(".pdf", "")
    chunks: List[Chunk] = []
    for sec in sections:
        for c in chunk_section(doc_text, sec, doc_id, len(chunks)):
            c.page_start = _page_of(page_offsets, c.char_start)
            c.page_end = _page_of(page_offsets, max(c.char_start, c.char_end - 1))
            chunks.append(c)
    print(f"    [chunks] {len(chunks)} section-aware chunks "
          f"(avg {int(np.mean([len(c.text) for c in chunks]))} chars)")

    pdoc = ProtocolDoc(
        doc_id=doc_id, path=path,
        title=(doc.metadata or {}).get("title") or doc_id,
        version_label=_doc_version(doc, path),
        n_pages=len(doc), text=doc_text, page_offsets=page_offsets,
        sections=sections, chunks=chunks,
        line_starts=line_starts, line_x=line_x).finalise()

    with open(cache_file, "wb") as f:
        pickle.dump(_pack(pdoc), f)
    return pdoc


def load_all(paths=None, use_cache: bool = True) -> List[ProtocolDoc]:
    docs = []
    for p in (paths or PDF_PATHS):
        if not os.path.exists(p):
            print(f"    [skip] not found: {p}")
            continue
        docs.append(load_protocol(p, use_cache))
    return docs


# =============================================================================
# 7. RETRIEVAL
# =============================================================================



# --- optional heavy deps -----------------------------------------------------
try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
    _ST_AVAILABLE = True
except Exception:                                            # pragma: no cover
    _ST_AVAILABLE = False

try:
    import faiss
    _FAISS_AVAILABLE = True
except Exception:                                            # pragma: no cover
    _FAISS_AVAILABLE = False

from rank_bm25 import BM25Okapi


_TOKEN = re.compile(r"[a-z0-9]+")
STOP = set("""a an the of and or to in for with on at by is are be was were as
that this these those from into per which who whom what when where how must
should may can will shall not no if then than there their its it's it""".split())


_SUFFIXES = ("ational", "ations", "ation", "ments", "ment", "ings", "ing",
             "ives", "ive", "ies", "ical", "ally", "als", "al", "ors", "or",
             "ers", "er", "ed", "es", "s")


def _stem(word: str) -> str:
    """Very small suffix stripper. Not linguistics -- just enough that
    'discontinuing' and 'Discontinuation', or 'statistical' and 'statistics',
    land on the same token. Applies everywhere (BM25 index, section titles,
    queries), so no vocabulary is hard-coded anywhere."""
    if len(word) <= 4 or word.isdigit():
        return word
    for suf in _SUFFIXES:
        if word.endswith(suf) and len(word) - len(suf) >= 4:
            return word[:-len(suf)]
    return word


def tokenize(text: str) -> List[str]:
    return [_stem(t) for t in _TOKEN.findall(text.lower())
            if t not in STOP and len(t) > 1]


# =============================================================================
# Result containers
# =============================================================================

@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float
    doc_id: str


@dataclass
class RetrievalResult:
    query: str
    mode: str                              # "section" | "hybrid"
    doc: ProtocolDoc
    chunks: List[RetrievedChunk]
    sections: List[Section] = field(default_factory=list)
    sub_queries: List[str] = field(default_factory=list)
    diagnostics: Dict = field(default_factory=dict)

    # ---- source view -------------------------------------------------------
    def source_view(self) -> Tuple[str, List[dict]]:
        """
        Concatenate the retrieved text into the single string the studio's
        left-hand pane renders, and return a doc-offset -> view-offset map so
        extracted items can be highlighted in it.
        """
        parts, mapping, cursor = [], [], 0
        ordered = sorted(self.chunks, key=lambda r: (r.chunk.char_start))
        merged: List[Tuple[int, int]] = []
        for r in ordered:                                   # merge adjacency
            s, e = r.chunk.char_start, r.chunk.char_end
            if merged and s <= merged[-1][1] + 2:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append((s, e))

        for s, e in merged:
            sec = self._section_for(s)
            header = f"\n[{sec.full_title}]  (p. {self.doc.page_of_offset(s) + 1})\n" if sec else ""
            if header:
                parts.append(header)
                cursor += len(header)
            body = self.doc.text[s:e].strip("\n") + "\n"
            parts.append(body)
            mapping.append({"doc_start": s, "doc_end": e, "view_start": cursor})
            cursor += len(body)
        return "".join(parts), mapping

    def _section_for(self, offset: int) -> Optional[Section]:
        for sec in self.doc.sections:
            if sec.char_start <= offset < sec.char_end:
                return sec
        return None

    def context_chars(self) -> int:
        return sum(len(r.chunk.text) for r in self.chunks)


# =============================================================================
# Retriever
# =============================================================================

class ProtocolRetriever:

    def __init__(self, docs: Optional[List[ProtocolDoc]] = None,
                 use_embeddings: bool = True):
        self.docs = docs if docs is not None else load_all()
        if not self.docs:
            raise RuntimeError("No protocol documents loaded -- check PDF_PATHS")
        self.doc = self.docs[0]                    # studio works one doc at a time
        self.chunks = self.doc.chunks
        self.sections = self.doc.sections

        print("################ Building retrieval structures ################")
        # ---- lexical -------------------------------------------------------
        self.bm25 = BM25Okapi([tokenize(c.embed_text()) for c in self.chunks])
        print(f"    [bm25] {len(self.chunks)} chunks indexed")

        # ---- dense ---------------------------------------------------------
        self.embedder = None
        self.index = None
        self.section_embeddings = None
        if use_embeddings and _ST_AVAILABLE and _FAISS_AVAILABLE:
            try:
                self._build_dense()
            except Exception as e:                              # pragma: no cover
                print(f"    [dense] unavailable ({e}) -- falling back to lexical only")
        else:
            print("    [dense] sentence-transformers/faiss not available -- lexical only")

        # ---- cross-encoder --------------------------------------------------
        self._item_counts: Dict[str, int] = {}     # section -> enumerated items
        self._nav_flags: Dict[str, bool] = {}      # section -> is it a TOC/index?
        self.reranker = None
        if USE_CROSS_ENCODER and _ST_AVAILABLE and CROSS_ENCODER:
            try:
                self.reranker = CrossEncoder(CROSS_ENCODER)
                print(f"    [rerank] {CROSS_ENCODER}")
            except Exception as e:                              # pragma: no cover
                print(f"    [rerank] unavailable ({e})")

    # ------------------------------------------------------------------ dense
    def _build_dense(self):
        cache = os.path.join(CACHE_DIR,
                             f"emb__{self.doc.doc_id}__{len(self.chunks)}.pkl")
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)
        if os.path.exists(cache):
            with open(cache, "rb") as f:
                chunk_emb, sec_emb = pickle.load(f)
            print("    [dense] embeddings loaded from cache")
        else:
            print(f"    [dense] embedding {len(self.chunks)} chunks...")
            chunk_emb = self.embedder.encode(
                [c.embed_text() for c in self.chunks], batch_size=16,
                convert_to_numpy=True, normalize_embeddings=True,
                show_progress_bar=True).astype(np.float32)
            sec_emb = self.embedder.encode(
                [s.path for s in self.sections], batch_size=32,
                convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
            os.makedirs(CACHE_DIR, exist_ok=True)
            with open(cache, "wb") as f:
                pickle.dump((chunk_emb, sec_emb), f)

        self.chunk_embeddings = chunk_emb
        self.section_embeddings = sec_emb
        self.index = faiss.IndexFlatIP(chunk_emb.shape[1])
        self.index.add(chunk_emb)
        print(f"    [dense] FAISS index: {self.index.ntotal} vectors")

    # ------------------------------------------------------------- routing
    def _section_scores(self, query: str) -> np.ndarray:
        """Similarity of the query to every section *title path*."""
        if self.section_embeddings is not None and self.embedder is not None:
            q = self.embedder.encode([query], convert_to_numpy=True,
                                     normalize_embeddings=True).astype(np.float32)
            return self.section_embeddings @ q[0]
        # lexical fallback: containment of the question's terms in the title
        # (containment beats F1 here -- "exclusion criteria" is fully contained
        # in "3.3.2 Exclusion Criteria" even though the title is longer)
        qt = set(tokenize(query))
        out = []
        for s in self.sections:
            best = 0.0
            for field_text in (s.title, s.full_title, s.path):
                st = set(tokenize(field_text))
                if not st or not qt:
                    continue
                containment = len(qt & st) / len(qt)
                f1 = 2 * len(qt & st) / (len(qt) + len(st))
                best = max(best, containment * 0.85 + f1 * 0.15)
            out.append(best)
        return np.array(out, dtype=np.float32)

    def is_navigational(self, sec_id: str) -> bool:
        """True for tables of contents, figure/table lists and similar
        navigation furniture: mostly dotted leaders and page numbers. Detected
        from the text itself, so it needs no list of section names."""
        if sec_id not in self._nav_flags:
            text = self.doc.section_text(sec_id)
            lines = [l for l in text.split("\n") if l.strip()]
            if len(lines) < 10:
                self._nav_flags[sec_id] = False
            else:
                leaders = sum(1 for l in lines
                              if re.search(r"\.{3,}\s*\d+\s*$", l)
                              or re.match(r"^\s*\d+(\.\d+)*\s+\S.*\s\d{1,3}\s*$", l))
                self._nav_flags[sec_id] = leaders / len(lines) > 0.4
        return self._nav_flags[sec_id]

    def section_item_count(self, sec_id: str) -> int:
        """How many enumerated items a section contains (cached).
        This is a property of the *document*, not of the question, so it works
        for any topic: a section that is a list behaves like a list."""
        if sec_id not in self._item_counts:
            sec = self.doc.sec_by_id[sec_id]
            items = segment_enumerated_items(self.doc.section_text(sec_id),
                                             sec.char_start,
                                             x_lookup=self.doc.x_at)
            self._item_counts[sec_id] = sum(len(g["items"])
                                            for g in group_items(items))
        return self._item_counts[sec_id]

    # ---- signal (c): grammatical enumeration cues, no topic vocabulary ----
    @staticmethod
    def enumeration_cues(query: str) -> dict:
        """
        Does the *grammar* of the question ask for a set rather than a fact?
        Deliberately topic-free -- it looks at quantifiers, plural heads and
        interrogative form, so it behaves the same for criteria, prohibited
        medications, visit procedures or anything else in the protocol.
        """
        q = " " + query.lower().strip() + " "
        hits = []
        if re.search(r"\b(all|every|each|any|both)\b", q):
            hits.append("quantifier")
        if re.search(r"\b(list|enumerate|itemi[sz]e|outline|summari[sz]e|overview|describe)\b", q):
            hits.append("enumeration verb")
        if re.search(r"\b(complete|entire|full|whole|comprehensive|exhaustive)\b", q):
            hits.append("completeness adjective")
        if re.search(r"\bwhat are\b|\bwhich\b|\bhow many\b|\bwhat kinds?\b|\bwhat types?\b", q):
            hits.append("set-seeking interrogative")
        # plural head noun: "criteria", "medications", "procedures", "endpoints"
        tail = [w for w in tokenize(query)[-3:] if len(w) > 3]
        if any(w.endswith("s") for w in tail) or "criteria" in tokenize(query):
            hits.append("plural head noun")
        if re.search(r"\bwhat is the\b|\bwhen (is|does|must)\b|\bhow much\b|\bwho\b|\bwhere\b", q) \
                and "quantifier" not in hits:
            hits.append("-specific fact form")            # negative signal

        positives = [h for h in hits if not h.startswith("-")]
        fact_form = any(h.startswith("-") for h in hits)
        # a single weak cue does not outweigh an explicit single-fact question
        wants_set = bool(positives) and (not fact_form or len(positives) >= 2)
        return {"cues": hits, "wants_set": wants_set,
                "score": round(min(1.0, len(positives) / 3.0), 2)}

    # ---- signal (a): LLM scope decision over a shortlist ------------------
    def _llm_route(self, query: str, candidates: List[Tuple[Section, float]]) -> Optional[dict]:
        if not (USE_LLM_ROUTER and llm_available() and candidates):
            return None
        listing = "\n".join(
            f"{i}. {s.full_title}  (pp. {s.page_start + 1}-{s.page_end + 1}, "
            f"{s.char_end - s.char_start:,} chars, "
            f"{self.section_item_count(s.sec_id)} enumerated items)"
            for i, (s, _) in enumerate(candidates, 1))
        prompt = f"""You are routing a question about a clinical study protocol.

QUESTION: {query}

CANDIDATE SECTIONS (retrieved as most relevant):
{listing}

Decide the SCOPE needed to answer completely and faithfully:
- "section": the answer is the content of one or more of these sections in full
  (any question whose honest answer is a set, a list, a procedure, or the whole
  of a topic -- if omitting part of a section would make the answer wrong or
  incomplete, choose this).
- "passage": the answer is one specific fact, value, date or definition that
  sits inside a passage; returning whole sections would add only noise.

Return ONLY: {{"scope": "section"|"passage", "sections": [<candidate numbers, most relevant first>], "reason": "<one short sentence>"}}"""
        try:
            out = parse_json(judge_complete(prompt, max_tokens=250))
            scope = str(out.get("scope", "")).lower()
            if scope not in ("section", "passage"):
                return None
            nums = [int(n) for n in out.get("sections", [])
                    if isinstance(n, (int, float, str)) and str(n).strip().isdigit()]
            picked = [candidates[n - 1][0] for n in nums if 1 <= n <= len(candidates)]
            return {"scope": scope, "sections": picked,
                    "reason": str(out.get("reason", ""))[:200]}
        except Exception as e:
            print(f"    [router] LLM routing unavailable ({e}) -- using signals")
            return None

    def route(self, query: str, sub_queries: Optional[List[str]] = None,
              fused: Optional[Dict[int, float]] = None) -> Tuple[str, List[Section], dict]:
        """
        Decide whether this question needs whole sections or targeted passages.
        No topic keyword list is involved: the decision comes from where the
        retrieved evidence actually lands, what shape that part of the document
        has, and how the question is phrased -- so it generalises to any
        question and any protocol.
        """
        sub_queries = sub_queries or [query]
        if fused is None:
            fused, _ = self._fuse(sub_queries)

        title_scores = self._section_scores(query)

        # ---- signal (b): where does the retrieved evidence concentrate? ----
        # only the strongest hits carry routing signal: the long tail of the
        # fused pool is noise and would flatten every concentration score
        top_hits = sorted(fused.items(), key=lambda kv: -kv[1])[:ROUTER_TOP_HITS]
        mass: Dict[str, float] = {}
        hits: Dict[str, int] = {}
        for cid, sc in top_hits:
            sid = self.chunks[cid].sec_id
            if self.is_navigational(sid):          # TOC hits are not evidence
                continue
            mass[sid] = mass.get(sid, 0.0) + sc
            hits[sid] = hits.get(sid, 0) + 1
        total_mass = sum(mass.values()) or 1.0
        by_mass = sorted(mass.items(), key=lambda kv: -kv[1])
        dominant_id, dominant_mass = (by_mass[0] if by_mass else (None, 0.0))
        concentration = dominant_mass / total_mass

        # ---- shortlist: top sections by title match AND by retrieved mass --
        shortlist: List[Tuple[Section, float]] = []
        seen = set()
        for idx in np.argsort(-title_scores)[:10]:
            s = self.sections[int(idx)]
            if s.sec_id not in seen and not self.is_navigational(s.sec_id):
                seen.add(s.sec_id)
                shortlist.append((s, float(title_scores[int(idx)])))
        for sid, m in by_mass[:ROUTER_SHORTLIST]:
            if sid not in seen:
                seen.add(sid)
                shortlist.append((self.doc.sec_by_id[sid], m / total_mass))
        shortlist = shortlist[:ROUTER_SHORTLIST]

        cues = self.enumeration_cues(query)
        info = {
            "concentration": round(float(concentration), 3),
            "dominant_section": (self.doc.sec_by_id[dominant_id].full_title
                                 if dominant_id else None),
            "dominant_items": self.section_item_count(dominant_id) if dominant_id else 0,
            "cue_score": round(cues["score"], 2),
            "cues": cues["cues"],
            "best_title_score": round(float(title_scores.max()) if len(title_scores) else 0.0, 3),
        }

        # ---- (a) LLM router decides first when it is available -------------
        verdict = self._llm_route(query, shortlist)
        if verdict:
            info["router"] = f"llm: {verdict['reason']}"
            if verdict["scope"] == "section" and verdict["sections"]:
                picked = self._expand_family(verdict["sections"], title_scores)
                if picked:
                    return "section", picked, info
            return "hybrid", [], info

        # ---- fallback: combine signals (b) and (c) -------------------------
        ranked_titles = [i for i in np.argsort(-title_scores)
                         if not self.is_navigational(self.sections[int(i)].sec_id)]
        best_idx = int(ranked_titles[0]) if ranked_titles else int(np.argmax(title_scores))
        best_sec = self.sections[best_idx]
        best_title = float(title_scores[best_idx])

        # candidate section: a confident title match, else where the evidence
        # concentrates
        candidate, why = None, ""
        if best_title >= SECTION_MATCH_THRESHOLD:
            candidate, why = best_sec, f"title match {best_title:.2f}"
        elif dominant_id and (concentration >= SECTION_CONCENTRATION
                              or hits.get(dominant_id, 0) >= SECTION_HIT_TRIGGER * 2):
            candidate, why = self.doc.sec_by_id[dominant_id], \
                (f"evidence concentration {concentration:.2f} "
                 f"({hits.get(dominant_id, 0)} of the top hits)")

        if candidate is not None:
            list_like = self.section_item_count(candidate.sec_id) >= MIN_LIST_ITEMS
            if cues["wants_set"] or list_like:
                picked = self._expand_family([candidate], title_scores)
                if picked:
                    info["router"] = (
                        f"signals: {why}, list_like={list_like}, "
                        f"cues={cues['cues'] or 'none'}")
                    return "section", picked, info

        info["router"] = (f"signals: no section commitment "
                          f"(best title {best_title:.2f}, "
                          f"concentration {concentration:.2f}) -- passage question")
        return "hybrid", [], info

    def _expand_family(self, picked: List[Section],
                       title_scores: np.ndarray) -> List[Section]:
        """Pull in child sections of anything picked (3.3.1 -> 3.3.1.1) and drop
        the selection if it would blow the whole-section guard rail."""
        out = {s.sec_id: s for s in picked}
        for base in list(picked):
            if not base.number:
                continue
            for s in self.sections:
                if s.number.startswith(base.number + ".") and s.sec_id not in out:
                    out[s.sec_id] = s
        chosen = sorted(out.values(), key=lambda s: s.char_start)
        total = sum(s.char_end - s.char_start for s in chosen)
        if total > MAX_SECTION_CHARS:
            chosen = sorted(picked, key=lambda s: s.char_start)[:1]
        return chosen

    # ------------------------------------------------------------ retrieval
    def _fuse(self, sub_queries: List[str]) -> Tuple[Dict[int, float], List[set]]:
        """Dense + BM25 across every sub-query, merged by reciprocal rank."""
        rrf: Dict[int, float] = {}
        per_sub_sets: List[set] = []
        for sq in sub_queries:
            dense_ids: List[int] = []
            if self.index is not None:
                q = self.embedder.encode([sq], convert_to_numpy=True,
                                         normalize_embeddings=True).astype(np.float32)
                _, idxs = self.index.search(q, min(TOP_K_DENSE, len(self.chunks)))
                dense_ids = [int(i) for i in idxs[0]]

            bm_scores = self.bm25.get_scores(tokenize(sq))
            bm_ids = [int(i) for i in np.argsort(-bm_scores)[:TOP_K_BM25]]

            for rank, cid in enumerate(dense_ids):
                rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (60 + rank)
            for rank, cid in enumerate(bm_ids):
                rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (60 + rank)
            per_sub_sets.append(set(dense_ids) | set(bm_ids))
        return rrf, per_sub_sets

    def retrieve(self, query: str, sub_queries: Optional[List[str]] = None,
                 force_mode: Optional[str] = None) -> RetrievalResult:
        sub_queries = sub_queries or [query]
        rrf, per_sub_sets = self._fuse(sub_queries)

        mode, sections, routing = self.route(query, sub_queries, rrf)
        if force_mode:
            routing["router"] = f"forced by user: {force_mode} " \
                                f"(auto would have chosen {mode})"
            mode = force_mode
            if force_mode == "hybrid":
                sections = []
            elif force_mode == "section" and not sections:
                # user forced whole-section: take the section the evidence
                # concentrates in, whatever the router thought
                mass: Dict[str, float] = {}
                for cid, sc in rrf.items():
                    sid = self.chunks[cid].sec_id
                    mass[sid] = mass.get(sid, 0.0) + sc
                if mass:
                    top = max(mass.items(), key=lambda kv: kv[1])[0]
                    sections = self._expand_family([self.doc.sec_by_id[top]],
                                                   self._section_scores(query))

        if mode == "section" and sections:
            picked_ids = []
            for s in sections:
                picked_ids.extend(self.doc.chunks_by_sec.get(s.sec_id, []))
            picked_ids = sorted(set(picked_ids))
            chunks = [RetrievedChunk(self.chunks[i], 1.0, self.doc.doc_id)
                      for i in picked_ids]
            diag = {
                "mode": "section",
                "sections": [s.full_title for s in sections],
                "section_chars": sum(s.char_end - s.char_start for s in sections),
                "chunks_returned": len(chunks),
                "routing": routing,
                "note": "whole-section retrieval -- no top-k truncation",
            }
            return RetrievalResult(query, "section", self.doc, chunks,
                                   sections, sub_queries, diag)

        # ------------------------------------------------------ hybrid arm
        ranked = sorted(rrf.items(), key=lambda kv: -kv[1])[:FUSED_CANDIDATES]
        cand_ids = [cid for cid, _ in ranked]

        # neighbour expansion -- a criterion's continuation is the next chunk
        expanded = set()
        for cid in cand_ids:
            for d in range(-NEIGHBOUR_WINDOW, NEIGHBOUR_WINDOW + 1):
                n = cid + d
                if 0 <= n < len(self.chunks) and \
                        self.chunks[n].sec_id == self.chunks[cid].sec_id:
                    expanded.add(n)
        cand_ids = sorted(expanded, key=lambda c: -rrf.get(c, 0.0))

        # cross-encoder rerank
        if self.reranker is not None and cand_ids:
            pairs = [(query, self.chunks[c].embed_text()) for c in cand_ids]
            ce = self.reranker.predict(pairs)
            order = np.argsort(-np.asarray(ce))
            cand_ids = [cand_ids[i] for i in order]
            scores = {cand_ids[i]: float(ce[order[i]]) for i in range(len(cand_ids))}
        else:
            scores = {c: float(rrf.get(c, 0.0)) for c in cand_ids}

        # pack to a character budget instead of a fixed k
        kept, used = [], 0
        for cid in cand_ids:
            ln = len(self.chunks[cid].text)
            if used + ln > CONTEXT_CHAR_BUDGET:
                continue
            kept.append(cid)
            used += ln

        # ---- section completion (topic-independent completeness) ----------
        # If the hits already cover a good share of some section, return the
        # rest of that section too. A partially covered list is the failure
        # this whole build exists to prevent, and it can happen to any topic,
        # not just the ones someone thought to put on a keyword list.
        kept_set = set(kept)
        per_section: Dict[str, int] = {}
        for cid in kept:
            sid = self.chunks[cid].sec_id
            per_section[sid] = per_section.get(sid, 0) + 1
        completed = []
        for sid, n_hits in sorted(per_section.items(), key=lambda kv: -kv[1]):
            all_ids = self.doc.chunks_by_sec.get(sid, [])
            if not all_ids:
                continue
            coverage = n_hits / len(all_ids)
            if coverage < SECTION_COVERAGE_TRIGGER and n_hits < SECTION_HIT_TRIGGER:
                continue
            missing = [c for c in all_ids if c not in kept_set]
            cost = sum(len(self.chunks[c].text) for c in missing)
            if not missing or used + cost > CONTEXT_CHAR_BUDGET:
                continue
            for c in missing:
                kept_set.add(c)
                kept.append(c)
                scores.setdefault(c, 0.0)
            used += cost
            completed.append(f"{self.doc.sec_by_id[sid].full_title} "
                             f"(+{len(missing)} chunks)")

        chunks = [RetrievedChunk(self.chunks[c], scores.get(c, 0.0), self.doc.doc_id)
                  for c in kept]
        sec_ids = {c.chunk.sec_id for c in chunks}
        secs = [s for s in self.sections if s.sec_id in sec_ids]

        diag = {
            "mode": "hybrid",
            "sub_queries": len(sub_queries),
            "fused_candidates": len(ranked),
            "after_neighbours": len(expanded),
            "chunks_returned": len(chunks),
            "context_chars": used,
            "distinct_sections": len(sec_ids),
            "sections_completed": completed,
            "routing": routing,
            "subquery_agreement": _agreement(per_sub_sets),
            "reranked": self.reranker is not None,
        }
        return RetrievalResult(query, "hybrid", self.doc, chunks, secs,
                               sub_queries, diag)

    # ------------------------------------------------- deterministic items
    def section_items(self, sections: List[Section]) -> List[dict]:
        """
        Structural item list for the retrieved sections -- this is what makes
        'give me ALL of them' reliable. Items come from the document's own
        numbering, not from model recall.
        """
        out = []
        for sec in sections:
            text = self.doc.section_text(sec.sec_id)
            items = segment_enumerated_items(text, sec.char_start,
                                             x_lookup=self.doc.x_at)
            groups = group_items(items)
            if not groups:
                continue
            out.append({"section": sec, "groups": groups,
                        "n_items": sum(len(g["items"]) for g in groups)})
        return out


def _agreement(sets: List[set]) -> Optional[float]:
    comparable = [s for s in sets if s]
    if len(comparable) < 2:
        return None
    scores = []
    for i in range(len(comparable)):
        for j in range(i + 1, len(comparable)):
            a, b = comparable[i], comparable[j]
            if a | b:
                scores.append(len(a & b) / len(a | b))
    return round(sum(scores) / len(scores), 4) if scores else None


# =============================================================================
# 8. LLM ACCESS
# =============================================================================


_azure = None
_groq = None
_cache: Dict[str, Any] = {}


def _key(*parts: str) -> str:
    return hashlib.sha256("||".join(parts).encode()).hexdigest()


# =============================================================================
# Clients
# =============================================================================

def azure_client():
    global _azure
    if _azure is None:
        from openai import AzureOpenAI
        _azure = AzureOpenAI(
            api_version=AZURE_OPENAI_API_VERSION,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY,
        )
    return _azure


def groq_client():
    global _groq
    if _groq is None:
        from groq import Groq
        _groq = Groq(api_key=GROQ_API_KEY)
    return _groq


def llm_available() -> bool:
    return "ENTER KEY" not in (AZURE_OPENAI_API_KEY or "")


# =============================================================================
# Low-level calls
# =============================================================================

def complete(prompt: str, system: str = "You are a precise regulatory "
             "affairs analyst working with clinical trial protocols.",
             max_tokens: int = 8000) -> str:
    ck = _key("azure", prompt, system)
    if ck in _cache:
        return _cache[ck]
    resp = azure_client().chat.completions.create(
        model=AZURE_OPENAI_DEPLOYMENT,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": prompt}],
        max_completion_tokens=max_tokens,
    )
    out = (resp.choices[0].message.content or "").strip()
    _cache[ck] = out
    return out


def judge_complete(prompt: str, max_tokens: int = 600,
                   temperature: float = 0.0) -> str:
    ck = _key("groq", prompt, str(temperature))
    if ck in _cache:
        return _cache[ck]
    resp = groq_client().chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens, temperature=temperature,
    )
    out = (resp.choices[0].message.content or "").strip()
    _cache[ck] = out
    return out


def parse_json(raw: str) -> Any:
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:json)?|```$", "", cleaned, flags=re.MULTILINE).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        m = re.search(r"[\[{].*[\]}]", cleaned, re.DOTALL)
        if not m:
            raise
        return json.loads(m.group(0))


# =============================================================================
# Query expansion (judge model)
# =============================================================================

def _generic_variants(query: str) -> List[str]:
    """
    Topic-free query variants, used to widen recall when the LLM expander is
    unavailable. These are transformations of the question itself -- keyword
    form and heading form -- not a lookup table of subjects someone anticipated
    in advance.
    """
    terms = tokenize(query)
    variants = []
    if terms:
        variants.append(" ".join(terms))                       # keyword form
        variants.append(" ".join(terms[-4:]))                  # head of the question
        variants.append(" ".join(sorted(set(terms), key=terms.index)[:6]))
    return [v for v in variants if v and v.lower() != query.lower()]


def expand_query(query: str, n: int = 3) -> List[str]:
    """Sub-queries for retrieval. The LLM writes protocol-flavoured
    paraphrases when it is reachable; otherwise we fall back to mechanical
    variants of the question. Neither path assumes anything about the topic."""
    subs = [query]

    if llm_available():
        prompt = (
            "You expand search queries over a long clinical trial protocol.\n"
            f"Generate {n} alternative queries that improve recall by using "
            "protocol/regulatory synonyms and by targeting different sub-parts "
            "of the question, including the section heading it would live "
            "under. Do NOT answer it. Return ONLY a JSON array of strings.\n\n"
            f"Question: {query}")
        try:
            out = parse_json(judge_complete(prompt, max_tokens=250, temperature=0.3))
            if isinstance(out, list):
                subs += [s.strip() for s in out
                         if isinstance(s, str) and s.strip() and s.strip() not in subs]
        except Exception as e:
            print(f"    [expand] LLM expansion unavailable ({e})")

    for v in _generic_variants(query):
        if v not in subs:
            subs.append(v)
    return subs


# =============================================================================
# Groundedness judge
# =============================================================================

def judge_groundedness(query: str, answer: str, context: str) -> dict:
    if not llm_available():
        return {"confidence": None, "context_sufficient": None,
                "reasoning": "judge unavailable (no API key configured)"}
    prompt = f"""You are a strict regulatory QA grader. Decide whether the ANSWER is
fully supported by the CONTEXT (not whether it reads well).

CONTEXT:
{context[:40000]}

QUERY: {query}

ANSWER:
{answer[:12000]}

Break the answer into individual factual claims. Score "confidence" as the
proportion directly supported by the context (1.0 = all supported, 0.6 = a
meaningful minority unsupported, 0.2 = mostly unsupported). Separately judge
whether the CONTEXT was sufficient to answer the QUERY at all.

Return ONLY: {{"confidence": <float>, "context_sufficient": <true|false>, "reasoning": "<one sentence>"}}"""
    try:
        out = parse_json(judge_complete(prompt, max_tokens=300))
        cs = out.get("context_sufficient")
        return {"confidence": float(out.get("confidence", 0.5)),
                "context_sufficient": cs if isinstance(cs, bool) else None,
                "reasoning": str(out.get("reasoning", ""))}
    except Exception as e:
        return {"confidence": None, "context_sufficient": None,
                "reasoning": f"judge unavailable: {e}"}


# =============================================================================
# 8b. GRADING SUITE
# =============================================================================
# A QA grading layer for each answer, in the spirit of the earlier studio but
# adapted to this protocol pipeline. Deliberately:
#   * keeps the SINGLE groundedness judge already defined above
#     (judge_groundedness, Groq llama-3.3-70b) -- no multi-judge consensus panel
#   * uses NO DeepEval
#   * leans on deterministic, code-based signals (embeddings we already have,
#     the structural item list, the retrieval scores) so most graders cost
#     nothing extra and cannot themselves hallucinate.
#
# Two families:
#   ANSWER graders   -- is the prose answer grounded, complete and well-formed?
#   RETRIEVAL graders -- did retrieval surface the right, complete evidence, and
#                        were the extracted items well grounded? (the "items"
#                        section grading you asked for)
#
# Everything lands in one flat dict (grade_answer -> {...}) that the studio
# renders as a grading panel and every export format writes as columns.
# =============================================================================

# Below this cosine, the answer has drifted from the retrieved source text.
SEMANTIC_MATCH_FLOOR   = 0.45
# Retrieval scores below this are "weak" chunks (RRF/cross-encoder scale is
# small and unnormalised, so this is a relative, not absolute, floor -- see
# _retrieval_quality which rescales before applying it).
WEAK_CHUNK_QUANTILE    = 0.5
# A set-seeking answer delivered as one undivided wall of text is a structure
# smell; these bound "well-formed".
LONG_PARAGRAPH_SENTENCES = 6     # a paragraph longer than this is a "wall"
LONG_ANSWER_SENTENCES    = 8     # answers longer than this are expected to be
                                 # broken into paragraphs / grouped


def _embed_texts(embedder, texts: List[str]) -> Optional[np.ndarray]:
    """Encode with the retriever's own embedder if it is available. Returns
    None in lexical-only mode so callers can degrade gracefully."""
    if embedder is None or not texts:
        return None
    try:
        return embedder.encode(texts, convert_to_numpy=True,
                               normalize_embeddings=True).astype(np.float32)
    except Exception as e:
        print(f"    [grade] embedding failed ({e})")
        return None


# -----------------------------------------------------------------------------
# Grader A1 -- Semantic match (answer vs retrieved context), code-based
# -----------------------------------------------------------------------------
def grade_semantic_match(answer: str, context: str, embedder) -> Optional[float]:
    """Cosine similarity between the answer and the retrieved context. A cheap,
    LLM-free cross-check on the judge: a low value means the answer wandered
    away from the source text even if the judge liked it."""
    if not answer.strip() or not context.strip():
        return None
    emb = _embed_texts(embedder, [answer[:8000], context[:40000]])
    if emb is None or emb.shape[0] < 2:
        return None
    cos = float(np.dot(emb[0], emb[1]))
    return round(max(0.0, min(1.0, cos)), 4)


# -----------------------------------------------------------------------------
# Grader A2 -- Claim decomposition & verification (LLM, reuses the judge model)
# -----------------------------------------------------------------------------
def _decompose_claims(answer: str) -> List[str]:
    """Split the answer into atomic, independently-checkable factual claims.
    Uses the same judge model already configured -- no new model, no panel."""
    if not answer.strip() or not llm_available():
        return []
    prompt = (
        "Break the following answer into a flat list of atomic factual claims. "
        "Each claim: one verifiable statement, self-contained, minimally "
        "reworded from the answer, NOT a heading or meta-comment. "
        "Return ONLY a JSON array of strings.\n\nANSWER:\n" + answer[:12000])
    try:
        out = parse_json(judge_complete(prompt, max_tokens=1200, temperature=0.0))
        if isinstance(out, list):
            return [str(c).strip() for c in out if str(c).strip()][:60]
    except Exception as e:
        print(f"    [grade] claim decomposition failed ({e})")
    return []


def _verify_claims_batch(claims: List[str], context: str) -> List[dict]:
    """Verify every claim against the context in ONE judge call (keeps latency
    and token cost flat regardless of claim count). Each verdict is
    supported / unsupported / contradicted."""
    if not claims or not llm_available():
        return []
    numbered = "\n".join(f"{i+1}. {c}" for i, c in enumerate(claims))
    prompt = f"""You are a strict fact-checker for clinical-protocol answers.
For EACH numbered claim decide, using ONLY the context, whether it is:
  "supported"    -- the context states or clearly implies it
  "unsupported"  -- the context does not mention it
  "contradicted" -- the context says otherwise

CONTEXT:
{context[:40000]}

CLAIMS:
{numbered}

Return ONLY a JSON array, one object per claim, same order:
[{{"id": <n>, "verdict": "supported|unsupported|contradicted"}}]"""
    try:
        out = parse_json(judge_complete(prompt, max_tokens=1500, temperature=0.0))
        verdicts = {}
        if isinstance(out, list):
            for o in out:
                try:
                    verdicts[int(o.get("id"))] = str(o.get("verdict", "")).lower().strip()
                except Exception:
                    continue
        results = []
        for i, claim in enumerate(claims, 1):
            v = verdicts.get(i, "error")
            if v not in ("supported", "unsupported", "contradicted"):
                v = "error"
            results.append({"claim": claim, "verdict": v})
        return results
    except Exception as e:
        print(f"    [grade] claim verification failed ({e})")
        return [{"claim": c, "verdict": "error"} for c in claims]


def grade_claims(answer: str, context: str) -> dict:
    """Grader A2: decompose the answer and verify each claim against context.
    grounding_ratio = supported / total is the headline hallucination signal."""
    claims = _decompose_claims(answer)
    if not claims:
        return {"claim_grounding_ratio": None, "claims_total": 0,
                "claims_supported": 0, "claims_unsupported": 0,
                "claims_contradicted": 0, "unsupported_claims": [],
                "contradicted_claims": [], "claims_error": not answer.strip()}
    verified = _verify_claims_batch(claims, context)
    supp = sum(1 for r in verified if r["verdict"] == "supported")
    unsup = sum(1 for r in verified if r["verdict"] == "unsupported")
    contra = sum(1 for r in verified if r["verdict"] == "contradicted")
    total = len(verified)
    ratio = round(supp / total, 4) if total else None
    return {
        "claim_grounding_ratio": ratio,
        "claims_total": total,
        "claims_supported": supp,
        "claims_unsupported": unsup,
        "claims_contradicted": contra,
        "unsupported_claims": [r["claim"] for r in verified if r["verdict"] == "unsupported"],
        "contradicted_claims": [r["claim"] for r in verified if r["verdict"] == "contradicted"],
        "claims_error": False,
    }


# -----------------------------------------------------------------------------
# Grader A3 -- Coverage: are the extracted items reflected in the prose answer?
# -----------------------------------------------------------------------------
_COVERAGE_STOP = STOP | set("""subject subjects patient patients must should
prior study drug dose within least protocol amendment part cohort criteria""".split())


def _content_tokens(text: str) -> set:
    return {t for t in tokenize(text) if t not in _COVERAGE_STOP and len(t) > 3}


def grade_item_coverage(answer: str, items: List[dict]) -> dict:
    """Grader A3: fraction of extracted structured items whose key terms appear
    in the prose answer. Directly measures the failure you flagged -- a
    long answer that silently drops important enumerated points. An item is
    'covered' if >=2 of its content tokens (or its only one) appear in the
    answer."""
    if not items:
        return {"item_coverage_ratio": None, "items_covered": 0,
                "items_uncovered": 0, "uncovered_items": []}
    ans_tok = _content_tokens(answer)
    covered, uncovered = 0, []
    for it in items:
        it_tok = _content_tokens(it.get("text", ""))
        if not it_tok:
            continue
        need = 1 if len(it_tok) <= 2 else 2
        if len(it_tok & ans_tok) >= need:
            covered += 1
        else:
            uncovered.append(it.get("text", "")[:140])
    checked = covered + len(uncovered)
    ratio = round(covered / checked, 4) if checked else None
    return {"item_coverage_ratio": ratio, "items_covered": covered,
            "items_uncovered": len(uncovered), "uncovered_items": uncovered[:25]}


# -----------------------------------------------------------------------------
# Grader A4 -- Answer structure quality (deterministic)
# -----------------------------------------------------------------------------
_SENT_SPLIT = re.compile(r"(?<=[.;:])\s+(?=[A-Z0-9(])")


def _sentences(text: str) -> List[str]:
    return [s for s in _SENT_SPLIT.split(text.strip()) if s.strip()]


def grade_structure(answer: str, question: str, wants_set: bool,
                    n_items: int) -> dict:
    """Grader A4: is the answer well-formed, or a run-on wall of text? Pure
    string analysis -- flags exactly the badly-structured, single-paragraph
    answer you attached. A set-seeking question answered as one giant paragraph
    with many items scores low on structure."""
    ans = (answer or "").strip()
    if not ans:
        return {"structure_score": None, "structure_flags": [],
                "n_paragraphs": 0, "n_sentences": 0,
                "longest_paragraph_sentences": 0}
    paras = [p for p in re.split(r"\n\s*\n", ans) if p.strip()]
    sents = _sentences(ans)
    para_sent_counts = [len(_sentences(p)) for p in paras] or [len(sents)]
    longest = max(para_sent_counts)
    has_bullets = bool(re.search(r"(?m)^\s*(?:[-\u2022*]|\d+[.)]|[a-z][.)])\s+", ans))
    has_headers = bool(re.search(r"(?m)^\s*[A-Z][^\n]{0,60}:\s*$", ans)) or \
                  bool(re.search(r"\*\*[^*]+\*\*", ans))

    flags = []
    score = 1.0
    # A long, set-seeking answer delivered as one paragraph is the core smell.
    if len(sents) >= LONG_ANSWER_SENTENCES and len(paras) <= 1 and not has_bullets:
        flags.append("wall_of_text")
        score -= 0.45
    if longest > LONG_PARAGRAPH_SENTENCES and not has_bullets:
        flags.append("overlong_paragraph")
        score -= 0.2
    if wants_set and n_items >= MIN_LIST_ITEMS and not (has_bullets or has_headers
                                                        or len(paras) >= 3):
        flags.append("set_question_not_grouped")
        score -= 0.25
    if (wants_set or n_items >= LONG_ANSWER_SENTENCES) and not (has_bullets or has_headers):
        flags.append("no_visual_structure")
        score -= 0.1
    score = round(max(0.0, min(1.0, score)), 3)
    return {"structure_score": score, "structure_flags": flags,
            "n_paragraphs": len(paras), "n_sentences": len(sents),
            "longest_paragraph_sentences": longest,
            "has_bullets": has_bullets, "has_headers": has_headers}


# -----------------------------------------------------------------------------
# Grader R1 -- Retrieval score distribution (code-based)
# -----------------------------------------------------------------------------
def _retrieval_quality(result: "RetrievalResult") -> dict:
    """Grader R1: distribution of retrieval scores across the returned chunks.
    A wide spread with many weak chunks means retrieval was patchy and the
    answer may rest on low-signal passages. Scores here are RRF / cross-encoder
    values on an arbitrary scale, so they are min-max rescaled to [0,1] before
    a relative weak-chunk floor is applied. In section mode every chunk is
    forced in at score 1.0, so the distribution is reported as uniform."""
    scores = [r.score for r in result.chunks]
    if not scores:
        return {"retrieval_n": 0, "retrieval_score_max": None,
                "retrieval_score_min": None, "retrieval_score_mean": None,
                "retrieval_score_spread": None, "weak_chunks_ratio": None}
    s_max, s_min = max(scores), min(scores)
    mean = sum(scores) / len(scores)
    spread = s_max - s_min
    if spread > 1e-9:
        rescaled = [(s - s_min) / spread for s in scores]
        weak = sum(1 for r in rescaled if r < WEAK_CHUNK_QUANTILE)
    else:
        weak = 0                      # all equal (e.g. section mode) -> none weak
    return {
        "retrieval_n": len(scores),
        "retrieval_score_max": round(float(s_max), 4),
        "retrieval_score_min": round(float(s_min), 4),
        "retrieval_score_mean": round(float(mean), 4),
        "retrieval_score_spread": round(float(spread), 4),
        "weak_chunks_ratio": round(weak / len(scores), 4),
    }


# -----------------------------------------------------------------------------
# Grader R2 -- Section completeness (did we capture whole enumerated lists?)
# -----------------------------------------------------------------------------
def _section_completeness(retriever: "ProtocolRetriever",
                          result: "RetrievalResult") -> dict:
    """Grader R2: for every section the answer drew on, what fraction of its
    chunks were retrieved, and -- for list-shaped sections -- were all the
    enumerated items captured? A partially covered criteria list is THE failure
    this pipeline exists to prevent, so surfacing it as a grade is the point."""
    per_sec: Dict[str, int] = {}
    for r in result.chunks:
        per_sec[r.chunk.sec_id] = per_sec.get(r.chunk.sec_id, 0) + 1
    if not per_sec:
        return {"section_coverage_ratio": None, "list_sections_total": 0,
                "list_sections_complete": 0, "partial_list_sections": []}

    coverages, list_total, list_complete, partial = [], 0, 0, []
    for sid, n_hit in per_sec.items():
        all_ids = result.doc.chunks_by_sec.get(sid, [])
        if not all_ids:
            continue
        cov = n_hit / len(all_ids)
        coverages.append(cov)
        if retriever.section_item_count(sid) >= MIN_LIST_ITEMS:
            list_total += 1
            if n_hit >= len(all_ids):
                list_complete += 1
            else:
                partial.append({
                    "section": result.doc.sec_by_id[sid].full_title,
                    "chunks": f"{n_hit}/{len(all_ids)}",
                    "coverage": round(cov, 3)})
    return {
        "section_coverage_ratio": round(sum(coverages) / len(coverages), 4) if coverages else None,
        "list_sections_total": list_total,
        "list_sections_complete": list_complete,
        "partial_list_sections": partial[:10],
    }


# -----------------------------------------------------------------------------
# Grader R3 -- Item grounding distribution (the "items" grading you asked for)
# -----------------------------------------------------------------------------
def _item_grounding(items: List[dict]) -> dict:
    """Grader R3: how well the extracted items align to verbatim source spans.
    Reuses the grounded/partial/ungrounded labels already computed per item and
    turns them into a graded ratio + a mean alignment score."""
    if not items:
        return {"item_grounding_ratio": None, "item_mean_score": None,
                "items_grounded": 0, "items_partial": 0, "items_ungrounded": 0}
    g = sum(1 for i in items if i.get("grounding") == "grounded")
    p = sum(1 for i in items if i.get("grounding") == "partial")
    u = sum(1 for i in items if i.get("grounding") == "ungrounded")
    scored = [i.get("score", 0.0) for i in items if i.get("score") is not None]
    return {
        "item_grounding_ratio": round(g / len(items), 4),
        "item_mean_score": round(sum(scored) / len(scored), 2) if scored else None,
        "items_grounded": g, "items_partial": p, "items_ungrounded": u,
    }


# -----------------------------------------------------------------------------
# Grader R4 -- Sub-query agreement + context utilisation
# -----------------------------------------------------------------------------
def _retrieval_agreement(result: "RetrievalResult", answer: str,
                         embedder) -> dict:
    """Grader R4: two cheap retrieval-health signals.
      * subquery_agreement -- mean Jaccard overlap between the hit sets of the
        expanded sub-queries (already computed in diagnostics for hybrid mode);
        low overlap means the query was ambiguous / recall was unstable.
      * context_utilisation -- of the retrieved chunks, what fraction are
        semantically close to the answer. High retrieval volume with low
        utilisation flags padded, noisy context."""
    agreement = (result.diagnostics or {}).get("subquery_agreement")
    utilisation = None
    if answer.strip() and embedder is not None and result.chunks:
        chunk_texts = [r.chunk.text for r in result.chunks][:60]
        emb = _embed_texts(embedder, [answer[:8000]] + chunk_texts)
        if emb is not None and emb.shape[0] > 1:
            sims = emb[1:] @ emb[0]
            utilisation = round(float(np.mean(sims >= 0.30)), 4)
    return {"subquery_agreement": agreement,
            "context_utilisation": utilisation}


# -----------------------------------------------------------------------------
# Orchestrator: run the whole suite and fold into one flat dict
# -----------------------------------------------------------------------------
def _overall_grade(g: dict) -> Optional[float]:
    """A single 0-1 headline grade: mean of the signals that are present.
    Weighted toward grounding (judge + claims + items) because faithfulness
    matters more than polish for a regulatory answer."""
    parts = []
    def add(v, w):
        if v is not None:
            parts.append((float(v), w))
    add(g.get("judge_confidence"), 2.0)
    add(g.get("claim_grounding_ratio"), 2.0)
    add(g.get("item_grounding_ratio"), 1.5)
    add(g.get("semantic_match"), 1.0)
    add(g.get("item_coverage_ratio"), 1.0)
    add(g.get("structure_score"), 0.8)
    add(g.get("section_coverage_ratio"), 0.7)
    if not parts:
        return None
    num = sum(v * w for v, w in parts)
    den = sum(w for _, w in parts)
    return round(num / den, 4)


def grade_answer(retriever: "ProtocolRetriever", result: "RetrievalResult",
                 question: str, answer: str, context: str, items: List[dict],
                 judge: dict, wants_set: bool = False,
                 run_llm_graders: bool = True) -> dict:
    """
    Full grading suite for one answered question. Returns a flat dict that the
    studio renders and the exporter writes. `judge` is the result of the
    existing single groundedness judge (kept, not replaced).

    LLM cost: at most TWO extra judge-model calls (claim decompose + verify),
    both gated by run_llm_graders. Every other grader is deterministic.
    """
    embedder = getattr(retriever, "embedder", None)
    grades: Dict[str, Any] = {}

    # ---- carry the existing single judge through as Grader G0 --------------
    grades["judge_confidence"]  = judge.get("confidence")
    grades["context_sufficient"] = judge.get("context_sufficient")
    grades["judge_reasoning"]   = judge.get("reasoning")

    # ---- ANSWER graders ----------------------------------------------------
    grades["semantic_match"] = grade_semantic_match(answer, context, embedder)
    if run_llm_graders:
        grades.update(grade_claims(answer, context))
    else:
        grades.update({"claim_grounding_ratio": None, "claims_total": 0,
                       "claims_supported": 0, "claims_unsupported": 0,
                       "claims_contradicted": 0, "unsupported_claims": [],
                       "contradicted_claims": [], "claims_error": False})
    grades.update(grade_item_coverage(answer, items))
    grades.update(grade_structure(answer, question, wants_set, len(items)))

    # ---- RETRIEVAL / ITEMS graders -----------------------------------------
    grades.update(_retrieval_quality(result))
    grades.update(_section_completeness(retriever, result))
    grades.update(_item_grounding(items))
    grades.update(_retrieval_agreement(result, answer, embedder))

    # ---- headline + review flags -------------------------------------------
    grades["overall_grade"] = _overall_grade(grades)

    flags = []
    if grades.get("semantic_match") is not None and grades["semantic_match"] < SEMANTIC_MATCH_FLOOR:
        flags.append("low_semantic_match")
    if grades.get("claims_contradicted", 0) > 0:
        flags.append("contradicted_claims")
    if grades.get("claim_grounding_ratio") is not None and grades["claim_grounding_ratio"] < 0.7:
        flags.append("weak_claim_grounding")
    if grades.get("item_coverage_ratio") is not None and grades["item_coverage_ratio"] < 0.6:
        flags.append("items_missing_from_answer")
    if grades.get("partial_list_sections"):
        flags.append("incomplete_list_section")
    if grades.get("structure_flags"):
        flags.append("weak_structure")
    if judge.get("context_sufficient") is False:
        flags.append("context_insufficient")
    grades["review_flags"] = flags
    # Lower = needs more scrutiny (mirrors the earlier studio's review_priority).
    base = grades.get("overall_grade")
    grades["review_priority"] = (round(max(0.0, base - 0.1 * len(flags)), 4)
                                 if base is not None else None)
    return grades


# =============================================================================
# 9. EXTRACTION + ORCHESTRATION
# =============================================================================


from rapidfuzz import fuzz


# =============================================================================
# Evidence -> source span alignment
# =============================================================================

def align_evidence(evidence: str, haystack: str,
                   window_hint: Optional[int] = None) -> dict:
    """
    Locate `evidence` inside `haystack` and return {start, end, score}.
    Exact match first, then a sliding fuzzy alignment. Score is 0-100.
    """
    ev = re.sub(r"\s+", " ", (evidence or "").strip())
    if not ev or not haystack:
        return {"start": None, "end": None, "score": 0.0}

    # ---- exact (whitespace-insensitive) ---------------------------------
    flat, index_map = _flatten(haystack)
    pos = flat.find(ev)
    if pos != -1:
        return {"start": index_map[pos],
                "end": index_map[min(pos + len(ev), len(index_map) - 1)],
                "score": 100.0}

    # ---- fuzzy ------------------------------------------------------------
    result = fuzz.partial_ratio_alignment(ev, flat, score_cutoff=PARTIAL_THRESHOLD)
    if result is None:
        return {"start": None, "end": None, "score": 0.0}
    s, e = result.dest_start, result.dest_end
    return {"start": index_map[min(s, len(index_map) - 1)],
            "end": index_map[min(e, len(index_map) - 1)],
            "score": round(float(result.score), 1)}


def _flatten(text: str):
    """Collapse whitespace but keep a map back to original offsets."""
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


def grounding_label(score: float) -> str:
    if score >= GROUNDED_THRESHOLD:
        return "grounded"
    if score >= PARTIAL_THRESHOLD:
        return "partial"
    return "ungrounded"


# =============================================================================
# Prompts
# =============================================================================

NARRATIVE_PROMPT = """## ROLE
You are a senior clinical development and regulatory affairs specialist reading a
clinical study protocol. Your output is reviewed by an SME.

## TASK
Answer the question using ONLY the protocol text supplied below.

## RULES
- Use only what is in the supplied text. No outside knowledge, no inferred
  rationale, no invented requirements. If the text states a rule but not the
  reason for it, give the rule only.
- Preserve exact values, thresholds, units, timeframes and section references.
- Do NOT drop items. If the protocol lists many criteria, every distinct one
  must appear -- completeness matters more than brevity.
- If the supplied text does not answer the question, say exactly:
  "The provided context does not contain enough information to fully answer this query."
- Length follows the question and the evidence, not a target.

## STRUCTURE  (this is graded -- follow it)
Do NOT return one long run-on paragraph. Structure the answer so an SME can scan it:
1. Open with ONE short lead sentence that directly answers the question.
2. Then organise the detail into GROUPED sections. Group by the protocol's own
   structure -- e.g. by cohort/part, or by category (disease/staging,
   biomarker & tissue, measurable disease, prior-therapy timing, organ
   function, demographics & contraception, etc.). Each group gets:
      - a short **bold heading** on its own line, then
      - a bullet list ("- ") of the items in that group, one criterion per bullet,
        keeping exact thresholds/units/timeframes.
3. Keep any part- or cohort-specific requirements under their own clearly
   labelled group so they are not blurred into the general criteria.
4. Put a blank line between groups.
A flat wall of text, or a set-type answer with no headings/bullets, is a
FAILED structure even if the facts are correct.

## PROTOCOL TEXT
{context}

## QUESTION
{question}

## OUTPUT
Return ONLY a JSON object:
{{
  "answer": "<the structured answer: lead sentence, then **bold group headings** each followed by '- ' bullets, blank line between groups. Use \\n for line breaks and \\n\\n between groups.>",
  "key_points": ["<short bullet capturing one distinct requirement>", "..."],
  "confidence": <float 0.0-1.0>
}}"""

STRUCTURED_PROMPT = """## ROLE
You are a clinical protocol information-extraction engine.

## TASK
Answer the question AND extract the discrete facts that support it, from the
protocol text below.

## RULES
- Every extracted item MUST include "evidence": a VERBATIM span copied
  character-for-character from the protocol text. Never paraphrase inside
  "evidence". If you cannot quote it verbatim, do not extract the item.
- "text" is your normalised one-line rendering of the item; "evidence" is the
  raw protocol wording it came from.
- Group items under short headings that reflect the protocol's own structure.
- Do not invent items, thresholds, or requirements.

## PROTOCOL TEXT
{context}

## QUESTION
{question}

## OUTPUT
Return ONLY a JSON object:
{{
  "title": "<short title for this extraction>",
  "answer": "<direct prose answer>",
  "groups": [
    {{"heading": "<group heading>",
      "items": [{{"text": "<normalised item>", "evidence": "<verbatim span>",
                  "attributes": {{"<key>": "<value>"}}}}]}}
  ],
  "confidence": <float 0.0-1.0>
}}"""

LABEL_PROMPT = """You are labelling items already extracted from a clinical protocol.
For each numbered item below, return a short category label and mark whether it is
relevant to the question.

QUESTION: {question}

ITEMS:
{items}

Return ONLY a JSON array, one object per item, in the same order:
[{{"id": <id>, "category": "<2-4 word category>", "relevant": <true|false>}}]"""


# =============================================================================
# Map-reduce narrative answer
# =============================================================================

def _split_for_map(context: str, budget: int = 14000) -> List[str]:
    if len(context) <= budget:
        return [context]
    parts, buf = [], []
    size = 0
    for para in context.split("\n"):
        if size + len(para) > budget and buf:
            parts.append("\n".join(buf))
            buf, size = [], 0
        buf.append(para)
        size += len(para) + 1
    if buf:
        parts.append("\n".join(buf))
    return parts


def narrative_answer(question: str, context: str) -> dict:
    """Answer over arbitrarily long context via map-reduce (never truncates)."""
    if not llm_available():
        return {"answer": "", "key_points": [], "confidence": None,
                "note": "LLM not configured -- showing structural extraction only"}

    parts = _split_for_map(context)
    try:
        if len(parts) == 1:
            out = parse_json(complete(
                NARRATIVE_PROMPT.format(context=parts[0], question=question)))
            return {"answer": out.get("answer", ""),
                    "key_points": out.get("key_points", []),
                    "confidence": out.get("confidence"),
                    "passes": 1}

        partials = []
        for i, part in enumerate(parts, 1):
            out = parse_json(complete(
                NARRATIVE_PROMPT.format(context=part, question=question)))
            if out.get("answer"):
                partials.append(f"[Segment {i}]\n{out['answer']}\n" +
                                "\n".join(f"- {p}" for p in out.get("key_points", [])))
        reduce_prompt = (
            "Merge the segment answers below into ONE answer to the question. "
            "Keep EVERY distinct fact and criterion; remove only exact duplicates; "
            "do not add anything new. "
            "Structure the merged answer for an SME to scan: one short lead "
            "sentence, then grouped **bold headings** (by cohort/part or by "
            "category) each followed by '- ' bullets, one criterion per bullet, "
            "exact thresholds/units/timeframes preserved, a blank line between "
            "groups. Do NOT return a single run-on paragraph. "
            "Return ONLY {\"answer\": \"...\", \"key_points\": [...], \"confidence\": <float>}.\n\n"
            f"QUESTION: {question}\n\n" + "\n\n".join(partials))
        out = parse_json(complete(reduce_prompt))
        return {"answer": out.get("answer", ""),
                "key_points": out.get("key_points", []),
                "confidence": out.get("confidence"),
                "passes": len(parts)}
    except Exception as e:
        return {"answer": "", "key_points": [], "confidence": None,
                "note": f"LLM answer unavailable: {e}"}


# =============================================================================
# Path A -- structural extraction (section mode)
# =============================================================================

def structural_extraction(retriever: ProtocolRetriever, result: RetrievalResult,
                          question: str, view_text: str, mapping: List[dict]) -> dict:
    blocks = retriever.section_items(result.sections)
    groups_out, n = [], 0

    for blk in blocks:
        sec = blk["section"]
        for g in blk["groups"]:
            items = []
            for it in g["items"]:
                span = _doc_span_to_view(it["char_start"], it["char_end"], mapping)
                items.append({
                    "id": f"i{n:04d}",
                    "text": it["text"],
                    "evidence": it["text"],
                    "marker": it.get("marker", ""),
                    "depth": it.get("depth", 0),
                    "page": result.doc.page_of_offset(it["char_start"]) + 1,
                    "section": sec.full_title,
                    "attributes": {"marker": it.get("marker", ""),
                                   "depth": it.get("depth", 0)},
                    "span": span,
                    # absolute offsets into the full document text, so the item
                    # can be highlighted in the whole PDF, not just the excerpt
                    "doc_span": {"start": it["char_start"], "end": it["char_end"]},
                    "score": 100.0 if span else 0.0,
                    "grounding": "grounded" if span else "ungrounded",
                })
                n += 1
            if items:
                groups_out.append({"heading": g["heading"] or sec.full_title,
                                   "section": sec.full_title, "items": items})

    if not groups_out:                       # section has prose, not a list
        # Pass the view->doc offset mapping through: without it every
        # generatively-extracted item comes back with doc_span=None and the
        # source pane has nothing to highlight (this is the whole difference
        # between a marker-less list, e.g. bullets the PDF renders without a
        # glyph, and one with "1)/a)/i." markers that the structural path finds).
        return generative_extraction(question, view_text, view_text, mapping)

    return {"title": " / ".join(s.full_title for s in result.sections[:2]),
            "groups": groups_out, "answer": "", "confidence": None,
            "method": "structural"}


def _doc_span_to_view(doc_start: int, doc_end: int, mapping: List[dict]) -> Optional[dict]:
    for m in mapping:
        if m["doc_start"] <= doc_start < m["doc_end"]:
            offset = m["view_start"] + (doc_start - m["doc_start"])
            length = min(doc_end, m["doc_end"]) - doc_start
            return {"start": offset, "end": offset + max(1, length)}
    return None


def _view_span_to_doc(view_start: int, view_end: int, mapping: List[dict]) -> Optional[dict]:
    """Inverse of _doc_span_to_view: turn an offset into the retrieved excerpt
    back into an absolute offset in the full document text, so a generatively
    extracted item can be highlighted in the whole PDF."""
    for m in mapping:
        view_seg_start = m["view_start"]
        view_seg_end = m["view_start"] + (m["doc_end"] - m["doc_start"])
        if view_seg_start <= view_start < view_seg_end:
            doc_start = m["doc_start"] + (view_start - m["view_start"])
            doc_end = m["doc_start"] + (view_end - m["view_start"])
            return {"start": doc_start, "end": doc_end}
    return None


# =============================================================================
# Path B -- generative extraction (hybrid mode)
# =============================================================================

def generative_extraction(question: str, context: str, view_text: str,
                          mapping: Optional[List[dict]] = None) -> dict:
    if not llm_available():
        return {"title": "Retrieved passages", "groups": [], "answer": "",
                "confidence": None, "method": "none",
                "note": "LLM not configured -- structured extraction requires the Azure model"}
    try:
        out = parse_json(complete(
            STRUCTURED_PROMPT.format(context=context[:50000], question=question)))
    except Exception as e:
        return {"title": "Retrieved passages", "groups": [], "answer": "",
                "confidence": None, "method": "none",
                "note": f"structured extraction unavailable: {e}"}

    groups, n = [], 0
    for g in out.get("groups", []):
        items = []
        for it in g.get("items", []):
            ev = it.get("evidence", "") or it.get("text", "")
            al = align_evidence(ev, view_text)
            span = ({"start": al["start"], "end": al["end"]}
                    if al["start"] is not None else None)
            doc_span = None
            if span and mapping is not None:
                doc_span = _view_span_to_doc(span["start"], span["end"], mapping)
            items.append({
                "id": f"i{n:04d}",
                "text": it.get("text", ev)[:600],
                "evidence": ev,
                "marker": "", "depth": 0,
                "attributes": it.get("attributes", {}) or {},
                "span": span,
                "doc_span": doc_span,
                "score": al["score"],
                "grounding": grounding_label(al["score"]),
                "page": None, "section": g.get("heading", ""),
            })
            n += 1
        if items:
            groups.append({"heading": g.get("heading", "Findings"),
                           "section": g.get("heading", ""), "items": items})

    return {"title": out.get("title", "Structured extraction"),
            "groups": groups, "answer": out.get("answer", ""),
            "confidence": out.get("confidence"), "method": "generative"}


# =============================================================================
# Orchestration
# =============================================================================

def _fully_covered_list_sections(retriever: "ProtocolRetriever",
                                 result: "RetrievalResult") -> List[Section]:
    """Sections whose every chunk was retrieved and which contain a real
    enumerated list. Topic-independent -- it asks what the retrieved evidence
    looks like, not what the question was about."""
    per_section: Dict[str, int] = {}
    for r in result.chunks:
        per_section[r.chunk.sec_id] = per_section.get(r.chunk.sec_id, 0) + 1
    out = []
    for sid, n in per_section.items():
        all_ids = result.doc.chunks_by_sec.get(sid, [])
        if all_ids and n >= len(all_ids) and \
                retriever.section_item_count(sid) >= MIN_LIST_ITEMS:
            out.append(result.doc.sec_by_id[sid])
    return sorted(out, key=lambda s: s.char_start)


def answer_question(retriever: ProtocolRetriever, question: str,
                    force_mode: Optional[str] = None,
                    run_judge: bool = True) -> dict:
    subs = expand_query(question)
    result = retriever.retrieve(question, subs, force_mode=force_mode)
    view_text, mapping = result.source_view()

    if result.mode == "section":
        extraction = structural_extraction(retriever, result, question,
                                           view_text, mapping)
        narrative = narrative_answer(question, view_text)
        extraction["answer"] = narrative.get("answer") or extraction.get("answer", "")
        key_points = narrative.get("key_points", [])
        note = narrative.get("note")
        confidence = narrative.get("confidence")
    else:
        # Even in passage mode, any section that ended up fully retrieved (via
        # section completion) and that is genuinely a list gets the
        # deterministic treatment -- structural items cannot drop an entry, and
        # this applies to whatever the question happened to be about.
        covered = _fully_covered_list_sections(retriever, result)
        if covered:
            sub_result = RetrievalResult(question, "section", result.doc,
                                         result.chunks, covered,
                                         result.sub_queries, result.diagnostics)
            extraction = structural_extraction(retriever, sub_result, question,
                                               view_text, mapping)
            extraction["method"] = "structural (completed sections)"
        else:
            extraction = generative_extraction(question, view_text, view_text,
                                                mapping)
        narrative = {"key_points": []}
        key_points = []
        note = extraction.get("note")
        confidence = extraction.get("confidence")
        if not extraction.get("answer"):
            narrative = narrative_answer(question, view_text)
            extraction["answer"] = narrative.get("answer", "")
            key_points = narrative.get("key_points", [])
            note = note or narrative.get("note")

    items = [it for g in extraction.get("groups", []) for it in g["items"]]
    stats = {
        "extracted": len(items),
        "grounded": sum(1 for i in items if i["grounding"] == "grounded"),
        "partial": sum(1 for i in items if i["grounding"] == "partial"),
        "ungrounded": sum(1 for i in items if i["grounding"] == "ungrounded"),
    }

    judge = {"confidence": None, "context_sufficient": None, "reasoning": ""}
    if run_judge and extraction.get("answer"):
        judge = judge_groundedness(question, extraction["answer"], view_text)

    # ---- grading suite -----------------------------------------------------
    # Deterministic graders always run; the two LLM graders (claim decompose +
    # verify) reuse the SINGLE judge model and are gated on run_judge so a user
    # who turned the judge off pays no extra LLM cost.
    wants_set = ProtocolRetriever.enumeration_cues(question).get("wants_set", False)
    grades = grade_answer(
        retriever, result, question,
        answer=extraction.get("answer", ""),
        context=view_text, items=items, judge=judge,
        wants_set=wants_set, run_llm_graders=run_judge)

    return {
        "question": question,
        "mode": result.mode,
        "document": {"id": result.doc.doc_id, "title": result.doc.title,
                     "version": result.doc.version_label,
                     "pages": result.doc.n_pages},
        "sections": [{"title": s.full_title,
                      "page_start": s.page_start + 1,
                      "page_end": s.page_end + 1} for s in result.sections],
        "source_text": view_text,
        "extraction": extraction,
        "answer": extraction.get("answer", ""),
        "key_points": key_points,
        "confidence": confidence,
        "judge": judge,
        "grades": grades,
        "wants_set": wants_set,
        "stats": stats,
        "diagnostics": result.diagnostics,
        "sub_queries": subs,
        "note": note,
    }


# =============================================================================
# 10. CLI
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description="Protocol Transformation Studio engine")
    ap.add_argument("pdf", nargs="?", default=PDF_PATHS[0])
    ap.add_argument("question", nargs="?", help="ask a question and print the extraction")
    ap.add_argument("--no-embeddings", action="store_true",
                    help="lexical-only (skips the sentence-transformers download)")
    ap.add_argument("--rebuild", action="store_true", help="ignore the parse cache")
    args = ap.parse_args()

    doc = load_protocol(args.pdf, use_cache=not args.rebuild)
    print(f"\n{doc.doc_id}: {doc.n_pages} pages | {len(doc.sections)} sections | "
          f"{len(doc.chunks)} chunks | {len(doc.text):,} chars")

    if not args.question:
        # index diagnostics: how well did the document decompose?
        lens = [len(c.text) for c in doc.chunks]
        print(f"chunk chars: mean {int(np.mean(lens))} | p95 "
              f"{int(np.percentile(lens, 95))} | max {max(lens)}")
        print("\nsections containing enumerated criteria:")
        for sec in doc.sections:
            items = segment_enumerated_items(doc.section_text(sec.sec_id), sec.char_start,
                                             x_lookup=doc.x_at)
            groups = group_items(items)
            n = sum(len(g["items"]) for g in groups)
            if n >= 5:
                print(f"  {sec.full_title[:70]:<72} {n:>4} items  "
                      f"(pp. {sec.page_start + 1}-{sec.page_end + 1})")
        return

    retriever = ProtocolRetriever([doc], use_embeddings=not args.no_embeddings)
    payload = answer_question(retriever, args.question)
    print(f"\nmode={payload['mode']}  stats={payload['stats']}")
    for s in payload["sections"][:5]:
        print(f"  section: {s['title']} (pp. {s['page_start']}-{s['page_end']})")
    if payload["answer"]:
        print("\n" + payload["answer"])
    print()
    for g in payload["extraction"]["groups"]:
        print(f"  # {g['heading']}")
        for it in g["items"]:
            flag = {"grounded": "+", "partial": "~", "ungrounded": "!"}[it["grounding"]]
            print(f"    {flag} {it['text'][:110]}")


if __name__ == "__main__":
    main()