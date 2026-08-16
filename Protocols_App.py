from __future__ import annotations

import io
import os
import csv
import json
import argparse
import datetime
import tempfile
import traceback

from flask import Flask, request, jsonify, Response

import Protocols_Pipeline as P
from Protocols_Pipeline import ProtocolRetriever, answer_question

app = Flask(__name__)

# -----------------------------------------------------------------------------
# Multi-document registry
# -----------------------------------------------------------------------------

RETRIEVERS: dict[str, ProtocolRetriever] = {}   # doc_id -> retriever
DOC_ORDER: list[str] = []                        # preserves upload/load order
SESSION_LOG: list[dict] = []
_ENTRY_SEQ = 0

USE_EMBEDDINGS = True
UPLOAD_DIR = os.path.join(tempfile.gettempdir(), "protocol_uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


def _register_docs_from_paths(paths: list[str]) -> list[str]:
    """Load one or more PDFs through pipeline_2's own loaders and build a
    retriever for each. Returns the list of newly-registered doc_ids."""
    new_ids: list[str] = []
    docs = P.load_all(paths)
    for d in docs:
        # A ProtocolRetriever built from a single already-loaded doc. This runs
        # exactly the same construction path pipeline_2 uses at startup.
        retr = ProtocolRetriever(docs=[d], use_embeddings=USE_EMBEDDINGS)
        RETRIEVERS[d.doc_id] = retr
        if d.doc_id not in DOC_ORDER:
            DOC_ORDER.append(d.doc_id)
        new_ids.append(d.doc_id)
    return new_ids


def get_retriever(doc_id: str | None = None) -> ProtocolRetriever:
    """Return the retriever for a given doc_id, or the first registered one."""
    if doc_id and doc_id in RETRIEVERS:
        return RETRIEVERS[doc_id]
    if DOC_ORDER:
        return RETRIEVERS[DOC_ORDER[0]]
    # Lazy default load (mirrors the old single-doc behaviour).
    _register_docs_from_paths(P.PDF_PATHS)
    if not DOC_ORDER:
        raise RuntimeError("No protocol documents loaded -- check PDF_PATHS")
    return RETRIEVERS[DOC_ORDER[0]]


def _doc_meta(doc_id: str) -> dict:
    r = RETRIEVERS[doc_id]
    d = r.doc
    return {
        "doc_id": d.doc_id,
        "title": d.title or d.doc_id,
        "version": d.version_label,
        "pages": d.n_pages,
        "sections": len(d.sections),
        "chunks": len(d.chunks),
        "chars": len(d.text),
        "dense": r.index is not None,
        "reranker": r.reranker is not None,
    }


# =============================================================================
# Interface
# =============================================================================

STUDIO_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Protocol Transformation Studio</title>
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
  .eyebrow{font-size:12px;letter-spacing:.12em;font-weight:700;color:var(--accent);text-transform:uppercase}
  h1{font-size:27px;margin:0 0 4px;font-weight:650;letter-spacing:-.01em;color:var(--ink)}
  .sub{color:var(--muted);margin:0 0 18px}
  .topline{display:flex;justify-content:space-between;align-items:center;gap:24px;flex-wrap:wrap;margin-bottom:6px}
  .topctrl{display:flex;align-items:center;gap:14px;flex-wrap:wrap}
  .docchip{display:flex;align-items:center;gap:8px;border:1px solid var(--line);border-radius:999px;
           padding:7px 14px;background:var(--wash);font-size:12.5px;color:#334155}
  .dot{width:8px;height:8px;border-radius:50%;background:var(--ok)}
  .dot.off{background:#cbd5e1}

  /* ---- ICON plc brand bar ---- */
  .brandbar{background:var(--accent);color:#fff}
  .brandbar-inner{max-width:1560px;margin:0 auto;padding:11px 28px;display:flex;align-items:center;gap:14px}
  .brandmark{font-size:21px;font-weight:800;letter-spacing:.09em;line-height:1}
  .brandmark sup{font-size:11px;font-weight:600;letter-spacing:.14em;opacity:.85;margin-left:3px}
  .brandbar-div{width:1px;height:20px;background:rgba(255,255,255,.4)}
  .brandbar-tag{font-size:12.5px;letter-spacing:.02em;opacity:.95;font-weight:500}

  /* ---- stats ---- */
  .stats{display:grid;grid-template-columns:repeat(5,1fr);border:1px solid var(--line);
         border-radius:10px;overflow:hidden;margin:14px 0 20px;border-top:3px solid var(--accent)}
  .stat{padding:16px 18px;display:flex;flex-direction:column;align-items:center;gap:6px;
        border-right:1px solid var(--line);background:var(--panel);text-align:center}
  .stat:last-child{border-right:0}
  .stat b{font-size:24px;font-weight:650;letter-spacing:-.02em;color:var(--ink)}
  .stat span{color:var(--muted);font-size:12.5px}
  .stat.g b{color:var(--ok)} .stat.p b{color:var(--warn)} .stat.u b{color:var(--bad)}

  /* ---- shell: collapsible left sidebar + main area ---- */
  .shell{display:grid;grid-template-columns:288px 1fr;gap:20px;align-items:start;transition:grid-template-columns .18s ease}
  .shell.collapsed{grid-template-columns:44px 1fr}
  .sidebar{border:1px solid var(--line);border-radius:10px;background:var(--panel);overflow:hidden;
           position:sticky;top:16px;align-self:start}
  .sidebar-head{display:flex;align-items:center;justify-content:space-between;gap:8px;
                padding:12px 14px;border-bottom:1px solid var(--line-soft)}
  .sidebar-head h3{margin:0;font-size:13.5px;font-weight:650;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
  .collapse-btn{background:var(--accent);color:#fff;border-color:var(--accent);font-size:13px;padding:5px 10px;
                border:1px solid var(--accent);border-radius:7px;line-height:1;cursor:pointer;flex:0 0 auto;
                box-shadow:0 1px 3px rgba(0,133,121,.25)}
  .collapse-btn:hover{background:var(--accent-dark);border-color:var(--accent-dark)}
  .sidebar-body{padding:14px}
  .shell.collapsed .sidebar-head h3{display:none}
  .shell.collapsed .sidebar-body{display:none}
  .shell.collapsed .sidebar-head{justify-content:center;padding:12px 6px}
  .shell.collapsed .collapse-btn{transform:rotate(180deg)}

  .upload-btn{display:flex;align-items:center;justify-content:center;gap:8px;width:100%;
    background:var(--accent);color:#fff;border:1px solid var(--accent);border-radius:8px;
    padding:11px 14px;font-size:13.5px;font-weight:650;cursor:pointer;margin-bottom:12px;
    box-shadow:0 2px 7px rgba(0,133,121,.24)}
  .upload-btn:hover{background:var(--accent-dark);border-color:var(--accent-dark)}
  .upload-btn svg{width:16px;height:16px;flex:0 0 auto}
  .upload-btn.small{width:auto;margin-bottom:0;padding:8px 14px;font-size:12.5px;font-weight:600;box-shadow:none}

  .doclist{display:flex;flex-direction:column;gap:8px}
  .docpill{display:flex;align-items:center;gap:7px;border:1px solid var(--line);background:var(--wash);
           border-radius:8px;padding:8px 11px;font-size:12.5px;color:#334155;cursor:pointer;user-select:none}
  .docpill.on{border-color:var(--accent);background:var(--accent-soft);color:var(--accent);font-weight:600}
  .docpill input{margin:0;flex:0 0 auto}
  .docpill span{white-space:normal;line-height:1.35}
  .scopeacts{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:12px}

  /* ---- ask bar ---- */
  .ask{display:flex;gap:10px;margin-bottom:16px;flex-wrap:wrap;align-items:center}
  .ask input[type=text]{flex:1;min-width:320px;padding:12px 14px;border:2px solid var(--accent);
        border-radius:8px;font-size:14px;font-family:var(--sans);background:#fbfffe;
        box-shadow:0 0 0 3px rgba(0,133,121,.06)}
  .ask input[type=text]:focus{outline:3px solid var(--accent-soft);border-color:var(--accent-dark)}
  button{font-family:var(--sans);font-size:13.5px;padding:11px 18px;border-radius:8px;border:1px solid var(--accent);
         background:var(--accent);color:#fff;cursor:pointer;font-weight:550}
  button:hover{background:var(--accent-dark);border-color:var(--accent-dark)}
  button.ghost{background:#fff;color:var(--ink);border-color:var(--line)}
  button.ghost:hover{background:var(--wash);color:var(--ink)}
  button.small{padding:7px 12px;font-size:12.5px}
  button:disabled{opacity:.55;cursor:not-allowed}
  select{padding:11px 10px;border:1px solid var(--line);border-radius:8px;background:#fff;
         font-family:var(--sans);font-size:13.5px;color:#334155}
  select:focus{outline:2px solid var(--accent-soft);border-color:var(--accent)}
  .ask .toggle{display:flex;align-items:center;gap:7px;border:1px solid var(--line);border-radius:8px;
               padding:11px 14px;background:#fff;font-size:13px;color:#334155;white-space:nowrap;cursor:pointer}
  .ask .toggle input{margin:0}

  /* ---- export dropdown ---- */
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

  /* ---- panels ---- */
  .cols{display:grid;grid-template-columns:1fr 1fr;gap:22px}
  @media(max-width:1180px){.cols{grid-template-columns:1fr}}
  .panelhead{display:flex;align-items:center;justify-content:space-between;margin-bottom:10px;gap:12px;min-height:38px}
  .panelhead h2{font-size:16px;margin:0;font-weight:600}
  .num{font-variant-numeric:tabular-nums;color:var(--accent);font-size:12.5px;margin-right:8px;font-weight:600}
  .hlnav{display:flex;align-items:center;gap:8px}
  .hllabel{font-size:12px;color:var(--muted);font-weight:600;margin-right:2px}
  .hlbtn{padding:4px 12px;border:1px solid var(--line);border-radius:7px;background:#fff;color:var(--accent);
         font-size:15px;font-weight:600;line-height:1;cursor:pointer;min-width:34px}
  .hlbtn:hover:not(:disabled){background:var(--accent-soft);border-color:var(--accent)}
  .hlbtn:disabled{color:#cbd5e1;cursor:default;background:#fff}
  .hlcount{font-size:12.5px;color:var(--muted);min-width:44px;text-align:center;font-variant-numeric:tabular-nums}
  .panel{border:1px solid var(--line);border-radius:10px;background:var(--panel);height:660px;overflow:auto}
  .panel .inner{padding:18px 20px}

  /* ---- source pane ---- */
  #source{font-family:var(--mono);font-size:14px;line-height:1.72;white-space:pre-wrap;word-break:break-word;color:#1e293b}
  #source .sec{color:var(--accent);font-weight:700;display:block;margin:18px 0 8px;font-family:var(--sans);font-size:13.5px}
  #source .sec:first-child{margin-top:0}
  mark{background:var(--mark);border-radius:2px;padding:0 1px;cursor:pointer}
  mark.focus{background:var(--mark-focus);text-decoration:underline;text-decoration-color:var(--rule);
             text-decoration-thickness:2px;text-underline-offset:3px}
  mark.grounded{background:#bbf7d0}
  mark.partial{background:#fde68a}
  mark.ungrounded{background:#fecaca}

  /* ---- extraction pane ---- */
  .tabs{display:inline-flex;border:1px solid var(--line);border-radius:8px;overflow:hidden}
  .tabs button{border:0;background:#fff;color:#334155;padding:8px 14px;border-radius:0;font-weight:500}
  .tabs button:hover{background:var(--wash);color:#334155}
  .tabs button.on{background:var(--accent-soft);color:var(--accent);font-weight:650}
  .group{margin-bottom:20px}
  .group h3{color:var(--accent);font-size:13.5px;margin:0 0 8px;font-weight:650;letter-spacing:.01em}
  .item{display:flex;gap:10px;padding:8px 10px;border-radius:7px;cursor:pointer;border:1px solid transparent}
  .item:hover{background:var(--wash)}
  .item.on{background:var(--accent-soft);border-color:#9fd8cf}
  .item .bullet{color:var(--muted);flex:0 0 30px;text-align:right;font-family:var(--mono);font-size:12px;padding-top:3px}
  .item .body{font-family:var(--mono);font-size:13.5px;line-height:1.6;color:#1e293b}
  .item .meta{margin-top:4px;font-family:var(--sans);font-size:11.5px;color:var(--muted)}
  .badge{display:inline-block;font-size:10.5px;padding:1px 7px;border-radius:999px;margin-left:6px;vertical-align:1px}
  .badge.grounded{background:#e7f7ee;color:#0f7a48}
  .badge.partial{background:#fef3c7;color:#92400e}
  .badge.ungrounded{background:#ffe4e6;color:#9f1239}
  .xhead{color:var(--accent);font-size:13.5px;margin:18px 0 8px;font-weight:650;letter-spacing:.01em}
  .answer{font-size:14px;line-height:1.7;white-space:pre-wrap}
  ul.points{margin:8px 0 0;padding-left:18px}
  ul.points li{margin-bottom:6px}

  /* ---- grading panel ---- */
  .gsummary{display:flex;align-items:center;gap:14px;flex-wrap:wrap;margin-bottom:14px;
    border:1px solid var(--line);border-left:3px solid var(--accent);border-radius:10px;
    background:var(--wash);padding:12px 16px}
  .gsummary .big{font-size:30px;font-weight:700;letter-spacing:-.02em;font-variant-numeric:tabular-nums}
  .gsummary .big.cb-hi{color:var(--ok)} .gsummary .big.cb-mid{color:var(--warn)} .gsummary .big.cb-lo{color:var(--bad)}
  .gsummary .biglabel{font-size:11.5px;color:var(--muted);font-weight:600;letter-spacing:.02em}
  .gsummary .divider{width:1px;align-self:stretch;background:var(--line)}
  .gsummary .prio{font-size:12.5px;color:#475569}
  .gsummary .prio b{font-variant-numeric:tabular-nums;color:var(--ink)}

  .gsection-title{color:var(--accent);font-size:12px;font-weight:700;letter-spacing:.06em;
    text-transform:uppercase;margin:18px 0 8px}
  .gsection-title:first-child{margin-top:0}
  .grades{display:grid;grid-template-columns:repeat(2,1fr);gap:10px}
  @media(max-width:520px){.grades{grid-template-columns:1fr}}
  .grade{border:1px solid var(--line);border-radius:9px;padding:10px 12px;background:#fff;position:relative}
  .grade .gk{font-size:11.5px;color:var(--muted);font-weight:600}
  .grade .gv{font-size:20px;font-weight:700;letter-spacing:-.01em;font-variant-numeric:tabular-nums;margin-top:2px}
  .grade .gsub{font-size:11px;color:#94a3b8;margin-top:1px}
  .grade .barwrap{height:6px;background:#eef2f7;border-radius:3px;margin-top:8px;overflow:hidden}
  .grade .bar{height:100%;border-radius:3px;background:var(--accent);transition:width .3s}
  .grade.na{opacity:.6}
  .grade .gv.cb-hi{color:var(--ok)} .grade .gv.cb-mid{color:var(--warn)} .grade .gv.cb-lo{color:var(--bad)}
  .grade .bar.cb-hi{background:var(--ok)} .grade .bar.cb-mid{background:#d9a441} .grade .bar.cb-lo{background:var(--bad)}

  .gflags{display:flex;flex-wrap:wrap;gap:6px;margin:10px 0 4px}
  .gflag{font-size:11px;font-weight:600;padding:3px 9px;border-radius:999px;
    background:#fdeaec;color:#b0233a;border:1px solid #f2b9c1}
  .gflag.ok{background:#e6f6ee;color:#0f7a43;border-color:#a7e0c0}
  .gnote{font-size:12px;color:#64748b;line-height:1.55;margin-top:6px}
  .gnote b{color:#475569}
  .glist{margin:6px 0 0;padding-left:16px;font-size:12px;color:#64748b}
  .glist li{margin-bottom:3px}
  .gjudge{border:1px solid var(--line);border-radius:9px;background:var(--wash);
    padding:10px 12px;font-size:12.5px;color:#475569;margin-top:10px;line-height:1.55}

  /* ---- footer ---- */
  .footer{margin-top:18px;border:1px solid var(--line);border-radius:10px;background:var(--wash);
          padding:12px 18px;display:flex;gap:22px;align-items:center;flex-wrap:wrap;font-size:12.5px;color:#475569}
  .footer label{display:flex;align-items:center;gap:7px;cursor:pointer}
  .diag{font-family:var(--mono);font-size:12.5px;color:var(--muted)}
  .empty{color:var(--muted);padding:40px 4px;text-align:center;font-size:13px}
  .spin{display:inline-block;width:13px;height:13px;border:2px solid #9fd8cf;border-top-color:var(--accent);
        border-radius:50%;animation:s .7s linear infinite;vertical-align:-2px;margin-right:8px}
  @keyframes s{to{transform:rotate(360deg)}}
  .note{background:#fffbeb;border:1px solid #fde68a;color:#92400e;border-radius:8px;padding:9px 12px;margin-bottom:14px;font-size:12.5px}

  /* ---- batch modal ---- */
  .modal-overlay{display:none;position:fixed;inset:0;background:rgba(15,43,40,.45);z-index:60;
    align-items:flex-start;justify-content:center;padding:5vh 20px}
  .modal-overlay.open{display:flex}
  .modal{background:#fff;border-radius:12px;width:100%;max-width:720px;max-height:90vh;
    display:flex;flex-direction:column;box-shadow:0 20px 50px rgba(0,0,0,.25)}
  .modal-head{display:flex;align-items:center;justify-content:space-between;padding:16px 20px;border-bottom:1px solid var(--line)}
  .modal-head h3{margin:0;font-size:16px;font-weight:650}
  .modal-close{background:none;border:none;color:var(--muted);font-size:22px;line-height:1;padding:2px 6px;cursor:pointer}
  .modal-close:hover{color:var(--ink);background:none}
  .modal-body{padding:16px 20px;overflow-y:auto}
  .modal-hint{margin:0 0 10px;font-size:12.5px;color:var(--muted)}
  #batchText{width:100%;min-height:120px;padding:10px 12px;border:1px solid var(--line);
    border-radius:8px;font-size:13.5px;font-family:var(--sans);resize:vertical;box-sizing:border-box}
  #batchText:focus{outline:2px solid var(--accent-soft);border-color:var(--accent)}
  .batch-filerow{display:flex;align-items:center;gap:10px;margin-top:10px}
  .batch-filename{font-size:12px;color:var(--muted)}
  .batch-progress{display:flex;align-items:center;gap:10px;margin-top:14px}
  .batch-progress-bar{flex:1;height:8px;border-radius:999px;background:var(--wash);border:1px solid var(--line);overflow:hidden}
  .batch-progress-bar > div{height:100%;background:var(--accent);width:2%;transition:width .25s ease}
  .batch-progress span{font-size:12px;color:var(--muted);white-space:nowrap;font-variant-numeric:tabular-nums}
  .batch-results{margin-top:16px}
  .batch-table{width:100%;border-collapse:collapse;font-size:12.5px}
  .batch-table th{text-align:left;color:var(--muted);font-weight:600;font-size:11.5px;text-transform:uppercase;
    letter-spacing:.03em;padding:6px 8px;border-bottom:1px solid var(--line)}
  .batch-table td{padding:8px;border-bottom:1px solid var(--line-soft);vertical-align:top}
  .batch-table tr.batch-err td{color:var(--bad)}
  .batch-table td:nth-child(1){color:var(--muted);font-variant-numeric:tabular-nums}
  .modal-foot{display:flex;justify-content:flex-end;gap:10px;padding:14px 20px;border-top:1px solid var(--line)}
</style>
</head>
<body>

<div class="brandbar">
  <div class="brandbar-inner">
    <span class="brandmark">ICON<sup>plc</sup></span>
    <span class="brandbar-div"></span>
    <span class="brandbar-tag">Protocol Transformation Studio</span>
  </div>
</div>

<div class="wrap">

  <div class="topline">
    <div>
      <div class="eyebrow">Source &rarr; Structure</div>
      <h1>Protocol Transformation Studio</h1>
      <p class="sub">Focus a structured item to illuminate its exact source span.</p>
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
      <div class="docchip" id="docchip"><span class="dot off"></span><span>loading…</span></div>
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
          <button class="ghost small" id="selAll">First doc</button>
          <button class="ghost small" id="selNone">Clear view</button>
        </div>
        <div class="doclist" id="doclist"><span class="empty" style="padding:6px">Loading documents…</span></div>
      </div>
    </aside>

    <div class="main">
      <div class="stats">
        <div class="stat"><b id="s-extracted">0</b><span>Extracted items</span></div>
        <div class="stat g"><b id="s-grounded">0</b><span>Source-grounded</span></div>
        <div class="stat p"><b id="s-partial">0</b><span>Partial match</span></div>
        <div class="stat u"><b id="s-ungrounded">0</b><span>Ungrounded</span></div>
        <div class="stat"><b id="s-grade">–</b><span>Overall grade</span></div>
      </div>

      <div class="ask">
        <input type="text" id="q" placeholder="Ask the protocol — e.g. What are the exclusion criteria?" autocomplete="off">
        <select id="mode" title="Retrieval mode">
          <option value="auto">Auto route</option>
          <option value="section">Force whole section</option>
          <option value="hybrid">Force passages</option>
        </select>
        <button id="go">Extract</button>
        <button type="button" class="ghost" id="batchBtn" title="Ask multiple questions at once">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" style="width:15px;height:15px;vertical-align:-2px;margin-right:5px">
            <path d="M8 6h13"></path><path d="M8 12h13"></path><path d="M8 18h13"></path>
            <path d="M3 6h.01"></path><path d="M3 12h.01"></path><path d="M3 18h.01"></path>
          </svg>Batch
        </button>
        <label class="toggle" title="Run the groundedness judge on each answer">
          <input type="checkbox" id="useJudge" checked> Run groundedness judge
        </label>
      </div>

      <div id="note-slot"></div>

      <div class="cols">
        <section>
          <div class="panelhead">
            <h2><span class="num">01</span>Full document</h2>
            <div class="hlnav">
              <span class="hllabel">Highlights</span>
              <button class="hlbtn" id="hlprev" title="Previous highlight" disabled>&#8249;</button>
              <span class="hlcount" id="hlcount">0 / 0</span>
              <button class="hlbtn" id="hlnext" title="Next highlight" disabled>&#8250;</button>
            </div>
          </div>
          <div class="panel"><div class="inner"><div id="source"><div class="empty">Loading document…</div></div></div></div>
        </section>

        <section>
          <div class="panelhead">
            <h2><span class="num">02</span>Structured extraction</h2>
            <div class="tabs">
              <button class="on" data-tab="items">Items</button>
              <button data-tab="answer">Answer</button>
              <button data-tab="grading">Grading</button>
            </div>
          </div>
          <div class="panel"><div class="inner">
            <div id="tab-items"><div class="empty">Extracted items appear here, each linked to its source span.</div></div>
            <div id="tab-answer" style="display:none"></div>
            <div id="tab-grading" style="display:none"><div class="empty">Ask a question to see the grading breakdown.</div></div>
          </div></div>
        </section>
      </div>

      <div class="footer">
        <span>Model: <b id="f-model">—</b></span>
        <span>Judge: <b id="f-judge">—</b></span>
        <span>Retrieval: <b id="f-retr">—</b></span>
        <span class="diag" id="f-diag"></span>
      </div>
    </div>
  </div>
</div>

<div class="modal-overlay" id="batchOverlay">
  <div class="modal">
    <div class="modal-head">
      <h3>Ask multiple questions</h3>
      <button type="button" class="modal-close" id="batchClose" aria-label="Close">&times;</button>
    </div>
    <div class="modal-body">
      <p class="modal-hint">One question per line. Each runs against the active document <span id="batchDocCount"></span> and is logged to the audit trail / export just like a single question.</p>
      <textarea id="batchText" placeholder="What are the exclusion criteria?&#10;What is the primary endpoint?&#10;What is the dosing schedule?"></textarea>
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
let PAYLOAD = null, FOCUS = null;
let DOCS = [], ACTIVE = null;                 // active doc_id (studio answers against one doc)
let DOC_TEXT = '', SECTION_SPANS = [];
let BATCH_RESULTS = [];
const MAX_BATCH_QUESTIONS = 50;

/* ---------------- docs + meta ---------------- */
function loadDocs(){
  return fetch('/api/docs').then(r => r.json()).then(m => {
    DOCS = m.documents || [];
    $('#f-model').textContent = m.model + (m.llm_available ? '' : ' (no key — structural mode)');
    $('#f-judge').textContent = m.judge_model;
    if (!ACTIVE && DOCS.length) ACTIVE = DOCS[0].doc_id;
    renderDocList();
    updateDocChip();
    if (ACTIVE) return loadMeta(ACTIVE);
  });
}

function updateDocChip(){
  const d = DOCS.find(x => x.doc_id === ACTIVE);
  if (!d){ $('#docchip').innerHTML = '<span class="dot off"></span><span>no document</span>'; return; }
  $('#docchip').innerHTML =
    `<span class="dot"></span><span>${esc(d.title)} · v${d.version} · ${d.pages} pages · ${d.sections} sections</span>`;
}

function renderDocList(){
  const box = $('#doclist');
  if (!DOCS.length){
    box.innerHTML = '<span class="empty" style="padding:6px">No documents yet. Upload PDFs to begin.</span>';
    return;
  }
  box.innerHTML = DOCS.map(d => `
    <label class="docpill ${d.doc_id === ACTIVE ? 'on' : ''}" data-id="${d.doc_id}" title="${esc(d.title)}">
      <input type="radio" name="activedoc" ${d.doc_id === ACTIVE ? 'checked' : ''} data-id="${d.doc_id}">
      <span>${esc(d.title)}${d.version ? ' · v' + esc(d.version) : ''}</span>
    </label>`).join('');
  $$('#doclist input').forEach(cb => cb.onchange = () => {
    ACTIVE = cb.dataset.id;
    renderDocList();
    updateDocChip();
    PAYLOAD = null;
    resetStats();
    $('#tab-items').innerHTML = '<div class="empty">Extracted items appear here, each linked to its source span.</div>';
    $('#tab-answer').innerHTML = '';
    loadMeta(ACTIVE);
  });
}

$('#selAll').onclick  = () => { if (DOCS.length){ ACTIVE = DOCS[0].doc_id; renderDocList(); updateDocChip(); loadMeta(ACTIVE); } };
$('#selNone').onclick = () => { ACTIVE = null; renderDocList(); updateDocChip();
  DOC_TEXT=''; SECTION_SPANS=[]; PAYLOAD=null; renderSource(); };

function loadMeta(docId){
  return fetch('/api/meta?id=' + encodeURIComponent(docId)).then(r => r.json()).then(m => {
    if (m.error){ note(m.error); return; }
    $('#f-retr').textContent = (m.dense ? 'dense + BM25' : 'BM25 only') + (m.reranker ? ' + rerank' : '');
    DOC_TEXT = m.document_text || '';
    SECTION_SPANS = m.section_spans || [];
    PAYLOAD = null;
    renderSource();
  });
}

/* ---------------- upload ---------------- */
$('#file').onchange = async e => {
  const files = Array.from(e.target.files || []);
  if (!files.length) return;
  $('#uplabel').innerHTML = '<span class="spin"></span>Uploading';
  let lastId = null;
  for (const f of files){
    const fd = new FormData();
    fd.append('file', f);
    try {
      const res = await fetch('/api/upload', {method:'POST', body: fd});
      const j = await res.json();
      if (j.error){ note(j.error); }
      else if (j.doc_id){ lastId = j.doc_id; }
    } catch(err){ note('Upload failed: ' + err); }
  }
  $('#uplabel').textContent = 'Upload PDF(s)';
  $('#file').value = '';
  if (lastId) ACTIVE = lastId;
  await loadDocs();
};

/* ---------------- ask ---------------- */
$('#go').onclick = ask;
$('#q').addEventListener('keydown', e => { if (e.key === 'Enter') ask(); });

function note(msg){ $('#note-slot').innerHTML = `<div class="note">${esc(msg)}</div>`; }

async function ask(){
  const question = $('#q').value.trim();
  if (!question) return;
  if (!ACTIVE){ note('Select or upload a document first.'); return; }
  $('#go').disabled = true;
  $('#go').innerHTML = '<span class="spin"></span>Working';
  $('#note-slot').innerHTML = '';
  try{
    const res = await fetch('/api/ask', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({question, doc_id: ACTIVE, mode: $('#mode').value, judge: $('#useJudge').checked})
    });
    PAYLOAD = await res.json();
    if (PAYLOAD.error){ note(PAYLOAD.error); return; }
    render();
  } catch(err){ note('Request failed: ' + err); }
  finally {
    $('#go').disabled = false;
    $('#go').textContent = 'Extract';
  }
}

/* ---------------- export dropdown (CSV / Excel / JSON) ---------------- */
$('#exportBtn').onclick = e => { e.stopPropagation(); $('#exportMenu').classList.toggle('open'); };
$$('#exportMenu button').forEach(b => b.onclick = e => {
  e.stopPropagation();
  location.href = '/api/export?format=' + encodeURIComponent(b.dataset.fmt);
  $('#exportMenu').classList.remove('open');
});
document.addEventListener('click', () => $('#exportMenu').classList.remove('open'));

/* ---------------- collapsible sidebar ---------------- */
$('#collapseBtn').onclick = () => {
  const shell = $('#shell');
  shell.classList.toggle('collapsed');
  const collapsed = shell.classList.contains('collapsed');
  $('#collapseBtn').title = collapsed ? 'Expand panel' : 'Collapse panel';
};

/* Arrow keys navigate highlights when not typing. */
document.addEventListener('keydown', e => {
  if (e.target && (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA')) return;
  if (e.key === 'ArrowRight'){ gotoHighlight(1); }
  else if (e.key === 'ArrowLeft'){ gotoHighlight(-1); }
});

/* ---------------- batch ask ---------------- */
function openBatch(){
  const d = DOCS.find(x => x.doc_id === ACTIVE);
  $('#batchDocCount').textContent = d ? `(${esc(d.title)})` : '(no document selected yet)';
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

// Load questions from a .txt (one per line) or .csv (first column) file.
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

async function runBatch(){
  const questions = [...new Set(
    $('#batchText').value.split(/\r?\n/).map(s => s.trim()).filter(Boolean)
  )];
  if (!questions.length){ $('#batchResults').innerHTML = '<div class="note">Enter at least one question, one per line.</div>'; return; }
  if (!ACTIVE){ $('#batchResults').innerHTML = '<div class="note">Select or upload a document before running a batch.</div>'; return; }
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
  const judge = $('#useJudge').checked;
  const mode  = $('#mode').value;

  for (let i = 0; i < total; i++){
    const question = questions[i];
    $('#batchProgressLabel').textContent = `${i} / ${total} — asking "${question.length > 40 ? question.slice(0, 40) + '…' : question}"`;
    $('#batchProgressFill').style.width = Math.max(4, Math.round((i / total) * 100)) + '%';
    try {
      const res = await fetch('/api/ask', {
        method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({question, doc_id: ACTIVE, mode, judge})
      });
      const payload = await res.json();
      if (payload.error){
        BATCH_RESULTS.push({index: i, question, ok: false, error: payload.error});
      } else {
        const j = payload.judge || {};
        const gr = payload.grades || {};
        BATCH_RESULTS.push({
          index: i, question, ok: true,
          answer: payload.answer,
          judge: j.confidence,
          grade: gr.overall_grade,
          grounded: (payload.stats || {}).grounded,
          extracted: (payload.stats || {}).extracted,
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
}

function renderBatchResults(){
  const box = $('#batchResults');
  if (!BATCH_RESULTS.length){ box.innerHTML = ''; return; }
  box.innerHTML = `
    <table class="batch-table">
      <thead><tr><th>#</th><th>Question</th><th>Answer</th><th>Items</th><th>Grade</th><th>Judge</th><th></th></tr></thead>
      <tbody>
        ${BATCH_RESULTS.map(r => {
          if (!r.ok){
            return `<tr class="batch-err"><td>${r.index + 1}</td><td>${esc(r.question)}</td>
                    <td colspan="5">Failed: ${esc(r.error || 'unknown error')}</td></tr>`;
          }
          const a = r.answer || '';
          const snippet = a.length > 140 ? a.slice(0, 140) + '…' : a;
          return `<tr>
            <td>${r.index + 1}</td>
            <td>${esc(r.question)}</td>
            <td>${esc(snippet)}</td>
            <td>${r.extracted != null ? r.extracted : '–'}</td>
            <td>${r.grade != null ? Math.round(r.grade * 100) + '%' : '–'}</td>
            <td>${r.judge != null ? r.judge.toFixed(2) : '–'}</td>
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
function resetStats(){
  $('#s-extracted').textContent = '0';
  $('#s-grounded').textContent = '0';
  $('#s-partial').textContent = '0';
  $('#s-ungrounded').textContent = '0';
  $('#s-grade').textContent = '–';
  $('#s-grade').className = '';
}

function render(){
  const st = PAYLOAD.stats;
  $('#s-extracted').textContent = st.extracted;
  $('#s-grounded').textContent = st.grounded;
  $('#s-partial').textContent = st.partial;
  $('#s-ungrounded').textContent = st.ungrounded;

  if (PAYLOAD.note) $('#note-slot').innerHTML = `<div class="note">${esc(PAYLOAD.note)}</div>`;

  const d = PAYLOAD.diagnostics || {};
  const j = PAYLOAD.judge || {};
  const route = (d.routing && d.routing.router) ? ' · route: ' + d.routing.router : '';
  const gg = PAYLOAD.grades || {};
  $('#f-diag').textContent =
    `mode=${PAYLOAD.mode} · chunks=${d.chunks_returned ?? '–'} · sections=${PAYLOAD.sections.length}` +
    (gg.overall_grade != null ? ` · grade=${Math.round(gg.overall_grade*100)}%` : '') +
    (j.confidence != null ? ` · judge=${j.confidence.toFixed(2)}` : '') +
    (j.context_sufficient === false ? ' · context flagged insufficient' : '') +
    ((d.sections_completed && d.sections_completed.length)
       ? ` · completed ${d.sections_completed.length} section(s)` : '') +
    route;

  const g = PAYLOAD.grades || {};
  const og = g.overall_grade;
  const gcell = $('#s-grade');
  gcell.textContent = pct(og);
  gcell.style.color = og == null ? '' :
    (og >= 0.8 ? 'var(--ok)' : og >= 0.5 ? 'var(--warn)' : 'var(--bad)');

  FOCUS = null;
  renderSource();
  renderItems();
  renderAnswer();
  renderGrades();
}

/* value -> confidence band class */
function cb(v){ return v == null ? '' : (v >= 0.8 ? 'cb-hi' : v >= 0.5 ? 'cb-mid' : 'cb-lo'); }
function pct(v){ return v == null ? '–' : Math.round(v * 100) + '%'; }
function num(v, d){ return v == null ? '–' : (typeof v === 'number' ? v.toFixed(d ?? 2) : v); }

function items(){
  return (PAYLOAD?.extraction?.groups || []).flatMap(g => g.items);
}

/* Render the ENTIRE document text once, with section headers injected at their
   offsets and the current answer's spans wrapped in <mark>, tinted by grounding. */
function renderSource(){
  const text = DOC_TEXT;
  if (!text){ $('#source').innerHTML = '<div class="empty">No document loaded.</div>'; return; }

  const marks = (PAYLOAD ? items() : [])
    .filter(i => i.doc_span)
    .map(i => ({start: i.doc_span.start, end: i.doc_span.end, id: i.id,
                grounding: i.grounding, score: i.score}))
    .sort((a,b) => a.start - b.start);

  const heads = SECTION_SPANS
    .map(s => ({at: s.start, label: s.title, page: s.page}))
    .sort((a,b) => a.at - b.at);

  let html = '', cursor = 0, last = -1, hi = 0;

  const flushHeadsUpTo = (limit) => {
    let chunk = '';
    while (hi < heads.length && heads[hi].at <= limit){
      const h = heads[hi];
      if (h.at >= cursor){
        chunk += esc(text.slice(cursor, h.at));
        chunk += `<span class="sec" id="sec-${hi}">[${esc(h.label)}]  (p. ${h.page})</span>`;
        cursor = h.at;
      }
      hi++;
    }
    return chunk;
  };

  for (const s of marks){
    if (s.start < last) continue;                          // skip overlaps
    html += flushHeadsUpTo(s.start);
    html += esc(text.slice(cursor, s.start));
    const gl = s.grounding ? ' ' + s.grounding : '';
    html += `<mark id="m-${s.id}" class="${gl.trim()}" data-id="${s.id}" `
          + `title="${s.grounding || ''}${s.score ? ' ' + Math.round(s.score) : ''}">`
          + `${esc(text.slice(s.start, s.end))}</mark>`;
    cursor = s.end; last = s.end;
  }
  html += flushHeadsUpTo(text.length);
  html += esc(text.slice(cursor));

  $('#source').innerHTML = html;
  $$('#source mark').forEach(m => m.onclick = () => {
    const idx = HL_ORDER.indexOf(m.id);
    if (idx !== -1) HL_INDEX = idx;
    focusItem(m.dataset.id, 'source');
    updateHlNav();
  });

  HL_ORDER = $$('#source mark').map(m => m.id);
  HL_INDEX = HL_ORDER.length ? 0 : -1;
  updateHlNav();
}

/* ---------------- highlight navigation ---------------- */
let HL_ORDER = [], HL_INDEX = -1;

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
  $$('#source mark').forEach(m => m.classList.toggle('focus', m.id === HL_ORDER[HL_INDEX]));
}

function gotoHighlight(delta){
  if (!HL_ORDER.length) return;
  HL_INDEX = Math.max(0, Math.min(HL_ORDER.length - 1, HL_INDEX + delta));
  const mark = $('#' + HL_ORDER[HL_INDEX]);
  if (mark){
    mark.scrollIntoView({behavior:'smooth', block:'center'});
    if (mark.dataset.id) focusItem(mark.dataset.id, 'source');
  }
  updateHlNav();
}

$('#hlprev').onclick = () => gotoHighlight(-1);
$('#hlnext').onclick = () => gotoHighlight(1);

function renderItems(){
  const groups = PAYLOAD.extraction.groups || [];
  if (!groups.length){
    $('#tab-items').innerHTML = '<div class="empty">No discrete items extracted for this question — see the Answer tab.</div>';
    return;
  }
  $('#tab-items').innerHTML = groups.map(g => `
    <div class="group">
      <h3>${esc(g.heading || 'Findings')}</h3>
      ${g.items.map(it => `
        <div class="item" id="it-${it.id}" data-id="${it.id}"
             style="margin-left:${Math.min(it.depth || 0, 4) * 16}px">
          <div class="bullet">${esc(it.marker || '\u2022')}</div>
          <div>
            <div class="body">${esc(it.text)}</div>
            <div class="meta">
              ${it.page ? 'p. ' + it.page + ' · ' : ''}${esc(it.section || '')}
              <span class="badge ${it.grounding}">${it.grounding}${it.score ? ' ' + Math.round(it.score) : ''}</span>
            </div>
          </div>
        </div>`).join('')}
    </div>`).join('');
  $$('.item').forEach(el => el.onclick = () => focusItem(el.dataset.id, 'items'));
}

function renderAnswer(){
  const pts = (PAYLOAD.key_points || []).map(p => `<li>${esc(p)}</li>`).join('');
  const j = PAYLOAD.judge || {};
  $('#tab-answer').innerHTML = `
    <div class="answer">
      <div class="xhead" style="margin-top:0">Answer</div>
      ${esc(PAYLOAD.answer || 'No narrative answer was generated (LLM not configured).')}
      ${pts ? `<div class="xhead">Key points</div><ul class="points">${pts}</ul>` : ''}
      <div class="xhead">Sections used</div>
      <ul class="points">${PAYLOAD.sections.map(s =>
        `<li>${esc(s.title)} <span style="color:#94a3b8">(pp. ${s.page_start}–${s.page_end})</span></li>`).join('')}</ul>
      ${j.reasoning ? `<div class="xhead">Groundedness judge</div>
        <div style="color:#475569">${j.confidence != null ? j.confidence.toFixed(2) + ' — ' : ''}${esc(j.reasoning)}</div>` : ''}
    </div>`;
}

/* ---------------- grading ---------------- */
function gradeCard(label, value, kind, sub){
  // kind: 'pct' (0-1 shown as %), 'num' (raw), 'ratio' (0-1 as %)
  const na = value == null;
  let display, band = '', barPct = null;
  if (na){ display = '–'; }
  else if (kind === 'num'){ display = num(value, 2); }
  else { display = pct(value); band = cb(value); barPct = Math.round(value * 100); }
  return `<div class="grade ${na ? 'na' : ''}">
    <div class="gk">${esc(label)}</div>
    <div class="gv ${band}">${display}</div>
    ${sub ? `<div class="gsub">${esc(sub)}</div>` : ''}
    ${barPct != null ? `<div class="barwrap"><div class="bar ${band}" style="width:${barPct}%"></div></div>` : ''}
  </div>`;
}

function renderGrades(){
  const box = $('#tab-grading');
  const g = PAYLOAD.grades;
  if (!g){ box.innerHTML = '<div class="empty">No grading available for this answer.</div>'; return; }

  const og = g.overall_grade;
  const flags = g.review_flags || [];

  // headline summary
  let html = `<div class="gsummary">
    <div>
      <div class="big ${cb(og)}">${pct(og)}</div>
      <div class="biglabel">OVERALL GRADE</div>
    </div>
    <div class="divider"></div>
    <div class="prio">
      Review priority <b>${g.review_priority != null ? g.review_priority.toFixed(2) : '–'}</b>
      <span style="color:#94a3b8">(lower = more scrutiny)</span><br>
      ${flags.length
        ? `<span style="color:var(--bad)">${flags.length} flag${flags.length===1?'':'s'} raised</span>`
        : `<span style="color:var(--ok)">no flags raised</span>`}
    </div>
  </div>`;

  // flags
  if (flags.length){
    html += `<div class="gflags">` +
      flags.map(f => `<span class="gflag">${esc(f.replace(/_/g,' '))}</span>`).join('') +
      `</div>`;
  }

  // ANSWER QUALITY
  html += `<div class="gsection-title">Answer quality</div><div class="grades">`;
  html += gradeCard('Groundedness judge', g.judge_confidence, 'pct',
                    g.context_sufficient === false ? 'context flagged insufficient' : 'single-judge, source-supported');
  html += gradeCard('Claim grounding', g.claim_grounding_ratio, 'ratio',
                    g.claims_total ? `${g.claims_supported}/${g.claims_total} claims supported` : 'no claims parsed');
  html += gradeCard('Semantic match', g.semantic_match, 'pct', 'answer vs source cosine');
  html += gradeCard('Item coverage', g.item_coverage_ratio, 'ratio',
                    g.items_covered != null ? `${g.items_covered} of ${g.items_covered + g.items_uncovered} items in prose` : '');
  html += gradeCard('Structure', g.structure_score, 'ratio',
                    `${g.n_paragraphs || 0} para · longest ${g.longest_paragraph_sentences || 0} sent`);
  html += `</div>`;

  // structure detail
  if (g.structure_flags && g.structure_flags.length){
    html += `<div class="gnote"><b>Structure issues:</b> ${g.structure_flags.map(f=>esc(f.replace(/_/g,' '))).join(', ')}.</div>`;
  }

  // RETRIEVAL & ITEMS
  html += `<div class="gsection-title">Retrieval &amp; items</div><div class="grades">`;
  html += gradeCard('Item grounding', g.item_grounding_ratio, 'ratio',
                    g.item_mean_score != null ? `mean align ${Math.round(g.item_mean_score)}` : '');
  html += gradeCard('Section coverage', g.section_coverage_ratio, 'ratio',
                    g.list_sections_total ? `${g.list_sections_complete}/${g.list_sections_total} list sections complete` : '');
  html += gradeCard('Retrieval mean', g.retrieval_score_mean, 'num',
                    g.retrieval_n ? `${g.retrieval_n} chunks · spread ${num(g.retrieval_score_spread,2)}` : '');
  html += gradeCard('Weak chunks', g.weak_chunks_ratio, 'ratio', 'lower is better');
  html += gradeCard('Sub-query agreement', g.subquery_agreement, 'ratio', 'recall stability');
  html += gradeCard('Context utilisation', g.context_utilisation, 'ratio', 'chunks close to answer');
  html += `</div>`;

  // partial list sections (the completeness failure this pipeline guards)
  if (g.partial_list_sections && g.partial_list_sections.length){
    html += `<div class="gnote"><b>Incompletely retrieved list sections:</b><ul class="glist">` +
      g.partial_list_sections.map(s =>
        `<li>${esc(s.section)} — ${esc(s.chunks)} chunks (${Math.round((s.coverage||0)*100)}%)</li>`).join('') +
      `</ul></div>`;
  }

  // contradicted / unsupported claims
  if (g.contradicted_claims && g.contradicted_claims.length){
    html += `<div class="gnote"><b style="color:var(--bad)">Contradicted claims (${g.contradicted_claims.length}):</b><ul class="glist">` +
      g.contradicted_claims.map(c => `<li>${esc(c)}</li>`).join('') + `</ul></div>`;
  }
  if (g.unsupported_claims && g.unsupported_claims.length){
    html += `<div class="gnote"><b>Unsupported claims (${g.unsupported_claims.length}) — not found in context:</b><ul class="glist">` +
      g.unsupported_claims.slice(0,10).map(c => `<li>${esc(c)}</li>`).join('') + `</ul></div>`;
  }
  // uncovered items (in the source but missing from the prose answer)
  if (g.uncovered_items && g.uncovered_items.length){
    html += `<div class="gnote"><b>Items in the extraction but missing from the prose answer (${g.items_uncovered}):</b><ul class="glist">` +
      g.uncovered_items.slice(0,10).map(c => `<li>${esc(c)}</li>`).join('') + `</ul></div>`;
  }

  // judge reasoning
  if (g.judge_reasoning){
    html += `<div class="gjudge"><b>Groundedness judge:</b> ${esc(g.judge_reasoning)}</div>`;
  }

  box.innerHTML = html;
}

function focusItem(id, from){
  FOCUS = id;
  $$('#source mark').forEach(m => m.classList.toggle('focus', m.dataset.id === id));
  $$('.item').forEach(i => i.classList.toggle('on', i.dataset.id === id));
  const target = from === 'items' ? $('#m-' + id) : $('#it-' + id);
  if (target) target.scrollIntoView({behavior:'smooth', block:'center'});
}

/* ---------------- tabs ---------------- */
$$('.tabs button').forEach(b => b.onclick = () => {
  $$('.tabs button').forEach(x => x.classList.toggle('on', x === b));
  ['items','answer','grading'].forEach(t =>
    $('#tab-' + t).style.display = (t === b.dataset.tab ? '' : 'none'));
});

function esc(s){
  return (s ?? '').replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
}

loadDocs();
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
    # Ensure at least the default document(s) are loaded.
    if not DOC_ORDER:
        try:
            _register_docs_from_paths(P.PDF_PATHS)
        except Exception:
            traceback.print_exc()
    documents = [_doc_meta(doc_id) for doc_id in DOC_ORDER]
    return jsonify({
        "documents": documents,
        "model": P.MODEL_NAME,
        "judge_model": P.JUDGE_MODEL,
        "llm_available": P.llm_available(),
    })


@app.route("/api/meta")
def meta():
    doc_id = request.args.get("id") or (DOC_ORDER[0] if DOC_ORDER else None)
    if not doc_id or doc_id not in RETRIEVERS:
        if not DOC_ORDER:
            try:
                _register_docs_from_paths(P.PDF_PATHS)
            except Exception as e:
                return jsonify({"error": f"no documents loaded: {e}"}), 400
        doc_id = doc_id if doc_id in RETRIEVERS else (DOC_ORDER[0] if DOC_ORDER else None)
    if not doc_id or doc_id not in RETRIEVERS:
        return jsonify({"error": "unknown document"}), 404

    r = RETRIEVERS[doc_id]
    d = r.doc
    return jsonify({
        "document": {"id": d.doc_id, "title": d.title or d.doc_id,
                     "version": d.version_label, "pages": d.n_pages,
                     "sections": len(d.sections), "chunks": len(d.chunks),
                     "chars": len(d.text)},
        "model": P.MODEL_NAME,
        "judge_model": P.JUDGE_MODEL,
        "llm_available": P.llm_available(),
        "dense": r.index is not None,
        "reranker": r.reranker is not None,
        "toc": [{"title": s.full_title, "level": s.level,
                 "page": s.page_start + 1} for s in d.sections],
        # full document text + section markers, so the left pane can show the
        # entire PDF and highlight the answer's spans inside it
        "document_text": d.text,
        "section_spans": [{"title": s.full_title,
                           "page": s.page_start + 1,
                           "start": s.char_start} for s in d.sections],
    })


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

        if not os.path.exists(dest) or os.path.getsize(dest) == 0:
            return jsonify({"error": "uploaded file is empty"}), 400

        new_ids = _register_docs_from_paths([dest])
        if not new_ids:
            return jsonify({"error": "could not parse the uploaded PDF"}), 400
        doc_id = new_ids[-1]
        m = _doc_meta(doc_id)
        return jsonify({"doc_id": doc_id, "title": m["title"], "version": m["version"]})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"failed to process PDF: {e}"}), 500


def _log_answer(question: str, mode: str, payload: dict) -> int:
    global _ENTRY_SEQ
    _ENTRY_SEQ += 1
    entry_id = _ENTRY_SEQ
    g = payload.get("grades", {}) or {}

    def _join(lst):
        return " | ".join(str(x) for x in (lst or []))

    SESSION_LOG.append({
        "entry_id": entry_id,
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "document": payload["document"].get("title") or payload["document"].get("id"),
        "question": question,
        "mode": payload["mode"],
        "sections": " | ".join(s["title"] for s in payload["sections"]),
        "items": payload["stats"]["extracted"],
        "grounded": payload["stats"]["grounded"],
        "partial": payload["stats"]["partial"],
        "ungrounded": payload["stats"]["ungrounded"],
        # ---- Grader G0: single groundedness judge (kept) --------------------
        "judge_confidence": payload["judge"].get("confidence"),
        "context_sufficient": payload["judge"].get("context_sufficient"),
        "judge_reasoning": payload["judge"].get("reasoning"),
        # ---- Headline -------------------------------------------------------
        "overall_grade": g.get("overall_grade"),
        "review_priority": g.get("review_priority"),
        "review_flags": _join(g.get("review_flags")),
        # ---- Answer graders -------------------------------------------------
        "semantic_match": g.get("semantic_match"),
        "claim_grounding_ratio": g.get("claim_grounding_ratio"),
        "claims_total": g.get("claims_total"),
        "claims_supported": g.get("claims_supported"),
        "claims_unsupported": g.get("claims_unsupported"),
        "claims_contradicted": g.get("claims_contradicted"),
        "unsupported_claims": _join(g.get("unsupported_claims")),
        "contradicted_claims": _join(g.get("contradicted_claims")),
        "item_coverage_ratio": g.get("item_coverage_ratio"),
        "items_uncovered": g.get("items_uncovered"),
        "uncovered_items": _join(g.get("uncovered_items")),
        "structure_score": g.get("structure_score"),
        "structure_flags": _join(g.get("structure_flags")),
        "n_paragraphs": g.get("n_paragraphs"),
        "n_sentences": g.get("n_sentences"),
        "longest_paragraph_sentences": g.get("longest_paragraph_sentences"),
        # ---- Retrieval / item graders --------------------------------------
        "item_grounding_ratio": g.get("item_grounding_ratio"),
        "item_mean_score": g.get("item_mean_score"),
        "section_coverage_ratio": g.get("section_coverage_ratio"),
        "list_sections_total": g.get("list_sections_total"),
        "list_sections_complete": g.get("list_sections_complete"),
        "retrieval_score_max": g.get("retrieval_score_max"),
        "retrieval_score_min": g.get("retrieval_score_min"),
        "retrieval_score_mean": g.get("retrieval_score_mean"),
        "retrieval_score_spread": g.get("retrieval_score_spread"),
        "weak_chunks_ratio": g.get("weak_chunks_ratio"),
        "subquery_agreement": g.get("subquery_agreement"),
        "context_utilisation": g.get("context_utilisation"),
        "answer": payload["answer"],
    })
    return entry_id


@app.route("/api/ask", methods=["POST"])
def ask():
    body = request.get_json(force=True) or {}
    question = (body.get("question") or "").strip()
    mode = body.get("mode") or "auto"
    doc_id = body.get("doc_id")
    run_judge = bool(body.get("judge", True))
    if not question:
        return jsonify({"error": "empty question"}), 400
    try:
        retriever = get_retriever(doc_id)
        payload = answer_question(
            retriever, question,
            force_mode=None if mode == "auto" else mode,
            run_judge=run_judge)
    except Exception as e:                                    # pragma: no cover
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

    entry_id = _log_answer(question, mode, payload)
    payload["entry_id"] = entry_id
    return jsonify(payload)


# =============================================================================
# Export (CSV / Excel / JSON) — same options and arrangement as App_2
# =============================================================================

EXPORT_COLS = [
    "entry_id", "timestamp", "document", "question", "mode", "sections",
    # extraction stats
    "items", "grounded", "partial", "ungrounded",
    # headline grade
    "overall_grade", "review_priority", "review_flags",
    # Grader G0 — single groundedness judge (kept; no consensus panel)
    "judge_confidence", "context_sufficient", "judge_reasoning",
    # Answer graders
    "semantic_match",
    "claim_grounding_ratio", "claims_total", "claims_supported",
    "claims_unsupported", "claims_contradicted",
    "unsupported_claims", "contradicted_claims",
    "item_coverage_ratio", "items_uncovered", "uncovered_items",
    "structure_score", "structure_flags",
    "n_paragraphs", "n_sentences", "longest_paragraph_sentences",
    # Retrieval / item graders
    "item_grounding_ratio", "item_mean_score",
    "section_coverage_ratio", "list_sections_total", "list_sections_complete",
    "retrieval_score_max", "retrieval_score_min", "retrieval_score_mean",
    "retrieval_score_spread", "weak_chunks_ratio",
    "subquery_agreement", "context_utilisation",
    # answer text last
    "answer",
]


@app.route("/api/export")
def export():
    fmt = (request.args.get("format") or "csv").strip().lower()
    cols = EXPORT_COLS
    rows = [{c: row.get(c) for c in cols} for row in SESSION_LOG]

    # ---- JSON ---------------------------------------------------------------
    if fmt == "json":
        return Response(
            json.dumps(rows, indent=2, default=str),
            mimetype="application/json",
            headers={"Content-Disposition": "attachment; filename=protocol_qa_session.json"})

    # ---- Excel (.xlsx) ------------------------------------------------------
    if fmt in ("xlsx", "excel"):
        try:
            import pandas as pd
        except Exception as e:
            return jsonify({"error": f"Excel export needs pandas ({e}). "
                                     f"Install with: pip install pandas openpyxl"}), 500
        df = pd.DataFrame(rows, columns=cols)
        last_err = None
        for engine in ("openpyxl", "xlsxwriter"):
            try:
                out = io.BytesIO()
                with pd.ExcelWriter(out, engine=engine) as writer:
                    df.to_excel(writer, index=False, sheet_name="QA session")
                out.seek(0)
                return Response(
                    out.getvalue(),
                    mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    headers={"Content-Disposition": "attachment; filename=protocol_qa_session.xlsx"})
            except Exception as e:
                last_err = e
        traceback.print_exc()
        return jsonify({"error": f"Excel export unavailable ({last_err}). "
                                 f"Install an .xlsx writer with: pip install openpyxl"}), 500

    # ---- CSV (default) ------------------------------------------------------
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=cols)
    w.writeheader()
    for row in rows:
        w.writerow(row)
    return Response(buf.getvalue(), mimetype="text/csv", headers={
        "Content-Disposition": "attachment; filename=protocol_qa_session.csv"})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", help="path to the protocol PDF")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--no-embeddings", action="store_true",
                    help="lexical-only mode (no model download required)")
    args = ap.parse_args()

    USE_EMBEDDINGS = not args.no_embeddings
    if args.pdf:
        P.PDF_PATHS = [args.pdf]
    # Pre-load the configured document(s) so the studio opens ready to use.
    try:
        _register_docs_from_paths(P.PDF_PATHS)
    except Exception:
        traceback.print_exc()
    app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024
    print(f"\n  Studio ready -> http://127.0.0.1:{args.port}\n")
    app.run(debug=False, port=args.port, threaded=True)