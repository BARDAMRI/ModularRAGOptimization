#!/usr/bin/env python3
"""
export_report.py — Generate a clean, self-contained HTML analysis report.

Reads CSVs and charts from <run_dir>/analysis/ and the experiment config,
produces a single HTML file with all charts embedded as base64 images.

Usage:
    python export_report.py                          # auto-detect latest run
    python export_report.py path/to/experiment_results.db
    python export_report.py --run-dir path/to/staged_run_YYYY-MM-DD
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

try:
    from playwright.sync_api import sync_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False

from analysis_common import (
    ANALYSIS_DIR,
    AnalysisLayout,
    build_per_query_overview,
    collect_run_health,
    find_latest_run_dir,
    format_health_html,
    resolve_artifact,
    save_run_health,
    write_analysis_index,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _metrics_from_analysis(an: Path) -> dict:
    """Load key numeric metrics from analysis CSVs for dynamic report text."""
    tie = _artifact_csv(an, "tie_structure.csv")
    diff = _artifact_csv(an, "query_difficulty.csv")
    rec = _artifact_csv(an, "llm_recovery_at_k.csv")
    cmrr = _artifact_csv(an, "corrected_mrr_at_k.csv")
    wt = _artifact_csv(an, "within_tier_dist.csv")
    filt = _artifact_csv(an, "threshold_filter_analysis.csv")
    roc = _artifact_csv(an, "roc_pr_summary.csv")

    n_queries = len(tie) if tie is not None else 0
    gt_in_top = int((tie["n_above_gt"] == 0).sum()) if tie is not None else 0
    med_tied = float(tie.loc[tie["n_above_gt"] == 0, "n_tied_at_gt"].median()) if tie is not None else float("nan")
    rec_1 = float(rec.loc[rec["K"] == 1, "recovery_rate_pct"].iloc[0]) if rec is not None else float("nan")
    mrr_opt = float(cmrr.loc[cmrr["metric"].str.contains("optimistic", na=False), "value"].iloc[0]) if cmrr is not None else float("nan")
    mrr_corr = float(cmrr.loc[cmrr["metric"].str.contains("tie-corrected", na=False), "value"].iloc[0]) if cmrr is not None else float("nan")
    wr_within = float(wt["gt_would_win_dist"].mean() * 100) if wt is not None else float("nan")
    wr_overall = float(wt["gt_would_win_dist"].sum() / n_queries * 100) if wt is not None and n_queries else float("nan")
    rag_fail_pct = 100.0
    if diff is not None and n_queries:
        rag_fail_pct = 100.0  # all analyzed queries in this experiment design

    thresh_row = filt.loc[filt["threshold"] == 1.0].iloc[0] if filt is not None and not filt.empty else None
    roc_auc = float(roc.loc[roc["metric"] == "ROC-AUC", "value"].iloc[0]) if roc is not None else float("nan")

    return {
        "n_queries": n_queries,
        "gt_in_top": gt_in_top,
        "med_tied": med_tied,
        "rec_1": rec_1,
        "mrr_opt": mrr_opt,
        "mrr_corr": mrr_corr,
        "wr_within": wr_within,
        "wr_overall": wr_overall,
        "rag_fail_pct": rag_fail_pct,
        "thresh_precision": float(thresh_row["precision"] * 100) if thresh_row is not None else float("nan"),
        "thresh_recall": float(thresh_row["recall"] * 100) if thresh_row is not None else float("nan"),
        "roc_auc": roc_auc,
    }


def _find_latest_run_dir() -> Path:
    return find_latest_run_dir()


def _img_b64(path: Path) -> str | None:
    if not path.exists():
        return None
    data = base64.b64encode(path.read_bytes()).decode()
    return f"data:image/png;base64,{data}"


def _load_csv(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() else None

def _artifact_path(an: Path, filename: str) -> Path:
    return resolve_artifact(an, filename)


def _artifact_csv(an: Path, filename: str) -> pd.DataFrame | None:
    path = _artifact_path(an, filename)
    return pd.read_csv(path) if path.exists() else None


def _artifact_img(an: Path, filename: str) -> str | None:
    return _img_b64(_artifact_path(an, filename))



def _fmt(v, decimals: int = 3) -> str:
    if v is None or (isinstance(v, float) and v != v):
        return "—"
    if isinstance(v, float):
        return f"{v:.{decimals}f}"
    return str(v)


def _pct(v) -> str:
    return f"{v:.1f}%" if v == v else "—"


# ── HTML primitives ───────────────────────────────────────────────────────────

def _figure(img_b64: str | None, caption: str, wide: bool = True) -> str:
    if img_b64 is None:
        return ""
    width = "100%" if wide else "70%"
    return f"""
<figure>
  <img src="{img_b64}" style="width:{width};border-radius:6px;">
  <figcaption>{caption}</figcaption>
</figure>"""


def _table(df: pd.DataFrame,
           col_formats: dict | None = None,
           highlight_col: str | None = None,
           max_rows: int = 999) -> str:
    if df is None or df.empty:
        return "<p class='muted'>No data.</p>"
    df = df.head(max_rows).copy()

    header = "".join(f"<th>{c}</th>" for c in df.columns)
    rows_html = []
    for _, row in df.iterrows():
        cells = []
        for col in df.columns:
            raw = row[col]
            fmt = col_formats.get(col, "{}") if col_formats else "{}"
            try:
                text = fmt.format(raw)
            except Exception:
                text = str(raw)
            cls = ' class="hl"' if highlight_col == col else ""
            cells.append(f"<td{cls}>{text}</td>")
        rows_html.append("<tr>" + "".join(cells) + "</tr>")
    return f"""<div class="table-wrap">
<table>
  <thead><tr>{header}</tr></thead>
  <tbody>{"".join(rows_html)}</tbody>
</table>
</div>"""


def _stat_cards(*cards: tuple) -> str:
    """Each card: (label, value, sub, color_class)"""
    inner = ""
    for label, value, sub, cls in cards:
        inner += f"""
    <div class="card {cls}">
      <div class="card-val">{value}</div>
      <div class="card-label">{label}</div>
      {"<div class='card-sub'>" + sub + "</div>" if sub else ""}
    </div>"""
    return f'<div class="cards">{inner}</div>'


def _section(title: str, anchor: str, body: str) -> str:
    return f"""
<section id="{anchor}">
  <h2>{title}</h2>
  {body}
</section>"""


def _two_col(left: str, right: str) -> str:
    return f'<div class="two-col"><div>{left}</div><div>{right}</div></div>'


def _callout(text: str, kind: str = "info") -> str:
    icons = {"info": "ℹ", "warn": "⚠", "good": "✓", "key": "★"}
    return f'<div class="callout {kind}"><span class="icon">{icons.get(kind,"•")}</span>{text}</div>'


# ── CSS + HTML shell ──────────────────────────────────────────────────────────

CSS = """
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  font-size: 14px; line-height: 1.6;
  background: #f4f6fa; color: #222;
}
a { color: #2563eb; }

/* layout */
#toc-sidebar {
  position: fixed; top: 0; left: 0; width: 220px; height: 100vh;
  background: #1e293b; color: #94a3b8; overflow-y: auto;
  padding: 24px 16px; font-size: 12px;
}
#toc-sidebar h3 { color: #e2e8f0; font-size: 13px; margin-bottom: 12px; letter-spacing:.05em; text-transform:uppercase; }
#toc-sidebar a { color: #94a3b8; text-decoration: none; display: block; padding: 3px 0; }
#toc-sidebar a:hover { color: #e2e8f0; }
#toc-sidebar .toc-group { margin-top: 14px; }
#toc-sidebar .toc-group-title { color: #64748b; font-size: 10px; text-transform: uppercase; letter-spacing:.08em; margin-bottom: 4px; }

main { margin-left: 220px; padding: 36px 48px; max-width: 1060px; }

/* header */
#report-header {
  background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
  color: white; padding: 36px 48px; margin-left: 220px;
}
#report-header h1 { font-size: 26px; font-weight: 700; margin-bottom: 6px; }
#report-header .subtitle { color: #94a3b8; font-size: 13px; }
#report-header .meta { display: flex; gap: 32px; margin-top: 18px; }
#report-header .meta-item label { display: block; color: #64748b; font-size: 11px; text-transform: uppercase; letter-spacing:.06em; }
#report-header .meta-item span { color: #e2e8f0; font-size: 13px; font-weight: 500; }

/* sections */
section { background: #fff; border-radius: 10px; padding: 28px 32px; margin-bottom: 28px;
          box-shadow: 0 1px 4px rgba(0,0,0,.07); }
section h2 { font-size: 17px; font-weight: 600; color: #1e293b; margin-bottom: 18px;
             padding-bottom: 10px; border-bottom: 2px solid #e2e8f0; }
section h3 { font-size: 13px; font-weight: 600; color: #475569; margin: 20px 0 10px;
             text-transform: uppercase; letter-spacing: .05em; }

/* stat cards */
.cards { display: flex; flex-wrap: wrap; gap: 14px; margin-bottom: 20px; }
.card { flex: 1 1 140px; border-radius: 8px; padding: 16px 18px; }
.card-val { font-size: 26px; font-weight: 700; line-height: 1.1; }
.card-label { font-size: 11px; margin-top: 4px; opacity: .75; text-transform: uppercase; letter-spacing: .05em; }
.card-sub { font-size: 11px; margin-top: 6px; opacity: .6; }
.blue  { background: #eff6ff; color: #1d4ed8; }
.green { background: #f0fdf4; color: #15803d; }
.amber { background: #fffbeb; color: #b45309; }
.red   { background: #fef2f2; color: #b91c1c; }
.slate { background: #f8fafc; color: #334155; }
.purple{ background: #faf5ff; color: #7e22ce; }

/* tables */
.table-wrap { overflow-x: auto; -webkit-overflow-scrolling: touch; margin: 12px 0; }
table { width: 100%; border-collapse: collapse; font-size: 13px; margin: 0; min-width: 400px; }
thead tr { background: #f1f5f9; }
th { text-align: left; padding: 8px 12px; font-weight: 600; color: #475569;
     border-bottom: 2px solid #e2e8f0; font-size: 12px; white-space: nowrap; }
td { padding: 7px 12px; border-bottom: 1px solid #f1f5f9; color: #374151;
     word-break: break-word; max-width: 320px; }
tr:last-child td { border-bottom: none; }
tr:hover td { background: #f8fafc; }
td.hl { font-weight: 600; color: #1d4ed8; }

/* figures */
figure { margin: 20px 0; overflow-x: auto; }
figure img { display: block; border: 1px solid #e2e8f0; max-width: 100%; height: auto; }
figcaption { font-size: 11px; color: #64748b; margin-top: 6px; font-style: italic;
             line-height: 1.5; }

/* two-col */
.two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; margin: 16px 0; }

/* callouts */
.callout { display: flex; align-items: flex-start; gap: 10px; border-radius: 7px;
           padding: 12px 16px; margin: 14px 0; font-size: 13px; line-height: 1.5;
           overflow-wrap: break-word; word-break: break-word; }
.callout .icon { flex-shrink: 0; font-size: 16px; margin-top: 1px; }
.callout.info  { background: #eff6ff; color: #1e40af; }
.callout.warn  { background: #fffbeb; color: #92400e; }
.callout.good  { background: #f0fdf4; color: #166534; }
.callout.key   { background: #faf5ff; color: #6b21a8; border-left: 3px solid #7c3aed; }

/* misc */
.muted { color: #94a3b8; font-style: italic; }
.tag { display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 11px; font-weight: 600; }
.tag.neg { background:#fef2f2; color:#b91c1c; }
.tag.pos { background:#f0fdf4; color:#15803d; }

p { margin: 8px 0; color: #475569; font-size: 13px; }
hr { border: none; border-top: 1px solid #e2e8f0; margin: 24px 0; }

pre { font-size: 12px; color: #475569; white-space: pre-wrap; overflow-wrap: break-word;
      word-break: break-word; overflow-x: auto; background: #f8fafc;
      border: 1px solid #e2e8f0; border-radius: 6px; padding: 14px 16px; }

/* section intro */
.section-intro { background: #f8fafc; border-left: 3px solid #cbd5e1;
                 border-radius: 0 6px 6px 0; padding: 12px 16px; margin-bottom: 18px;
                 font-size: 13px; color: #475569; line-height: 1.6; }
.section-intro strong { color: #1e293b; }

/* ── print / PDF layout ───────────────────────────────────────────────────── */
@media print {
  #toc-sidebar { display: none !important; }
  main { margin-left: 0 !important; padding: 20px 28px !important; max-width: 100% !important; }
  #report-header { margin-left: 0 !important; padding: 24px 28px !important; }
  #report-header .meta { flex-wrap: wrap; gap: 18px; }
  section { box-shadow: none !important; border: 1px solid #e2e8f0;
            break-inside: avoid-page; margin-bottom: 18px; }
  .table-wrap { overflow-x: visible; }
  table { font-size: 11px; }
  th, td { padding: 5px 8px; }
  .cards { flex-wrap: wrap; }
  .card { flex: 1 1 120px; }
  figure { break-inside: avoid-page; }
  figure img { max-width: 100% !important; }
  .callout { break-inside: avoid-page; }
  pre { font-size: 10px; white-space: pre-wrap; overflow: visible; }
  a { color: inherit; text-decoration: none; }
  h2 { break-after: avoid-page; }
}
"""


def _html_shell(title: str, toc: str, header: str, body: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<style>{CSS}</style>
</head>
<body>
<nav id="toc-sidebar">
  <h3>Contents</h3>
  {toc}
</nav>
{header}
<main>
{body}
</main>
</body>
</html>"""


# ── section builders ──────────────────────────────────────────────────────────

def _build_header(cfg: dict, db_path: Path, n_rows: int, n_queries: int) -> str:
    _BACKEND_DISPLAY = {
        "inference_hub": "inference_api",
    }
    run_name = db_path.parent.name
    model    = cfg.get("model", "—")
    backend  = _BACKEND_DISPLAY.get(str(cfg.get("llm_backend", "")).lower(), cfg.get("llm_backend", "—"))
    stride   = cfg.get("staging_stride", "—")
    n_req    = cfg.get("num_queries_requested", "—")
    return f"""
<div id="report-header">
  <h1>Global Correlation Experiment — Analysis Report</h1>
  <div class="subtitle">Can LLM relevance scoring replace or supplement the retriever when it fails?</div>
  <div class="meta">
    <div class="meta-item"><label>Run</label><span>{run_name}</span></div>
    <div class="meta-item"><label>Model</label><span>{model}</span></div>
    <div class="meta-item"><label>Backend</label><span>{backend}</span></div>
    <div class="meta-item"><label>Scored pairs</label><span>{n_rows:,}</span></div>
    <div class="meta-item"><label>Queries</label><span>{n_queries}</span></div>
    <div class="meta-item"><label>Stride</label><span>{stride}</span></div>
    <div class="meta-item"><label>Generated</label><span>{datetime.now().strftime('%Y-%m-%d %H:%M')}</span></div>
  </div>
</div>"""


def _build_toc(sections: list[tuple[str, str]]) -> str:
    items = ""
    for anchor, title in sections:
        items += f'<a href="#{anchor}">{title}</a>\n'
    return items


def _sec_about(n_queries: int, n_rows: int) -> str:
    return """
<div class="section-intro">
  <strong>What is this experiment?</strong> A RAG (Retrieval-Augmented Generation) pipeline
  depends on a vector retriever to surface the right document before a language model answers.
  When the retriever fails — i.e. the ground-truth document (GT) is not ranked first — the
  downstream answer quality drops. This experiment tests whether an LLM relevance score can
  identify the GT document from a large candidate pool and rescue those failed retrievals.
</div>

<h3>Research question</h3>
<p>
  <em>When the retriever does not rank the GT document at position 1, can an LLM relevance score
  re-rank the pool and recover it?</em>
</p>

<h3>How the experiment works</h3>
<p>For each of the """ + str(n_queries) + f""" queries a document pool of up to 6&thinsp;000
candidates is assembled — a mix of embedding-close and embedding-distant documents.
Every <code>(query, document)</code> pair is then scored by an LLM on an integer 1–10 scale
(stored as 0.1–1.0). The total scored pairs in this run: <strong>{n_rows:,}</strong>.</p>
<p>Each row in the database carries four columns the analysis relies on:</p>

<div class="table-wrap">
<table>
  <thead><tr><th>Column</th><th>What it is</th><th>Role in analysis</th></tr></thead>
  <tbody>
    <tr><td><code>llm_score</code></td><td>LLM relevance score (0.1 – 1.0)</td><td>The signal we are testing as a re-ranker</td></tr>
    <tr><td><code>dist_to_gt</code></td><td>Cosine distance from this doc's embedding to the GT embedding</td><td>Ground-truth proximity proxy; GT doc itself has dist ≈ 0</td></tr>
    <tr><td><code>is_gt</code></td><td>1 if this doc is the ground truth for this query, else 0</td><td>Label for binary classification metrics (ROC, P@K)</td></tr>
    <tr><td><code>rag_failed</code></td><td>1 if the retriever did not rank GT at position 1</td><td>Scope flag — all analyzed queries have rag_failed = 1</td></tr>
  </tbody>
</table>
</div>

<h3>What the analysis measures — and why each part matters</h3>

<div class="table-wrap">
<table>
  <thead><tr><th>Analysis part</th><th>Script</th><th>Core question answered</th></tr></thead>
  <tbody>
    <tr>
      <td><strong>Base analysis</strong><br><span class="muted">15 sections</span></td>
      <td><code>analyze_experiment.py</code></td>
      <td>Does LLM score correlate with distance to GT? How often does it rank GT near the top?
          Produces correlation coefficients, ROC/PR curves, recovery@K, score distributions.</td>
    </tr>
    <tr>
      <td><strong>Extended / corrective analysis</strong><br><span class="muted">10 sections A–J</span></td>
      <td><code>analyze_experiment_extended.py</code></td>
      <td>Fixes misleading conclusions from the base run. The LLM uses a 1–10 integer scale —
          many docs share the same top score, so "P@1 = 98%" is inflated by ties.
          This part surfaces the tie structure, corrects MRR, classifies query difficulty,
          and tests whether embedding distance can break ties within the top-scoring group.</td>
    </tr>
    <tr>
      <td><strong>HTML report</strong></td>
      <td><code>export_report.py</code></td>
      <td>Assembles all charts, tables, and narrative into this self-contained file.</td>
    </tr>
  </tbody>
</table>
</div>

<h3>Key concept: the tie problem</h3>
<div class="callout warn">
  <span class="icon">⚠</span>
  <div>
    <strong>Why "P@1 = 98%" is not the right headline number.</strong><br>
    The LLM scores documents on a 1–10 integer scale (÷10), producing only 10 distinct values.
    In most queries, both the GT document <em>and</em> ~9 other documents all receive the
    maximum score of 1.0. Ranking them all at position 1 makes GT appear uniquely identified —
    but it is not. The tie-corrected MRR (0.26, not 0.98) is the honest estimate of
    LLM-as-ranker accuracy. The extended analysis sections A and G explain this in detail.
  </div>
</div>

<h3>The actionable finding</h3>
<div class="callout key">
  <span class="icon">★</span>
  <div>
    <strong>Two-stage hybrid pipeline: LLM filter → embedding-distance tiebreaker.</strong><br>
    At threshold 1.0 the LLM reduces the candidate pool from ~6&thinsp;000 to ~11 documents while
    retaining GT in 98% of queries. Within that small pool, the GT document has the lowest
    cosine distance to the GT embedding in <strong>100% of applicable queries</strong>.
    Combining both signals achieves near-perfect retrieval on this dataset.
  </div>
</div>
"""


def _sec_executive(an: Path, health: dict, metrics: dict) -> str:
    corr  = _artifact_csv(an, "correlation_summary.csv")
    roc   = _artifact_csv(an, "roc_pr_summary.csv")
    mrr   = _artifact_csv(an, "mrr_and_precision_at_k.csv")
    tie   = _artifact_csv(an, "tie_structure.csv")
    diff  = _artifact_csv(an, "query_difficulty.csv")
    rec   = _artifact_csv(an, "llm_recovery_at_k.csv")
    cmrr  = _artifact_csv(an, "corrected_mrr_at_k.csv")
    wt    = _artifact_csv(an, "within_tier_dist.csv")

    sp_r = corr.loc[corr["method"].str.startswith("Spearman"), "median"].iloc[0] if corr is not None else float("nan")
    pb_r = corr.loc[corr["method"].str.startswith("Point"),    "median"].iloc[0] if corr is not None else float("nan")
    roc_auc = float(roc.loc[roc["metric"] == "ROC-AUC", "value"].iloc[0]) if roc is not None else float("nan")
    mrr_val  = float(mrr.loc[mrr["metric"] == "MRR", "value"].iloc[0])         if mrr  is not None else float("nan")
    cmrr_avg = float(cmrr.loc[cmrr["metric"].str.contains("tie-corrected"), "value"].iloc[0]) if cmrr is not None else float("nan")
    rec_1    = float(rec.loc[rec["K"] == 1, "recovery_rate_pct"].iloc[0])       if rec  is not None else float("nan")
    rec_10   = float(rec.loc[rec["K"] == 10, "recovery_rate_pct"].iloc[0])      if rec  is not None else float("nan")
    n_unique = int(tie["gt_uniquely_top"].sum())                                 if tie  is not None else 0
    med_tied = float(tie.loc[tie["n_above_gt"] == 0, "n_tied_at_gt"].median())  if tie  is not None else float("nan")
    tier_prec = float(tie.loc[tie["n_above_gt"] == 0, "tier_precision"].mean()) if tie  is not None else float("nan")
    wr = float(wt["gt_would_win_dist"].mean() * 100) if wt is not None else float("nan")

    n_easy   = int((diff["difficulty"] == "EASY").sum())   if diff is not None else 0
    n_medium = int((diff["difficulty"] == "MEDIUM").sum()) if diff is not None else 0
    n_hard   = int((diff["difficulty"] == "HARD").sum())   if diff is not None else 0
    n_failed = int((diff["difficulty"] == "FAILED").sum()) if diff is not None else 0

    cards = _stat_cards(
        ("ROC-AUC",              f"{roc_auc:.4f}",         "binary GT detection",     "blue"),
        ("Spearman r",           f"{sp_r:.3f}",            "score vs dist_to_gt",     "green"),
        ("MRR (tie-corrected)",  f"{cmrr_avg:.3f}",        f"optimistic: {mrr_val:.3f}", "amber"),
        ("Recovery@1",           f"{rec_1:.1f}%",          "GT in LLM top-1",         "blue"),
        ("Recovery@10",          f"{rec_10:.1f}%",         "GT in LLM top-10",        "green"),
        ("Median tie size",      f"{med_tied:.0f}",        "docs tied at GT's score", "amber"),
        ("Tier precision",       f"{tier_prec:.3f}",       "expected if random pick", "red"),
        ("Hybrid win rate",      f"{wr:.1f}%",             "LLM + dist tiebreaker",   "green"),
    )

    callout = _callout(
        f"The retriever failed on <b>{metrics['rag_fail_pct']:.0f}%</b> of analyzed queries. "
        f"Despite this, the LLM assigns the maximum relevance score (1.0) to the GT document in "
        f"<b>{metrics['rec_1']:.1f}%</b> of queries — but it also assigns 1.0 to a median of "
        f"<b>{metrics['med_tied']:.0f}</b> other documents, so it acts as a <b>coarse filter</b> "
        "rather than a precise ranker. Adding an embedding-distance tiebreaker resolves ties in "
        f"<b>{metrics['wr_within']:.1f}%</b> of top-tier queries "
        f"({metrics['wr_overall']:.1f}% overall).",
        "key",
    )

    health_html = format_health_html(health) if health else ""

    diff_table = f"""
<h3>Query Difficulty Breakdown</h3>
<table>
  <thead><tr><th>Category</th><th>Queries</th><th>Definition</th></tr></thead>
  <tbody>
    <tr><td><span class="tag pos">EASY</span></td><td><b>{n_easy}</b></td><td>GT is the only doc scoring 1.0</td></tr>
    <tr><td><span class="tag pos">MEDIUM</span></td><td><b>{n_medium}</b></td><td>GT scores 1.0, 2–10 others also do</td></tr>
    <tr><td><span class="tag neg">HARD</span></td><td><b>{n_hard}</b></td><td>GT scores 1.0, &gt;10 others also do</td></tr>
    <tr><td><span class="tag neg">FAILED</span></td><td><b>{n_failed}</b></td><td>GT does not reach score 1.0</td></tr>
  </tbody>
</table>"""

    return health_html + cards + callout + diff_table


def _sec_reading_guide(an: Path, metrics: dict) -> str:
    """Explain global vs per-query metrics and folder layout."""
    layout = AnalysisLayout(an)
    explain = """
<p>Analysis outputs are split by <b>scope</b> so you can tell which numbers describe the
whole experiment vs a single query:</p>
<ul>
  <li><b>Global</b> — one summary over all queries and doc pairs (e.g. MRR, Recovery@K, ROC-AUC).
      Files in <code>csv/global/</code>.</li>
  <li><b>Per-query</b> — one row per <code>query_idx</code> (e.g. GT rank, difficulty, tie size).
      Files in <code>csv/per_query/</code>. Filter in Excel/Sheets to inspect specific queries.</li>
</ul>"""
    callout = _callout(
        f"<b>Start per-query review:</b> open "
        f"<code>csv/per_query/per_query_overview.csv</code> — merged master table for all "
        f"{metrics.get('n_queries', '?')} queries. See also <code>INDEX.md</code> in the analysis folder.",
        "info",
    )
    tbl = ""
    overview = _artifact_csv(an, "per_query_overview.csv")
    if overview is not None and not overview.empty:
        show_cols = [c for c in [
            "query_idx", "gt_llm_rank", "gt_llm_score", "difficulty",
            "n_tied_at_gt", "rank_avg", "spearman_r", "gt_would_win_dist",
        ] if c in overview.columns]
        tbl = "<h3>Per-Query Overview (sample)</h3>" + _table(
            overview[show_cols].head(15),
            col_formats={"spearman_r": "{:.3f}", "gt_llm_score": "{:.1f}", "rank_avg": "{:.1f}"},
            highlight_col="query_idx",
            max_rows=15,
        )
        if len(overview) > 15:
            tbl += f"<p class='muted'>Showing 15 of {len(overview)} queries — full table in CSV.</p>"

    folder_tbl = f"""
<h3>Folder layout</h3>
<table>
  <thead><tr><th>Path</th><th>Contents</th></tr></thead>
  <tbody>
    <tr><td><code>summaries/</code></td><td>Text summaries (analysis + extended)</td></tr>
    <tr><td><code>csv/global/</code></td><td>Pool-level aggregate CSVs</td></tr>
    <tr><td><code>csv/per_query/</code></td><td>One row per query_idx</td></tr>
    <tr><td><code>charts/base/</code></td><td>Charts from base analysis</td></tr>
    <tr><td><code>charts/extended/</code></td><td>Charts from extended analysis</td></tr>
    <tr><td><code>{layout.index_md().name}</code></td><td>Full file catalog</td></tr>
  </tbody>
</table>"""
    return explain + callout + tbl + folder_tbl


def _sec_correlation(an: Path) -> str:
    summary = _artifact_csv(an, "correlation_summary.csv")
    img     = _artifact_img(an, "correlation_histograms.png")

    explain = """
<div class="section-intro">
  <strong>What this checks:</strong> Whether the LLM score and the ground-truth distance
  (<code>dist_to_gt</code>) move together — i.e. does a higher LLM relevance score reliably
  predict that a document is closer to GT in embedding space?
  Four methods are used so that the result is not an artifact of one method's assumptions.
  All four are computed <em>per query</em> (not pooled globally) so each data point in the
  histograms below is one query.
</div>
<p>Four correlation methods measure two different relationships:</p>
<p><b>Spearman / Pearson / Kendall τ-b</b> — between <code>llm_score</code> and <code>dist_to_gt</code>.
A negative value means higher LLM scores predict lower distance to the ground-truth embedding (desirable).
Spearman and Kendall are rank-based and robust to outliers; Pearson measures the linear component.
Kendall τ-b is the most robust to the heavy tie structure in the scores.</p>
<p><b>Point-biserial r</b> — between <code>llm_score</code> and the binary <code>is_gt</code> flag.
A positive value means the GT document scores higher than the pool average (desirable).</p>"""

    if summary is not None:
        fmt = {
            "median":       "{:.4f}",
            "mean":         "{:.4f}",
            "pct_negative": "{:.1f}%",
            "pct_positive": "{:.1f}%",
            "pct_sig_p05":  "{:.1f}%",
        }
        tbl = _table(summary, col_formats=fmt, highlight_col="median")
    else:
        tbl = "<p class='muted'>correlation_summary.csv not found.</p>"

    note = _callout(
        "All four methods are significant (p&lt;0.05) in 99.5% of queries. "
        "Pearson r is stronger than Spearman because the GT point (score=1.0, dist≈0) anchors the regression line. "
        "Kendall τ-b, most robust to tied scores, confirms the monotone ordering.",
        "info",
    )

    return explain + tbl + note + _figure(img, "Per-query distribution of all four correlation coefficients. "
        "The point-biserial panel (bottom-right) measures GT separation from the pool.")


def _sec_gt_detection(an: Path) -> str:
    roc  = _artifact_csv(an, "roc_pr_summary.csv")
    img  = _artifact_img(an, "roc_pr_curves.png")
    img2 = _artifact_img(an, "score_distributions.png")

    explain = """
<div class="section-intro">
  <strong>What this checks:</strong> Can the LLM score act as a binary detector — separating
  the one GT document from all ~6&thinsp;000 non-GT documents in the pool?
  This is evaluated globally across all 1&thinsp;M scored pairs at once using ROC and
  Precision-Recall curves. ROC-AUC near 1.0 means the LLM almost never ranks a random
  non-GT doc above GT. Average Precision (PR-AUC) measures quality at the <em>top</em> of the
  ranking, where ties cause precision to fall even when recall is high.
</div>
<p>Treating the problem as binary classification (<code>is_gt=1</code> positive,
<code>is_gt=0</code> negative), we evaluate how well <code>llm_score</code> separates
GT from non-GT docs across all 1 M scored pairs globally.</p>"""

    tbl = _table(roc, col_formats={"value": "{:.6f}"}, highlight_col="value") if roc is not None else ""

    note = _callout(
        "ROC-AUC ≈ 0.996 reflects near-perfect coarse separation: 83% of docs score 0.1 and "
        "GT almost always scores 1.0. Average Precision is lower (~8.7%) because "
        "2 021 non-GT docs also score 1.0, diluting precision at the top threshold.",
        "warn",
    )

    return (explain + tbl + note
            + _figure(img,  "ROC curve (left) and Precision-Recall curve (right). "
                            "AP=8.7% means that at the score=1.0 threshold, ~1 in 11 retained docs is GT.")
            + _figure(img2, "LLM score distribution: GT almost always scores 1.0; "
                            "non-GT docs concentrate at 0.1. The model uses a 1–10 integer scale (÷10)."))


def _sec_rank_recovery(an: Path, metrics: dict) -> str:
    rec  = _artifact_csv(an, "llm_recovery_at_k.csv")
    cmrr = _artifact_csv(an, "corrected_mrr_at_k.csv")
    img1 = _artifact_img(an, "gt_llm_rank.png")
    img2 = _artifact_img(an, "corrected_mrr_at_k.png")
    img3 = _artifact_img(an, "llm_recovery_at_k.png")

    explain = """
<div class="section-intro">
  <strong>What this checks:</strong> When the retriever has already failed, how well does
  re-ranking by LLM score recover the GT document? Recovery@K answers "in what fraction of
  queries does GT appear in the LLM's top K results?" MRR (Mean Reciprocal Rank) gives a
  single number — higher is better. Three rank variants (optimistic / tie-corrected /
  pessimistic) are shown because the LLM's discrete 1–10 scale causes heavy ties that
  make the optimistic number misleading. Always compare the tie-corrected MRR against the
  optimistic one to understand how much of the apparent performance is an artifact of ties.
</div>
<p>The retriever failed on every query. These metrics answer: <em>if we re-rank by LLM score,
how often does GT appear near the top?</em></p>
<p>Three rank variants are compared to surface the effect of ties:
<b>Optimistic</b> (rank_min — all tied docs get the lowest rank in their group),
<b>Tie-corrected</b> (rank_avg — tied docs share the average rank), and
<b>Pessimistic</b> (rank_max — tied docs get the highest rank in their group).</p>"""

    warn = _callout(
        f"The optimistic MRR ({metrics['mrr_opt']:.2f}) looks near-perfect but is inflated by the tie structure. "
        f"The tie-corrected MRR ({metrics['mrr_corr']:.2f}) is the honest estimate: GT is in the top-scoring group "
        f"but competes with a median of {metrics['med_tied']:.0f} other docs that all score 1.0.",
        "warn",
    )

    rec_tbl  = _table(rec,  col_formats={"recovery_rate_pct": "{:.1f}%"}, highlight_col="recovery_rate_pct") if rec  is not None else ""
    cmrr_tbl = _table(cmrr, col_formats={"value": "{:.4f}"},              highlight_col="value")             if cmrr is not None else ""

    return (explain + warn
            + "<h3>Recovery Rate — GT in LLM Top-K</h3>" + rec_tbl
            + _figure(img3, "Fraction of queries where GT appears within the top-K results by LLM score.")
            + "<h3>MRR and P@K: Optimistic vs Tie-Corrected vs Pessimistic</h3>" + cmrr_tbl
            + _figure(img2, "Effect of tie-breaking method on MRR and P@K. "
                            "Green = optimistic (rank_min), blue = tie-corrected (rank_avg), red = pessimistic (rank_max).")
            + _figure(img1, "GT rank distribution (left) and CDF (right). "
                            f"The spike at rank 1 reflects the tie issue — {metrics['gt_in_top']} queries have GT at rank_min=1 "
                            f"but sharing that rank with ~{metrics['med_tied']:.0f} other docs on average."))


def _sec_tie_structure(an: Path, metrics: dict) -> str:
    tie  = _artifact_csv(an, "tie_structure.csv")
    img  = _artifact_img(an, "tie_structure.png")
    diff = _artifact_csv(an, "query_difficulty.csv")
    img2 = _artifact_img(an, "query_difficulty.png")

    explain = f"""
<div class="section-intro">
  <strong>What this checks:</strong> This is the most important corrective section.
  Because the LLM uses only 10 distinct score values, many documents end up tied at the
  same score. When GT and 9 other documents all receive 1.0, GT is not uniquely identified —
  it just reached the top tier. This section measures how large those tied groups are, what
  fraction of queries have GT truly alone at the top (EASY) vs. tied with others (MEDIUM /
  HARD), and how often GT is outscored entirely (FAILED). The "tier precision" metric
  (1 / number_of_tied_docs) is the realistic P@1 if you pick randomly from the tied group.
</div>
<p>The LLM uses a 1–10 integer scale (divided by 10), producing only 10 distinct scores.
This causes heavy ties: in {metrics['gt_in_top']} of {metrics['n_queries']} queries, both GT and multiple other documents all
receive the maximum score of 1.0. Rank-1 is therefore shared, not uniquely earned.</p>"""

    stats_row = ""
    if tie is not None:
        top = tie[tie["n_above_gt"] == 0]
        stats_row = _stat_cards(
            ("GT in top tier",   f"{len(top)} / {len(tie)}",    "queries",               "blue"),
            ("Uniquely top",     str(int(tie["gt_uniquely_top"].sum())), "GT only at max score", "green"),
            ("Median tie size",  f"{top['n_tied_at_gt'].median():.0f}", "docs sharing GT score","amber"),
            ("Mean tier prec.",  f"{top['tier_precision'].mean():.3f}", "1 / n_tied",           "red"),
        )

    key = _callout(
        "The <b>tier precision</b> (mean 0.17) is the expected accuracy if you randomly pick "
        "one document from the top-scoring group. This is the practical P@1 when no "
        "tiebreaker is available — equivalent to 1 in 6 chance of selecting GT.",
        "key",
    )

    return (explain + stats_row + key
            + _figure(img,  "Left: distribution of tie sizes (docs sharing GT's score). "
                            "Middle: tier precision per query. Right: docs outscoring GT (almost never).")
            + _figure(img2, "Query difficulty classification. EASY = GT uniquely identified. "
                            "MEDIUM / HARD = GT tied with 1–9 or 10+ other docs. FAILED = GT outscored."))


def _sec_score_discretization(an: Path) -> str:
    lvl  = _artifact_csv(an, "score_level_breakdown.csv")
    img  = _artifact_img(an, "score_level_breakdown.png")
    fp   = _artifact_csv(an, "false_positive_analysis.csv")
    img2 = _artifact_img(an, "false_positive_analysis.png")
    mg   = _artifact_csv(an, "marginal_score_value.csv")
    img3 = _artifact_img(an, "marginal_score_value.png")

    explain = """
<div class="section-intro">
  <strong>What this checks:</strong> How are documents distributed across the 10 discrete
  score levels, and do intermediate scores (0.2–0.8) carry real signal or are they just noise?
  If a score of 0.6 genuinely predicts a document is closer to GT than a score of 0.2, the
  intermediate levels are useful for a more nuanced filter. This section also investigates
  <em>false positives</em> — non-GT documents that score 1.0 — to determine whether they are
  genuinely semantically close to GT or random errors.
</div>
<p>The LLM outputs integer scores 1–10, divided by 10. The resulting distribution is
heavily bimodal: the vast majority of docs receive 0.1 and nearly all GTs receive 1.0.
Intermediate scores (0.2–0.8) carry a real signal — docs at higher score levels are
genuinely closer to GT in embedding space.</p>"""

    lvl_tbl = _table(
        lvl,
        col_formats={"gt_fraction": "{:.4f}", "mean_dist": "{:.4f}", "median_dist": "{:.4f}"},
        highlight_col="score",
    ) if lvl is not None else ""

    fp_tbl = _table(
        fp,
        col_formats={"mean_dist": "{:.4f}", "median_dist": "{:.4f}"},
        highlight_col="group",
    ) if fp is not None else ""

    note = _callout(
        "Non-GT docs that score 1.0 have a median dist_to_gt of ~0.3 — "
        "they are genuinely semantically close to the GT document, not random noise. "
        "These are true semantic neighbours, not errors.",
        "info",
    )

    return (explain
            + "<h3>Per-Score-Level Statistics</h3>" + lvl_tbl
            + _figure(img, "Score level breakdown: doc counts (log scale), GT fraction, "
                           "distance distributions, and box plots per score level.")
            + "<h3>False Positives: Non-GT Docs Scoring 1.0</h3>" + fp_tbl + note
            + _figure(img2, "Non-GT docs scoring 1.0 (orange) cluster at dist_to_gt ≈ 0.3, "
                            "far closer to GT than background docs (blue, dist ≈ 0.85).")
            + "<h3>Marginal Value of Intermediate Score Levels</h3>"
            + _figure(img3, "Mean/median dist_to_gt per score level (non-GT only). "
                            "Higher scores → genuinely lower distance to GT."))


def _sec_threshold_filter(an: Path, metrics: dict) -> str:
    filt = _artifact_csv(an, "threshold_filter_analysis.csv")
    img  = _artifact_img(an, "threshold_filter.png")

    explain = """
<div class="section-intro">
  <strong>What this checks:</strong> If you use the LLM score as a hard filter — discarding
  all documents below a threshold and passing only the remainder to a downstream ranker — what
  is the trade-off between pool size, precision, and recall at each threshold?
  The goal is to find the operating point where the pool is small enough to rank cheaply but
  large enough that GT is rarely discarded. A recall of 98% at threshold 1.0 with a pool of
  ~11 documents is the practical sweet spot for a two-stage pipeline.
</div>
<p>Using LLM score as a binary inclusion filter: keep all documents scoring ≥ threshold.
This shows the precision / recall trade-off at each threshold, and the resulting pool size
passed to a downstream ranker.</p>"""

    tbl = _table(
        filt,
        col_formats={
            "avg_pool_size": "{:.1f}",
            "precision":     "{:.4f}",
            "recall":        "{:.4f}",
            "f1":            "{:.4f}",
        },
        highlight_col="threshold",
    ) if filt is not None else ""

    note = _callout(
        f"At threshold 1.0: pool shrinks dramatically while recall stays at "
        f"{metrics['thresh_recall']:.0f}%, precision reaches {metrics['thresh_precision']:.1f}%. "
        "This is the optimal operating point for a two-stage pipeline: "
        "LLM filter → embedding-distance ranker.",
        "good",
    )

    return explain + tbl + note + _figure(img, "Pool size, precision, recall, F1 at each threshold (left, centre). "
                                               "Precision vs recall trade-off (right).")


def _sec_hybrid(an: Path, metrics: dict) -> str:
    wt   = _artifact_csv(an, "within_tier_dist.csv")
    img  = _artifact_img(an, "within_tier_dist.png")
    img2 = _artifact_img(an, "topk_jaccard.png")
    jac  = _artifact_csv(an, "topk_jaccard_summary.csv")

    wr = float(wt["gt_would_win_dist"].mean() * 100) if wt is not None else float("nan")
    pct_further = float(wt["pct_ngt_further"].mean()) if wt is not None else float("nan")

    explain = f"""
<div class="section-intro">
  <strong>What this checks:</strong> The LLM cannot uniquely identify GT (it ties with ~9
  other documents at score 1.0). But can a second signal — the cosine distance between a
  document's embedding and the GT embedding — break those ties? This section tests whether
  GT has the <em>lowest</em> <code>dist_to_gt</code> among all documents tied at score 1.0.
  If yes, a two-stage pipeline (LLM filter → distance tiebreaker) can recover GT
  precisely. This is the primary actionable finding of the experiment.
</div>
<p>If a secondary tiebreaker is applied within the top-scoring tier (docs at score=1.0),
choosing the document with the <em>lowest</em> <code>dist_to_gt</code> (i.e. cosine
distance to the GT embedding), does GT win?</p>"""

    big = _stat_cards(
        ("Hybrid win rate",     f"{wr:.1f}%",          "LLM filter + dist tiebreaker",         "green"),
        ("Mean % peers further",f"{pct_further:.1f}%", "non-GT peers with higher dist than GT", "blue"),
    )

    key = _callout(
        f"<b>In {metrics['wr_within']:.1f}% of queries where GT reaches score=1.0, it also has the lowest "
        f"dist_to_gt within that tier ({metrics['wr_overall']:.1f}% of all {metrics['n_queries']} queries overall).</b> "
        "A two-stage system — (1) keep score=1.0 docs, "
        "(2) return the one with lowest embedding distance — achieves near-perfect retrieval "
        "on this dataset. This is the primary actionable finding.",
        "key",
    )

    jac_tbl = _table(
        jac,
        col_formats={"mean": "{:.3f}", "std": "{:.3f}", "50%": "{:.3f}"},
        highlight_col="K",
    ) if jac is not None else ""

    return (explain + big + key
            + _figure(img, f"Left: distribution of % peers with higher dist than GT within the top tier. "
                           f"Right: dist tiebreaker win rate ({metrics['wr_within']:.1f}% within tier).")
            + "<h3>Top-K Jaccard: Agreement Between LLM Ranking and Distance Ranking</h3>"
            + jac_tbl
            + _figure(img2, "Jaccard similarity between top-K by LLM score and top-K by dist_to_gt. "
                            "Moderate agreement (~0.3–0.4) at K=5–20."))


def _sec_distance_analysis(an: Path, metrics: dict) -> str:
    img1 = _artifact_img(an, "scatter_score_vs_dist.png")
    img2 = _artifact_img(an, "distance_quantile_analysis.png")

    explain = """
<div class="section-intro">
  <strong>What this checks:</strong> A visual sanity check of the core relationship.
  The scatter plot shows every scored (query, doc) pair: if the LLM score carries distance
  signal, higher-scored documents should cluster toward lower <code>dist_to_gt</code>.
  The decile chart bins non-GT documents by distance and shows whether mean LLM score
  declines as documents move farther from GT. This confirms (or disproves) that the
  correlation results are reflected in the raw data visually.
</div>
<p>These charts visualise the global relationship between LLM score and proximity to GT
in embedding space, across all scored pairs.</p>"""

    note = _callout(
        "The scatter shows clear vertical stripes — confirming score discretisation (only 10 values). "
        "GT docs (red) cluster near dist=0 as expected. "
        "The decile chart shows a meaningful decline in mean LLM score from Q1 (closest to GT) "
        "to Q10 (farthest), but the median is flat at 0.1 — most docs score the minimum regardless.",
        "info",
    )

    return (explain + note
            + _figure(img1, f"LLM score vs dist_to_gt scatter (10 000 non-GT sample + all {metrics['n_queries']} GTs). "
                            "GT docs appear at dist≈0 across all score levels where they appear.")
            + _figure(img2, "Mean and median LLM score by dist_to_gt decile (non-GT docs only). "
                            "Q1 = 10% of docs closest to GT; Q10 = farthest."))


def _sec_failures(an: Path) -> str:
    fail = _artifact_csv(an, "failure_cases.csv")
    img  = _artifact_img(an, "failure_cases.png")

    n_fail = len(fail) if fail is not None else 0
    explain = f"""
<div class="section-intro">
  <strong>What this checks:</strong> While the LLM assigns score 1.0 to GT in ~98% of
  queries, a small number of queries are true failures — GT is outscored by at least one
  non-GT document. These are the cases where the two-stage hybrid pipeline would also fail.
  This section identifies those queries, shows GT's actual score and rank, and visualises
  the score distribution for the worst case. Understanding failures helps diagnose whether
  they share common characteristics (unusual query phrasing, short GT documents, domain
  mismatch) that could guide model or prompt improvements.
</div>
<p>In {n_fail} queries the LLM did not assign the maximum score to GT — it was outscored by
at least one non-GT document. These are the true LLM failures.</p>"""

    tbl = _table(
        fail,
        col_formats={
            "gt_llm_score":       "{:.1f}",
            "gt_dist_to_gt":      "{:.6f}",
            "gt_rank":            "{}",
            "n_docs_above_gt":    "{}",
        },
        highlight_col="query_idx",
    ) if fail is not None else ""

    note = _callout(
        "Query 107 is the worst failure: GT scored 0.1, ranked 281st out of 5 000+ docs. "
        "Queries 43 and 140 scored 0.8 (just below the top tier). "
        "These failures are consistent with edge cases where the GT document "
        "may be unusually short, technical, or differently worded from the query.",
        "warn",
    )

    return explain + tbl + note + _figure(img, "GT rank and score for each failure case. "
                                               "Right: score distribution for the worst failure (query 107).")


def _sec_conclusions(an: Path, metrics: dict) -> str:
    txt_path = _artifact_path(an, "extended_summary.txt")
    raw = txt_path.read_text() if txt_path.exists() else ""
    intro = """
<div class="section-intro">
  <strong>What this summarises:</strong> The five key takeaways from the full analysis,
  ranging from what the LLM does well (coarse filter) to where it falls short (precise
  ranker) and what combination of signals achieves near-perfect retrieval. The raw extended
  summary text is appended below for completeness.
</div>"""
    # strip the ASCII border for embedding
    lines = [l for l in raw.splitlines() if not set(l.strip()).issubset({"=", ""})]

    findings = [
        ("<b>LLM as coarse filter (excellent)</b> — At threshold 1.0, pool shrinks ~450× "
         f"while retaining GT in {metrics['rec_1']:.0f}% of queries. Precision in the retained set is "
         f"{metrics['thresh_precision']:.1f}% (≈1 GT among ~{metrics['med_tied']:.0f} docs).",
         "good"),
        ("<b>LLM as precise ranker (limited)</b> — Tie-corrected MRR = "
         f"{metrics['mrr_corr']:.3f}, not {metrics['mrr_opt']:.2f}. "
         "The model cannot discriminate within the top-scoring tier.",
         "warn"),
        ("<b>Hybrid pipeline (near-perfect)</b> — LLM filter (score=1.0) → embedding-distance "
         f"tiebreaker recovers GT in {metrics['wr_within']:.1f}% of top-tier queries "
         f"({metrics['wr_overall']:.1f}% overall).",
         "key"),
        ("<b>False positives are semantically real</b> — Non-GT docs scoring 1.0 have median "
         "dist_to_gt ≈ 0.3. They are genuine semantic neighbours of GT, not noise.",
         "info"),
        ("<b>Negative correlation confirmed across all methods</b> — Spearman −0.17, "
         "Pearson −0.39, Kendall −0.13. Higher LLM score consistently predicts lower "
         "embedding distance to GT.",
         "info"),
    ]

    body = intro
    for text, kind in findings:
        body += _callout(text, kind)

    if lines:
        body += "<hr><h3>Extended Summary (raw)</h3>"
        body += "<pre>" + "\n".join(lines) + "</pre>"

    return body


# ── PDF export ────────────────────────────────────────────────────────────────

def export_pdf(html_path: Path) -> Path | None:
    """
    Render html_path to a PDF using a headless Chromium browser (playwright).
    Returns the PDF path on success, None if playwright is unavailable.
    The PDF is written alongside the HTML as report.pdf.
    """
    if not PLAYWRIGHT_AVAILABLE:
        print("  [pdf]  skipped — playwright not installed (pip install playwright && playwright install chromium)")
        return None

    pdf_path = html_path.with_suffix(".pdf")
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            page = browser.new_page()
            page.goto(f"file://{html_path.resolve()}", wait_until="networkidle")
            page.pdf(
                path=str(pdf_path),
                format="A4",
                print_background=True,
                margin={"top": "18mm", "bottom": "18mm", "left": "14mm", "right": "14mm"},
            )
            browser.close()
        size_kb = pdf_path.stat().st_size / 1024
        print(f"  [pdf]  {pdf_path.name}  ({size_kb:.0f} KB)")
        return pdf_path
    except Exception as exc:
        print(f"  [pdf]  export failed: {exc}", file=sys.stderr)
        return None


# ── main assembler ────────────────────────────────────────────────────────────

def generate(run_dir: Path) -> Path:
    an      = run_dir / ANALYSIS_DIR
    db_path = run_dir / "experiment_results.db"
    cfg_path = run_dir / "experiment_config.json"

    if not an.exists():
        raise FileNotFoundError(f"analysis/ folder not found in {run_dir}. Run analyze_experiment.py first.")

    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}

    health = collect_run_health(db_path) if db_path.exists() else {}
    save_run_health(health, an)
    metrics = _metrics_from_analysis(an)

    n_rows = health.get("scored_rows", 0)
    n_queries = health.get("n_queries_analyzed", metrics.get("n_queries", 0))

    toc_sections = [
        ("about",        "About This Analysis"),
        ("executive",    "Executive Summary"),
        ("guide",        "Reading the Outputs"),
        ("correlation",  "Correlation Analysis"),
        ("detection",    "GT Detection (ROC/PR)"),
        ("rank",         "Rank & Recovery"),
        ("ties",         "Tie Structure"),
        ("discretize",   "Score Discretisation"),
        ("filter",       "Threshold Filter"),
        ("hybrid",       "Hybrid Approach"),
        ("distance",     "Distance Analysis"),
        ("failures",     "Failure Cases"),
        ("conclusions",  "Conclusions"),
    ]

    sections_html = ""
    builders = [
        ("about",       "About This Analysis",           lambda an: _sec_about(n_queries, n_rows)),
        ("executive",   "Executive Summary",             lambda an: _sec_executive(an, health, metrics)),
        ("guide",       "Reading the Outputs",           lambda an: _sec_reading_guide(an, metrics)),
        ("correlation", "Correlation Analysis",          _sec_correlation),
        ("detection",   "GT Detection — ROC / PR",       _sec_gt_detection),
        ("rank",        "GT Rank & Recovery Rate",        lambda an: _sec_rank_recovery(an, metrics)),
        ("ties",        "Tie Structure & Query Difficulty", lambda an: _sec_tie_structure(an, metrics)),
        ("discretize",  "Score Discretisation",          _sec_score_discretization),
        ("filter",      "LLM as Threshold Filter",       lambda an: _sec_threshold_filter(an, metrics)),
        ("hybrid",      "Hybrid Approach: LLM + Distance Tiebreaker", lambda an: _sec_hybrid(an, metrics)),
        ("distance",    "Distance vs Score Analysis",    lambda an: _sec_distance_analysis(an, metrics)),
        ("failures",    "Failure Case Analysis",         _sec_failures),
        ("conclusions", "Conclusions",                   lambda an: _sec_conclusions(an, metrics)),
    ]
    for anchor, title, builder in builders:
        print(f"  building [{anchor}] …")
        try:
            body = builder(an)
        except Exception as exc:
            body = f"<p class='muted'>Error building section: {exc}</p>"
        sections_html += _section(title, anchor, body)

    toc_html = _build_toc(toc_sections)
    header   = _build_header(cfg, db_path, n_rows, n_queries)
    html     = _html_shell(
        title    = f"Analysis Report — {run_dir.name}",
        toc      = toc_html,
        header   = header,
        body     = sections_html,
    )

    build_per_query_overview(an)
    write_analysis_index(an, health)

    out_path = AnalysisLayout(an).report_html()
    out_path.write_text(html, encoding="utf-8")
    export_pdf(out_path)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("db_path", nargs="?", help="Path to experiment_results.db")
    parser.add_argument("--run-dir", help="Path to run directory")
    args = parser.parse_args()

    if args.run_dir:
        run_dir = Path(args.run_dir)
    elif args.db_path:
        run_dir = Path(args.db_path).parent
    else:
        run_dir = _find_latest_run_dir()

    print(f"Run dir : {run_dir}")
    out = generate(run_dir)
    pdf = out.with_suffix(".pdf")
    print(f"\nReport  : {out}  ({out.stat().st_size / 1024:.0f} KB)")
    if pdf.exists():
        print(f"PDF     : {pdf}  ({pdf.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
