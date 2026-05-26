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
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

RESULTS_ROOT = Path("results/global_exp")
ANALYSIS_DIR = "analysis"


# ── helpers ───────────────────────────────────────────────────────────────────

def _find_latest_run_dir() -> Path:
    candidates = sorted(RESULTS_ROOT.glob("staged_run_*"))
    if not candidates:
        candidates = sorted(p.parent for p in RESULTS_ROOT.glob("*/experiment_results.db"))
    if not candidates:
        raise FileNotFoundError(f"No run directories found under {RESULTS_ROOT}")
    return candidates[-1]


def _img_b64(path: Path) -> str | None:
    if not path.exists():
        return None
    data = base64.b64encode(path.read_bytes()).decode()
    return f"data:image/png;base64,{data}"


def _load_csv(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() else None


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
    return f"""
<table>
  <thead><tr>{header}</tr></thead>
  <tbody>{"".join(rows_html)}</tbody>
</table>"""


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
table { width: 100%; border-collapse: collapse; font-size: 13px; margin: 12px 0; }
thead tr { background: #f1f5f9; }
th { text-align: left; padding: 8px 12px; font-weight: 600; color: #475569;
     border-bottom: 2px solid #e2e8f0; font-size: 12px; white-space: nowrap; }
td { padding: 7px 12px; border-bottom: 1px solid #f1f5f9; color: #374151; }
tr:last-child td { border-bottom: none; }
tr:hover td { background: #f8fafc; }
td.hl { font-weight: 600; color: #1d4ed8; }

/* figures */
figure { margin: 20px 0; }
figure img { display: block; border: 1px solid #e2e8f0; }
figcaption { font-size: 11px; color: #64748b; margin-top: 6px; font-style: italic; }

/* two-col */
.two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; margin: 16px 0; }

/* callouts */
.callout { display: flex; align-items: flex-start; gap: 10px; border-radius: 7px;
           padding: 12px 16px; margin: 14px 0; font-size: 13px; line-height: 1.5; }
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
    run_name = db_path.parent.name
    model    = cfg.get("model", "—")
    backend  = cfg.get("llm_backend", "—")
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


def _sec_executive(an: Path) -> str:
    corr  = _load_csv(an / "correlation_summary.csv")
    roc   = _load_csv(an / "roc_pr_summary.csv")
    mrr   = _load_csv(an / "mrr_and_precision_at_k.csv")
    tie   = _load_csv(an / "tie_structure.csv")
    diff  = _load_csv(an / "query_difficulty.csv")
    rec   = _load_csv(an / "llm_recovery_at_k.csv")
    cmrr  = _load_csv(an / "corrected_mrr_at_k.csv")
    wt    = _load_csv(an / "within_tier_dist.csv")

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
        "The retriever failed on <b>100%</b> of queries. Despite this, the LLM assigns the "
        f"maximum relevance score (1.0) to the GT document in <b>{rec_1:.1f}%</b> of queries — "
        f"but it also assigns 1.0 to a median of <b>{med_tied:.0f}</b> other documents, so it acts "
        "as a <b>coarse filter</b> rather than a precise ranker. Adding an embedding-distance "
        f"tiebreaker resolves ties perfectly in <b>{wr:.1f}%</b> of cases.",
        "key",
    )

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

    return cards + callout + diff_table


def _sec_correlation(an: Path) -> str:
    summary = _load_csv(an / "correlation_summary.csv")
    img     = _img_b64(an / "correlation_histograms.png")

    explain = """
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
    roc  = _load_csv(an / "roc_pr_summary.csv")
    img  = _img_b64(an / "roc_pr_curves.png")
    img2 = _img_b64(an / "score_distributions.png")

    explain = """
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


def _sec_rank_recovery(an: Path) -> str:
    rec  = _load_csv(an / "llm_recovery_at_k.csv")
    cmrr = _load_csv(an / "corrected_mrr_at_k.csv")
    img1 = _img_b64(an / "gt_llm_rank.png")
    img2 = _img_b64(an / "corrected_mrr_at_k.png")
    img3 = _img_b64(an / "llm_recovery_at_k.png")

    explain = """
<p>The retriever failed on every query. These metrics answer: <em>if we re-rank by LLM score,
how often does GT appear near the top?</em></p>
<p>Three rank variants are compared to surface the effect of ties:
<b>Optimistic</b> (rank_min — all tied docs get the lowest rank in their group),
<b>Tie-corrected</b> (rank_avg — tied docs share the average rank), and
<b>Pessimistic</b> (rank_max — tied docs get the highest rank in their group).</p>"""

    warn = _callout(
        "The optimistic MRR (0.98) looks near-perfect but is inflated by the tie structure. "
        "The tie-corrected MRR (0.26) is the honest estimate: GT is in the top-scoring group "
        "but competes with a median of 9 other docs that all score 1.0.",
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
                            "The spike at rank 1 reflects the tie issue — 195 queries have GT at rank_min=1 "
                            "but sharing that rank with ~9 other docs."))


def _sec_tie_structure(an: Path) -> str:
    tie  = _load_csv(an / "tie_structure.csv")
    img  = _img_b64(an / "tie_structure.png")
    diff = _load_csv(an / "query_difficulty.csv")
    img2 = _img_b64(an / "query_difficulty.png")

    explain = """
<p>The LLM uses a 1–10 integer scale (divided by 10), producing only 10 distinct scores.
This causes heavy ties: in 195 of 199 queries, both GT and multiple other documents all
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
    lvl  = _load_csv(an / "score_level_breakdown.csv")
    img  = _img_b64(an / "score_level_breakdown.png")
    fp   = _load_csv(an / "false_positive_analysis.csv")
    img2 = _img_b64(an / "false_positive_analysis.png")
    mg   = _load_csv(an / "marginal_score_value.csv")
    img3 = _img_b64(an / "marginal_score_value.png")

    explain = """
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


def _sec_threshold_filter(an: Path) -> str:
    filt = _load_csv(an / "threshold_filter_analysis.csv")
    img  = _img_b64(an / "threshold_filter.png")

    explain = """
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
        "At threshold 1.0: pool shrinks from ~5 000 to ~11 docs per query, "
        "recall stays at 98%, precision reaches 8.8%. "
        "This is the optimal operating point for a two-stage pipeline: "
        "LLM filter → embedding-distance ranker.",
        "good",
    )

    return explain + tbl + note + _figure(img, "Pool size, precision, recall, F1 at each threshold (left, centre). "
                                               "Precision vs recall trade-off (right).")


def _sec_hybrid(an: Path) -> str:
    wt   = _load_csv(an / "within_tier_dist.csv")
    img  = _img_b64(an / "within_tier_dist.png")
    img2 = _img_b64(an / "topk_jaccard.png")
    jac  = _load_csv(an / "topk_jaccard_summary.csv")

    wr = float(wt["gt_would_win_dist"].mean() * 100) if wt is not None else float("nan")
    pct_further = float(wt["pct_ngt_further"].mean()) if wt is not None else float("nan")

    explain = f"""
<p>If a secondary tiebreaker is applied within the top-scoring tier (docs at score=1.0),
choosing the document with the <em>lowest</em> <code>dist_to_gt</code> (i.e. cosine
distance to the GT embedding), does GT win?</p>"""

    big = _stat_cards(
        ("Hybrid win rate",     f"{wr:.1f}%",          "LLM filter + dist tiebreaker",         "green"),
        ("Mean % peers further",f"{pct_further:.1f}%", "non-GT peers with higher dist than GT", "blue"),
    )

    key = _callout(
        f"<b>In {wr:.1f}% of queries where GT reaches score=1.0, it also has the lowest "
        "dist_to_gt within that tier.</b> A two-stage system — (1) keep score=1.0 docs, "
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
            + _figure(img, "Left: 100% of queries have GT as the lowest-distance doc within the top-tier. "
                           "Right: win rate bar chart.")
            + "<h3>Top-K Jaccard: Agreement Between LLM Ranking and Distance Ranking</h3>"
            + jac_tbl
            + _figure(img2, "Jaccard similarity between top-K by LLM score and top-K by dist_to_gt. "
                            "Moderate agreement (~0.3–0.4) at K=5–20."))


def _sec_distance_analysis(an: Path) -> str:
    img1 = _img_b64(an / "scatter_score_vs_dist.png")
    img2 = _img_b64(an / "distance_quantile_analysis.png")

    explain = """
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
            + _figure(img1, "LLM score vs dist_to_gt scatter (10 000 non-GT sample + all 199 GTs). "
                            "GT docs appear at dist≈0 across all score levels where they appear.")
            + _figure(img2, "Mean and median LLM score by dist_to_gt decile (non-GT docs only). "
                            "Q1 = 10% of docs closest to GT; Q10 = farthest."))


def _sec_failures(an: Path) -> str:
    fail = _load_csv(an / "failure_cases.csv")
    img  = _img_b64(an / "failure_cases.png")

    n_fail = len(fail) if fail is not None else 0
    explain = f"""
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


def _sec_conclusions(an: Path) -> str:
    txt_path = an / "extended_summary.txt"
    raw = txt_path.read_text() if txt_path.exists() else ""
    # strip the ASCII border for embedding
    lines = [l for l in raw.splitlines() if not set(l.strip()).issubset({"=", ""})]

    findings = [
        ("<b>LLM as coarse filter (excellent)</b> — At threshold 1.0, pool shrinks ~450× "
         "while retaining GT in 98% of queries. Precision in the retained set is 8.8% "
         "(1 GT among ~11 docs).",
         "good"),
        ("<b>LLM as precise ranker (limited)</b> — Tie-corrected MRR = 0.257, not 0.98. "
         "The model cannot discriminate within the top-scoring tier.",
         "warn"),
        ("<b>Hybrid pipeline (near-perfect)</b> — LLM filter (score=1.0) → embedding-distance "
         "tiebreaker recovers GT in 100% of the top-tier queries (97.5% overall, accounting "
         "for 4 failures where GT never reaches score=1.0).",
         "key"),
        ("<b>False positives are semantically real</b> — Non-GT docs scoring 1.0 have median "
         "dist_to_gt ≈ 0.3. They are genuine semantic neighbours of GT, not noise.",
         "info"),
        ("<b>Negative correlation confirmed across all methods</b> — Spearman −0.17, "
         "Pearson −0.39, Kendall −0.13. Higher LLM score consistently predicts lower "
         "embedding distance to GT.",
         "info"),
    ]

    body = ""
    for text, kind in findings:
        body += _callout(text, kind)

    if lines:
        body += "<hr><h3>Extended Summary (raw)</h3>"
        body += "<pre style='font-size:12px;color:#475569;white-space:pre-wrap'>" + "\n".join(lines) + "</pre>"

    return body


# ── main assembler ────────────────────────────────────────────────────────────

def generate(run_dir: Path) -> Path:
    an      = run_dir / ANALYSIS_DIR
    db_path = run_dir / "experiment_results.db"
    cfg_path = run_dir / "experiment_config.json"

    if not an.exists():
        raise FileNotFoundError(f"analysis/ folder not found in {run_dir}. Run analyze_experiment.py first.")

    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}

    # DB stats
    n_rows, n_queries = 0, 0
    if db_path.exists():
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*), COUNT(DISTINCT query_idx) FROM results WHERE llm_score IS NOT NULL")
        n_rows, n_queries = cur.fetchone()
        conn.close()

    toc_sections = [
        ("executive",    "Executive Summary"),
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
        ("executive",   "Executive Summary",            _sec_executive),
        ("correlation", "Correlation Analysis",          _sec_correlation),
        ("detection",   "GT Detection — ROC / PR",       _sec_gt_detection),
        ("rank",        "GT Rank & Recovery Rate",        _sec_rank_recovery),
        ("ties",        "Tie Structure & Query Difficulty", _sec_tie_structure),
        ("discretize",  "Score Discretisation",          _sec_score_discretization),
        ("filter",      "LLM as Threshold Filter",       _sec_threshold_filter),
        ("hybrid",      "Hybrid Approach: LLM + Distance Tiebreaker", _sec_hybrid),
        ("distance",    "Distance vs Score Analysis",    _sec_distance_analysis),
        ("failures",    "Failure Case Analysis",         _sec_failures),
        ("conclusions", "Conclusions",                   _sec_conclusions),
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

    out_path = an / "report.html"
    out_path.write_text(html, encoding="utf-8")
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
    print(f"\nReport  : {out}")
    print(f"Size    : {out.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
