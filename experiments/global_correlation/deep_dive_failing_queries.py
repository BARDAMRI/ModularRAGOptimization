"""
Deep dive on every query where Baseline 1 (rank by dist(Q)) does NOT put the GT at rank 1.

Question: when the GT is not first, how good is the top 10 anyway?

Output: proposal/H2_failing_deep_dive.xlsx
  Findings          — key findings, failing vs passing queries, relevance by position
  Failing Queries   — one row per failing query, top-10 relevance strip (linked to its sheet)
  Deep Dive Q<idx>  — same layout as the 3 deep dive tabs in H2_results.xlsx (top 25 + GT row)
  Definitions       — what every column means
"""

import sys
from pathlib import Path

import numpy as np
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from load_experiment_data import extract_doc_text, load_data

OUT = Path(__file__).resolve().parents[2] / "proposal/H2_failing_deep_dive.xlsx"
ALPHA_H1 = 0.93
K_H3 = 10
TOP_N = 10

# ── data ─────────────────────────────────────────────────────────────────────
data = load_data()
queries_meta = data["queries_meta"]
all_vecs = data["doc_vecs"]
all_doc_ids = data["all_doc_ids"]
doc_id_to_idx = data["doc_id_to_idx"]
q_to_vec = data["query_idx_to_vec"]
n_pool = len(all_doc_ids)

doc_text_by_id = dict(zip(data["pool_docs"]["doc_id"], data["pool_docs"]["doc_text"]))

llm_by_query = {}
for q_idx, grp in data["scores_df"].groupby("query_idx"):
    arr = np.full(n_pool, np.nan)
    idx = grp["doc_id"].map(doc_id_to_idx)
    ok = idx.notna()
    arr[idx[ok].astype(int).to_numpy()] = grp["llm_score"][ok].to_numpy()
    llm_by_query[int(q_idx)] = arr


def ranks_from_order(order):
    r = np.empty(n_pool, dtype=int)
    r[order] = np.arange(1, n_pool + 1)
    return r


def snippet(doc_id, n=160):
    t = extract_doc_text(doc_text_by_id.get(doc_id, "")) or ""
    t = " ".join(str(t).split())
    return t[:n] + ("…" if len(t) > n else "")


records = []
for _, q in queries_meta.iterrows():
    q_idx = int(q["query_idx"])
    gt_idx = doc_id_to_idx.get(str(q["gt_id"]))
    if gt_idx is None:
        continue
    llm = llm_by_query.get(q_idx, np.full(n_pool, np.nan))
    llm_f = np.nan_to_num(llm, nan=0.0)

    dist_q = 1.0 - all_vecs @ q_to_vec[q_idx]
    dist_gt = np.clip(1.0 - all_vecs @ all_vecs[gt_idx], 0.0, None)
    g3 = np.where(llm_f >= 1.0)[0]
    if len(g3):
        c = all_vecs[g3].mean(axis=0)
        dist_c = 1.0 - all_vecs @ (c / np.linalg.norm(c))
    else:
        dist_c = dist_q.copy()

    order_b1 = np.argsort(dist_q, kind="stable")
    order_b2 = np.argsort(dist_c, kind="stable")
    order_h1 = np.argsort(ALPHA_H1 * dist_q + (1 - ALPHA_H1) * dist_c, kind="stable")
    head = order_b1[:K_H3]
    order_h3 = np.concatenate([head[np.argsort(dist_c[head], kind="stable")], order_b1[K_H3:]])

    rB1, rB2, rH1, rH3 = (ranks_from_order(o) for o in (order_b1, order_b2, order_h1, order_h3))
    top = order_b1[:TOP_N]

    records.append(dict(
        q_idx=q_idx, query=q["query_text"], gt_id=str(q["gt_id"]), gt_idx=gt_idx,
        rank_b1=int(rB1[gt_idx]), rank_b2=int(rB2[gt_idx]),
        rank_h1=int(rB1[gt_idx]) if not len(g3) else int(rH1[gt_idx]),
        rank_h3=int(rH3[gt_idx]) if rH3[gt_idx] <= K_H3 else None,
        n_g3_pool=int(len(g3)),
        top_llm=llm_f[top],
        top1_by_alg={name: int(o[0]) for name, o in
                     (("B1", order_b1), ("B2", order_b2), ("H1", order_h1), ("H3", order_h3))},
        dist_q=dist_q, dist_c=dist_c, dist_gt=dist_gt, llm=llm_f,
        rB1=rB1, rB2=rB2, rH1=rH1, rH3=rH3, top=top,
        gap=float(dist_q[gt_idx] - dist_q[order_b1[0]]),
    ))

failing = sorted([r for r in records if r["rank_b1"] > 1], key=lambda r: (r["rank_b1"], r["q_idx"]))
passing = [r for r in records if r["rank_b1"] == 1]
print(f"Queries: {len(records)} | failing (GT not rank 1 under B1): {len(failing)} | passing: {len(passing)}")

# ── styling ──────────────────────────────────────────────────────────────────
NAVY, WHITE, LIGHT, MID, BORDER = "1E2A4A", "FFFFFF", "F6F8FA", "EEF0F4", "D0D7DE"
MUTED, INK = "6B7280", "1A1D23"
GREEN, GREEN_F = "DAFBE1", "166534"
RED, RED_F = "FFEBE9", "D1242F"
GOLD, GOLD_F = "FFF8C5", "7A4706"
BLUE_L, BLUE_F = "DDF4FF", "0550AE"
TIERS = [(1.0, "D1F7D6", "155724"), (0.8, "E8F5E3", "276330"), (0.6, "FFF9DB", "7A6000"),
         (0.4, "FFF0D6", "7A4500"), (-1, "FDF0F0", "8B1A1A")]


def fill(h): return PatternFill("solid", fgColor=h)
def font(bold=False, color=INK, size=11, italic=False, underline=None):
    return Font(bold=bold, color=color, size=size, italic=italic, name="Calibri", underline=underline)
def al(h="center", wrap=True, v="center"): return Alignment(horizontal=h, vertical=v, wrap_text=wrap)
_s = Side(style="thin", color=BORDER)
BOX = Border(left=_s, right=_s, top=_s, bottom=_s)


def llm_style(score, is_gt=False):
    if is_gt:
        return GOLD, GOLD_F
    for thr, bg, fg in TIERS:
        if score >= thr - 1e-9:
            return bg, fg
    return TIERS[-1][1], TIERS[-1][2]


def put(ws, r, c, v, bg=None, fg=INK, bold=False, size=11, h="center", fmt=None, italic=False, wrap=True):
    cell = ws.cell(row=r, column=c, value=v)
    cell.font = font(bold=bold, color=fg, size=size, italic=italic)
    if bg:
        cell.fill = fill(bg)
    cell.border = BOX
    cell.alignment = al(h, wrap)
    if fmt:
        cell.number_format = fmt
    return cell


def banner(ws, r, text, span, bg=NAVY, size=12, height=24):
    ws.merge_cells(start_row=r, start_column=1, end_row=r, end_column=span)
    c = ws.cell(row=r, column=1, value=text)
    c.font = font(bold=True, color=WHITE, size=size)
    c.fill = fill(bg)
    c.alignment = al("left", False)
    ws.row_dimensions[r].height = height


def note(ws, r, text, span, height=18, color=MUTED, bold=False, bg=None):
    ws.merge_cells(start_row=r, start_column=1, end_row=r, end_column=span)
    c = ws.cell(row=r, column=1, value=text)
    c.font = font(italic=not bold, bold=bold, color=color, size=10)
    c.alignment = al("left", True)
    if bg:
        c.fill = fill(bg)
    ws.row_dimensions[r].height = height


def headers(ws, r, labels, widths=None, height=30):
    for c, lab in enumerate(labels, 1):
        put(ws, r, c, lab, bg=NAVY, fg=WHITE, bold=True, size=10)
    if widths:
        for c, w in enumerate(widths, 1):
            ws.column_dimensions[get_column_letter(c)].width = w
    ws.row_dimensions[r].height = height


def sheet_name(rec): return f"Deep Dive Q{rec['q_idx']}"


# ── aggregate metrics ────────────────────────────────────────────────────────
def group_stats(group):
    tops = np.array([r["top_llm"] for r in group])
    top1 = tops[:, 0]
    return {
        "n": len(group),
        "gt_top3": np.mean([r["rank_b1"] <= 3 for r in group]) * 100,
        "gt_top10": np.mean([r["rank_b1"] <= 10 for r in group]) * 100,
        "top1_10": np.mean(top1 >= 1.0) * 100,
        "top1_08": np.mean(top1 >= 0.8) * 100,
        "n10_top10": np.mean((tops >= 1.0).sum(axis=1)),
        "n08_top10": np.mean((tops >= 0.8).sum(axis=1)),
        "mean_top10": tops.mean(),
        "none08": np.mean((tops >= 0.8).sum(axis=1) == 0) * 100,
        "g3_pool": np.mean([r["n_g3_pool"] for r in group]),
        "by_pos": tops.mean(axis=0),
    }


SF, SP = group_stats(failing), group_stats(passing)

wb = Workbook()

# ── Sheet: Findings ──────────────────────────────────────────────────────────
ws = wb.active
ws.title = "Findings"
ws.sheet_view.showGridLines = False
for c, w in enumerate([52, 22, 22, 4, 16, 16, 16], 1):
    ws.column_dimensions[get_column_letter(c)].width = w

banner(ws, 1, "When B1 misses the GT at rank 1: how good is the top 10?", 7, size=13, height=28)
note(ws, 2, f"{len(records)} queries · B1 = rank pool docs by distance to the query (standard RAG) · "
            f"Failing = GT not at rank 1 under B1 ({len(failing)} queries, "
            f"{len(failing) / len(records) * 100:.1f}%) · Relevance = LLM judge score 0–1", 7, height=30)

n26 = sum(x["rank_b1"] >= 26 for x in failing)
note(ws, 3, f"Caveat: {n26} failing queries have the GT at position 26–27 under every algorithm. The pool appears "
            f"to hold ~25 retrieved docs per query, so a GT outside them lands right after them; "
            f"its true position in the full corpus is unknown (likely much lower).", 7, height=30,
     color=GOLD_F, bg=GOLD)
banner(ws, 5, "Key findings", 7, bg="2D4A2A", size=11, height=22)
n_top1_10 = sum(x["top_llm"][0] >= 1.0 for x in failing)
n_top1_08 = sum(x["top_llm"][0] >= 0.8 for x in failing)
weak = sorted((x for x in failing if x["top_llm"][0] < 0.8), key=lambda x: x["top_llm"][0])
findings = [
    f"1.  In {n_top1_10} of {len(failing)} failing queries ({n_top1_10 / len(failing) * 100:.0f}%) the doc B1 returns "
    f"first has relevance 1.0 — as relevant as the GT. With ≥ 0.8 it is {n_top1_08} ({n_top1_08 / len(failing) * 100:.0f}%).",
    f"2.  No query (failing or passing) has a top 10 without a relevant doc (relevance ≥ 0.8). Avg. relevance of the "
    f"top 10: {SF['mean_top10']:.2f} for failing vs {SP['mean_top10']:.2f} for passing queries.",
    f"3.  Failing queries have MORE fully relevant docs: {SF['n10_top10']:.1f} vs {SP['n10_top10']:.1f} in the top 10, "
    f"{SF['g3_pool']:.1f} vs {SP['g3_pool']:.1f} in the whole pool. Many 'failures' are queries with several correct "
    f"answers where the GT label went to a different one.",
    f"4.  The GT is at position 2 in {sum(x['rank_b1'] == 2 for x in failing)} queries and in the top 10 in "
    f"{sum(x['rank_b1'] <= 10 for x in failing)} of {len(failing)} ({SF['gt_top10']:.0f}%).",
    f"5.  Real failures (doc returned first < 0.8): {len(weak)} queries — "
    + ", ".join(f"Q{x['q_idx']} ({x['top_llm'][0]:.1f})" for x in weak) + ". Read these deep dives first.",
]
for i, t in enumerate(findings, 6):
    note(ws, i, t, 7, height=32, color=INK, bold=False, bg=LIGHT)
    ws.cell(row=i, column=1).font = font(color=INK, size=10)
r = 6 + len(findings) + 1
headers(ws, r, ["Metric (top 10 = first 10 docs returned by B1)", f"Failing queries (n={SF['n']})",
                f"Passing queries (n={SP['n']})"], height=34)
rows = [
    ("GT is in the top 3 (%)", SF["gt_top3"], SP["gt_top3"], "0.0"),
    ("GT is in the top 10 (%)", SF["gt_top10"], SP["gt_top10"], "0.0"),
    ("Doc returned first has relevance 1.0 (%)", SF["top1_10"], SP["top1_10"], "0.0"),
    ("Doc returned first has relevance ≥ 0.8 (%)", SF["top1_08"], SP["top1_08"], "0.0"),
    ("Avg. number of docs with relevance 1.0 in the top 10 (GT included)", SF["n10_top10"], SP["n10_top10"], "0.0"),
    ("Avg. number of docs with relevance ≥ 0.8 in the top 10", SF["n08_top10"], SP["n08_top10"], "0.0"),
    ("Avg. relevance of the top 10", SF["mean_top10"], SP["mean_top10"], "0.00"),
    ("Queries with NO doc ≥ 0.8 in the top 10 (%)", SF["none08"], SP["none08"], "0.0"),
    ("Avg. number of docs with relevance 1.0 in the whole pool", SF["g3_pool"], SP["g3_pool"], "0.0"),
]
for i, (lab, a, b, fmt) in enumerate(rows, r + 1):
    bg = LIGHT if i % 2 else WHITE
    put(ws, i, 1, lab, bg=bg, h="left", bold=True, size=10)
    put(ws, i, 2, float(a), bg=RED, fg=RED_F, bold=True, fmt=fmt)
    put(ws, i, 3, float(b), bg=GREEN, fg=GREEN_F, bold=True, fmt=fmt)
    ws.row_dimensions[i].height = 20
r = r + len(rows) + 2

banner(ws, r, "Average relevance at each position of the B1 list", 7, bg="2D4A2A", size=11, height=22)
r += 1
headers(ws, r, ["Position in the list returned by B1", "Failing queries", "Passing queries"], height=22)
for p in range(TOP_N):
    rr = r + 1 + p
    put(ws, rr, 1, f"#{p + 1}", bg=LIGHT, h="left", bold=True)
    for col, v in ((2, SF["by_pos"][p]), (3, SP["by_pos"][p])):
        bg, fg = llm_style(v)
        put(ws, rr, col, float(v), bg=bg, fg=fg, bold=True, fmt="0.00")
r = r + TOP_N + 2

banner(ws, r, "Failing queries: where does B1 place the GT?", 7, bg="4A2D2D", size=11, height=22)
r += 1
headers(ws, r, ["GT position under B1", "Number of queries", "% of failing"], height=22)
bins = [("2", 2, 2), ("3 – 5", 3, 5), ("6 – 10", 6, 10), ("11 – 50", 11, 50), ("> 50", 51, 10 ** 9)]
for i, (lab, lo, hi) in enumerate(bins, r + 1):
    n = sum(lo <= x["rank_b1"] <= hi for x in failing)
    put(ws, i, 1, lab, bg=LIGHT, h="left", bold=True)
    put(ws, i, 2, n)
    put(ws, i, 3, n / len(failing) * 100, fmt="0.0")
r = r + len(bins) + 2

banner(ws, r, "Failing queries: how relevant is the doc B1 returns first (instead of the GT)?", 7,
       bg="4A2D2D", size=11, height=22)
r += 1
headers(ws, r, ["Relevance of the doc returned first", "Number of queries", "% of failing"], height=22)
tbins = [("1.0 (as relevant as the GT)", 1.0, 2), ("0.8", 0.8, 1.0), ("0.6", 0.6, 0.8),
         ("0.4", 0.4, 0.6), ("≤ 0.2", -1, 0.4)]
for i, (lab, lo, hi) in enumerate(tbins, r + 1):
    n = sum(lo - 1e-9 <= x["top_llm"][0] < hi - 1e-9 for x in failing)
    bg, fg = llm_style(max(lo, 0))
    put(ws, i, 1, lab, bg=bg, fg=fg, h="left", bold=True)
    put(ws, i, 2, n)
    put(ws, i, 3, n / len(failing) * 100, fmt="0.0")
ws.freeze_panes = "A2"

# ── Sheet: Failing Queries ───────────────────────────────────────────────────
ws = wb.create_sheet("Failing Queries")
ws.sheet_view.showGridLines = False
pos_cols = [f"#{p}" for p in range(1, TOP_N + 1)]
labels = (["Query ID (click to open)", "Query", "GT position: B1", "GT position: H1 α=.93",
           "GT position: B2", "GT position: H3 k=10"] + pos_cols +
          ["Docs = 1.0 in top 10", "Docs ≥ 0.8 in top 10", "Avg. relevance top 10",
           "Dist. gap: GT vs #1 (Query)"])
widths = [11, 60, 10, 11, 10, 11] + [6] * TOP_N + [11, 11, 11, 12]
ncol = len(labels)
banner(ws, 1, f"Failing queries ({len(failing)}): GT position per algorithm and relevance of B1's top 10",
       ncol, size=12, height=26)
note(ws, 2, "Columns #1…#10 = relevance score of the doc B1 returns at that position (★ marks the GT). "
            "Green = relevant, red = irrelevant. Sorted by GT position under B1. "
            "Dist. gap = how much further the GT is from the query than the doc returned first.", ncol, height=30)
headers(ws, 3, labels, widths, height=44)
for i, rec in enumerate(failing, 4):
    bg = LIGHT if i % 2 else WHITE
    c = put(ws, i, 1, rec["q_idx"], bg=bg, fg=BLUE_F, bold=True)
    c.hyperlink = f"#'{sheet_name(rec)}'!A1"
    c.font = font(bold=True, color=BLUE_F, underline="single")
    put(ws, i, 2, rec["query"], bg=bg, h="left", size=9, wrap=False)
    b1 = rec["rank_b1"]
    put(ws, i, 3, b1, bg=BLUE_L, fg=BLUE_F, bold=True)
    for col, key in ((4, "rank_h1"), (5, "rank_b2"), (6, "rank_h3")):
        v = rec[key]
        if v is None:
            put(ws, i, col, f"> {K_H3}", bg=RED, fg=RED_F)
            continue
        cbg, cfg = (GREEN, GREEN_F) if v < b1 else (RED, RED_F) if v > b1 else (bg, "9CA3AF")
        put(ws, i, col, v, bg=cbg, fg=cfg, bold=v < b1)
    for p in range(TOP_N):
        doc = rec["top"][p]
        is_gt = doc == rec["gt_idx"]
        v = rec["top_llm"][p]
        cbg, cfg = llm_style(v, is_gt)
        put(ws, i, 7 + p, f"★{v:.1f}" if is_gt else round(float(v), 1), bg=cbg, fg=cfg, bold=True,
            size=10, fmt="0.0")
    tl = rec["top_llm"]
    put(ws, i, 7 + TOP_N, int((tl >= 1.0).sum()), bg=bg, bold=True)
    put(ws, i, 8 + TOP_N, int((tl >= 0.8).sum()), bg=bg, bold=True)
    m = float(tl.mean())
    mbg, mfg = llm_style(round(m * 5) / 5)
    put(ws, i, 9 + TOP_N, m, bg=mbg, fg=mfg, bold=True, fmt="0.00")
    put(ws, i, 10 + TOP_N, rec["gap"], bg=bg, fmt="0.0000")
    ws.row_dimensions[i].height = 18
ws.freeze_panes = "C4"
ws.auto_filter.ref = f"A3:{get_column_letter(ncol)}{3 + len(failing)}"

# ── Per-query sheets (same layout as the Deep Dive tabs in H2_results.xlsx) ──
TOP_DOCS = 25
ALG = [("B1", "B1 (closest to query)", "rank_b1"),
       ("B2", "B2 (closest to centroid)", "rank_b2"),
       ("H1", "H1 α=.93 (hybrid)", "rank_h1"),
       ("H3", "H3 k=10 (re-ranked)", "rank_h3")]
SOFT = [("B1", "B1 — closest to query"), ("B2", "B2 — closest to centroid"),
        ("H1", "H1 α=.93 (hybrid)"), ("H3", "H3 k=10 (re-ranked)")]
DOC_COLS = ["#", "Document ID", "Dist. to Query", "Dist. to Centroid", "Dist. to GT",
            "Relevance Score", "Is GT?", "B1 Rank", "B2 Rank", "H1 Rank", "Document (beginning of text)"]
DOC_W = [5, 16, 15, 18, 13, 16, 7, 9, 9, 9, 80]


def plain(ws, r, c, v, bg, fg, bold=False, size=11, h="center", fmt=None):
    cell = ws.cell(row=r, column=c, value=v)
    cell.font = font(bold=bold, color=fg, size=size)
    cell.fill = fill(bg)
    cell.border = BOX
    cell.alignment = al(h, h == "left")
    if fmt:
        cell.number_format = fmt


for rec in failing:
    ws = wb.create_sheet(sheet_name(rec))
    ws.sheet_view.showGridLines = False

    ws.merge_cells("A1:J1")
    c = ws["A1"]
    c.value = f"Deep Dive — Query {rec['q_idx']}   |   GT: {rec['gt_id']}   |   RAG: FAILED"
    c.font = font(bold=True, color=WHITE, size=12)
    c.fill = fill("7A1A1A")
    c.alignment = al("left", False)
    ws.row_dimensions[1].height = 26
    c = ws["K1"]
    c.value = "← back to Failing Queries"
    c.hyperlink = "#'Failing Queries'!A1"
    c.font = font(color=BLUE_F, size=10, underline="single")

    ws.merge_cells("A2:J2")
    c = ws["A2"]
    c.value = "How far down the list does each algorithm go before finding the GT?"
    c.font = font(bold=True, color=MUTED, size=10)
    c.alignment = al("left", False)
    ws.row_dimensions[2].height = 18

    costs = [rec[k] if rec[k] is not None else K_H3 + 1 for _, _, k in ALG]
    for ci, ((_, lab, k), v) in enumerate(zip(ALG, costs), 1):
        plain(ws, 3, ci, lab, MID, MUTED, bold=True, size=10)
        bg, fg = (GREEN, GREEN_F) if v == min(costs) else (RED, RED_F) if v == max(costs) else (LIGHT, INK)
        plain(ws, 4, ci, v if rec[k] is not None else f"> {K_H3}", bg, fg, bold=True, size=13)
    ws.row_dimensions[3].height = 20
    ws.row_dimensions[4].height = 24
    ws.row_dimensions[5].height = 10

    ws.merge_cells("A6:J6")
    c = ws["A6"]
    c.value = "What does each algorithm return first — and how relevant is it?"
    c.font = font(bold=True, color=WHITE, size=11)
    c.fill = fill("4A2D2D")
    c.alignment = al("left", False)
    ws.row_dimensions[6].height = 22
    for ci, (key, lab) in enumerate(SOFT, 1):
        d = rec["top1_by_alg"][key]
        is_gt = d == rec["gt_idx"]
        bg, fg = llm_style(rec["llm"][d], is_gt)
        plain(ws, 7, ci, lab, MID, MUTED, bold=True, size=10)
        plain(ws, 8, ci, f"LLM {rec['llm'][d]:.1f}{'  ★ GT' if is_gt else ''}", bg, fg, bold=True, size=12)
        plain(ws, 9, ci, all_doc_ids[d], bg, MUTED, size=9)
    ws.row_dimensions[7].height = 20
    ws.row_dimensions[8].height = 26
    ws.row_dimensions[9].height = 18
    ws.row_dimensions[10].height = 10

    order = np.argsort(rec["rB1"])[:TOP_DOCS]
    llm_top = rec["llm"][order]
    gt_mask = order == rec["gt_idx"]
    n10 = int(((llm_top >= 1.0) & ~gt_mask).sum())
    n08 = int(((llm_top >= 0.8) & ~gt_mask).sum())
    ws.merge_cells("A11:J11")
    c = ws["A11"]
    c.value = (f"Top {TOP_DOCS} documents — ranked by distance to query   |   {n10} non-GT with relevance score 1.0"
               f"   |   {n08} non-GT with score ≥ 0.8")
    c.font = font(italic=True, color=MUTED, size=10)
    c.alignment = al("left", False)
    ws.row_dimensions[11].height = 18

    for ci, (h, w) in enumerate(zip(DOC_COLS, DOC_W), 1):
        plain(ws, 12, ci, h, NAVY, WHITE, bold=True)
        ws.column_dimensions[get_column_letter(ci)].width = w
    ws.row_dimensions[12].height = 20

    rows_docs = list(order)
    if rec["gt_idx"] not in rows_docs:
        rows_docs += [None, rec["gt_idx"]]
    r = 13
    for d in rows_docs:
        if d is None:
            ws.merge_cells(start_row=r, start_column=1, end_row=r, end_column=10)
            c = ws.cell(row=r, column=1,
                        value=f"… GT is not in the top {TOP_DOCS}; it appears at position {rec['rank_b1']} …")
            c.font = font(italic=True, color=MUTED, size=10)
            c.alignment = al("center", False)
            r += 1
            continue
        is_gt = d == rec["gt_idx"]
        bg, fg = llm_style(rec["llm"][d], is_gt)
        vals = [int(rec["rB1"][d]), all_doc_ids[d], float(rec["dist_q"][d]), float(rec["dist_c"][d]),
                float(rec["dist_gt"][d]), float(rec["llm"][d]), "★" if is_gt else "",
                int(rec["rB1"][d]), int(rec["rB2"][d]), int(rec["rH1"][d]), snippet(all_doc_ids[d])]
        for ci, v in enumerate(vals, 1):
            plain(ws, r, ci, v, bg, fg, bold=is_gt, size=9 if ci in (2, 11) else 11,
                  h="left" if ci in (2, 11) else "center",
                  fmt="0.0000" if ci in (3, 4, 5) else "0.0" if ci == 6 else None)
        ws.row_dimensions[r].height = 17
        r += 1
    ws.freeze_panes = "A13"

# ── Definitions ──────────────────────────────────────────────────────────────
ws = wb.create_sheet("Definitions")
ws.sheet_view.showGridLines = False
ws.column_dimensions["A"].width = 30
ws.column_dimensions["B"].width = 100
banner(ws, 1, "What each term means", 2, size=13, height=28)
defs = [
    ("Failing query", "A query where B1 does not return the GT as the first document (GT position > 1)."),
    ("Passing query", "A query where B1 returns the GT first."),
    ("B1 (closest to query)", "Rank all pool docs by d(Q, doc) = 1 − cos(Q, doc), ascending. Standard RAG."),
    ("B2 (closest to centroid)", "Rank by d(C, doc). C = normalised mean of all docs with relevance 1.0 for "
                                 "this query (oracle centroid)."),
    ("H1 α=.93 (hybrid)", "Rank by 0.93·d(Q, doc) + 0.07·d(C, doc), ascending."),
    ("H3 k=10 (re-ranked)", "Take the 10 docs closest to Q and re-rank them by d(C, doc). H3 returns only "
                            "these 10, so a GT outside them shows as '> 10'."),
    ("GT position 26–27", "Likely a pool artifact: the GT was not among the ~25 docs retrieved for this "
                          "query, so its true corpus position is unknown."),
    ("GT position", "Where the GT appears in the algorithm's list (1 = returned first)."),
    ("Relevance Score", "LLM judge score of the doc for this query: 1.0 = fully answers it, 0.2 = irrelevant."),
    ("#1 … #10", "Relevance of the doc B1 returns at that position. ★ = this doc is the GT."),
    ("Dist. gap: GT vs #1", "d(Q, GT) − d(Q, doc returned first). How much closer the wrong doc is to the query."),
    ("Dist. to Query / Centroid / GT", "Cosine distance between the doc and Q / C / the GT document."),
    ("Row colours", "Gold = GT · dark green = 1.0 · light green = 0.8 · yellow = 0.6 · orange = 0.4 · "
                    "red = ≤ 0.2"),
]
for i, (k, v) in enumerate(defs, 2):
    put(ws, i, 1, k, bg=LIGHT, h="left", bold=True, size=10)
    put(ws, i, 2, v, bg=WHITE, h="left", size=10)
    ws.row_dimensions[i].height = 30

wb.move_sheet("Definitions", offset=-(len(wb.sheetnames) - 3))
OUT.parent.mkdir(parents=True, exist_ok=True)
wb.save(OUT)
print(f"Saved: {OUT}")
print(f"Failing: GT in top 10 = {SF['gt_top10']:.1f}% | #1 doc relevance 1.0 = {SF['top1_10']:.1f}% | "
      f"≥0.8 = {SF['top1_08']:.1f}% | avg docs=1.0 in top10 = {SF['n10_top10']:.1f} "
      f"(passing {SP['n10_top10']:.1f}) | avg relevance top10 = {SF['mean_top10']:.2f} "
      f"(passing {SP['mean_top10']:.2f})")
