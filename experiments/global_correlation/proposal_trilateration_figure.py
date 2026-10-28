"""
Figure for the proposal: two questions drawn in two dimensions so that every
document keeps its exact cosine distance to the question and to the correct
answer (trilateration). Plain-language axis names and legend.

Usage:
  python experiments/global_correlation/proposal_trilateration_figure.py [q_idx ...]
Writes proposal/figures/trilateration_qNNN.png (default: 37 and 118).
"""
import json
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "results/cloud_runs/staged_run_2026-08-03_08-32-45"
EMB = RUN / "embeddings"
GRP = RUN / "analysis/csv/group_analysis"
OUT = ROOT / "proposal/figures"

C_Q, C_GT, C_C = "#D7263D", "#1E7A4B", "#1F5FBF"
C_REL, C_NEAR, C_OTHER = "#F08A24", "#8FBDEB", "#C9CCD1"


def load():
    qv = np.load(EMB / "query_vecs.npy")
    dv = np.load(EMB / "doc_vecs.npy")
    qids = json.loads((EMB / "query_ids.json").read_text())
    dids = [str(d) for d in json.loads((EMB / "doc_ids.json").read_text())]
    qv /= np.linalg.norm(qv, axis=1, keepdims=True)
    dv /= np.linalg.norm(dv, axis=1, keepdims=True)
    texts = (pd.read_parquet(RUN / "scoring_queue.parquet", columns=["query_idx", "query_text"])
             .drop_duplicates("query_idx").set_index("query_idx")["query_text"].to_dict())
    return qv, dv, {int(q): i for i, q in enumerate(qids)}, {d: i for i, d in enumerate(dids)}, texts


def place(d_q, d_gt, L):
    """2-D point with exact distances d_q to (0,0) and d_gt to (L,0)."""
    x = (d_q**2 - d_gt**2 + L**2) / (2 * L)
    y = np.sqrt(np.clip(d_q**2 - x**2, 0, None))
    return x, y


def draw(q, qv, dv, qpos, dpos, texts):
    pq = pd.read_csv(GRP / "per_query" / f"query_{q:03d}.csv")
    pq["doc_id"] = pq["doc_id"].astype(str)
    pq = pq[pq.doc_id.isin(dpos)].copy()
    idx = np.array([dpos[d] for d in pq.doc_id])
    V = dv[idx]
    Q = qv[qpos[q]]
    gt_row = pq[pq.is_gt == 1].index[0]
    GT = V[pq.index.get_loc(gt_row)]
    rel = (pq.llm_score >= 1.0).values
    C = V[rel].mean(0); C /= np.linalg.norm(C)

    d_q, d_gt, d_c = 1 - V @ Q, 1 - V @ GT, 1 - V @ C
    L, r_c, d_qc = 1 - Q @ GT, 1 - C @ GT, 1 - Q @ C
    x, y = place(d_q, d_gt, L)
    cx, cy = place(d_qc, r_c, L)
    n_before = int(((d_q < L) & ~pq.is_gt.values.astype(bool)).sum())
    near_c = (d_c < r_c) & ~rel
    n_near_c = int((d_c < r_c).sum())

    fig, ax = plt.subplots(figsize=(3.7, 4.15), dpi=300)
    other = ~rel & ~near_c
    ax.scatter(x[other], y[other], s=2.5, c=C_OTHER, alpha=0.45, lw=0, zorder=1)
    ax.scatter(x[near_c], y[near_c], s=11, c=C_NEAR, alpha=0.9, lw=0, zorder=3)
    ax.scatter(x[rel], y[rel], s=15, c=C_REL, alpha=0.95, lw=0, zorder=4)
    ax.add_patch(Circle((0, 0), L, fill=False, ls="--", lw=1.1, ec=C_Q, zorder=5))
    ax.add_patch(Circle((cx, cy), r_c, fill=False, ls="--", lw=1.1, ec=C_C, zorder=5))
    ax.scatter([0], [0], marker="*", s=110, c=C_Q, ec="white", lw=0.8, zorder=7)
    ax.scatter([L], [0], marker="D", s=50, c=C_GT, ec="white", lw=0.8, zorder=7)
    ax.scatter([cx], [cy], marker="P", s=75, c=C_C, ec="white", lw=0.8, zorder=7)

    # circle annotations in words
    ax.annotate(f"closer to the\nquestion than\nthe answer:\n{n_before} documents",
                xy=(-L * 0.71, L * 0.71), xytext=(-0.235, 0.42), fontsize=5.8, color=C_Q, ha="left", va="bottom",
                arrowprops=dict(arrowstyle="-", color=C_Q, lw=0.8))
    ax.annotate(f"as close to the centroid as the\ncorrect answer: {n_near_c} documents",
                xy=(cx + r_c * 0.72, cy + r_c * 0.70), xytext=(cx + r_c * 0.72 + 0.03, cy + r_c + 0.07),
                fontsize=5.8, color=C_C, ha="left", arrowprops=dict(arrowstyle="-", color=C_C, lw=0.8))

    handles = [
        Line2D([], [], marker="*", ls="", ms=8, c=C_Q, label="the question"),
        Line2D([], [], marker="D", ls="", ms=5, c=C_GT, label="the correct answer"),
        Line2D([], [], marker="P", ls="", ms=6, c=C_C, label="centroid of the 1.0 documents"),
        Line2D([], [], marker="o", ls="", ms=4, c=C_REL, label=f"documents the judge scored 1.0 ({int(rel.sum())})"),
        Line2D([], [], marker="o", ls="", ms=3.5, c=C_NEAR, label=f"in blue circle, scored below 1.0 ({int(near_c.sum())})"),
        Line2D([], [], marker="o", ls="", ms=2, c=C_OTHER, label=f"all other documents ({int(other.sum())})"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=5.4, frameon=True, framealpha=0.95, borderpad=0.5, labelspacing=0.35, handletextpad=0.4)

    ax.set_xlabel("Position along the line from the question\n"
                  f"to the correct answer (answer at {L:.2f})", fontsize=6.4)
    ax.set_ylabel("Distance off that line", fontsize=6.4)
    title = "\n".join(textwrap.wrap(f"Question {q}: “{texts.get(q, '').strip()}”", 56)[:3])
    ax.set_title(title, fontsize=5.8, loc="left", pad=5)
    xmax = max(0.75, float(np.percentile(x, 99.5)))
    ax.set_xlim(-0.24, xmax); ax.set_ylim(-0.03, 1.1)
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(labelsize=5.8, length=2.5); ax.grid(alpha=0.25, lw=0.5)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    fig.tight_layout()
    out = OUT / f"trilateration_q{q:03d}.png"
    fig.savefig(out, bbox_inches='tight', pad_inches=0.05); plt.close(fig)
    print(f"q{q}: d(Q,GT)={L:.3f} d(C,GT)={r_c:.3f} d(Q,C)={d_qc:.3f} before={n_before} nearC={n_near_c} rel={rel.sum()} -> {out}")


if __name__ == "__main__":
    qs = [int(a) for a in sys.argv[1:]] or [37, 118]
    qv, dv, qpos, dpos, texts = load()
    for q in qs:
        draw(q, qv, dv, qpos, dpos, texts)
