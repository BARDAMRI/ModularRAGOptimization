"""Print-ready (light theme) figures and summary numbers for the research proposal.

Outputs to proposal/figures/ and prints the numbers used in the text.
"""
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from load_experiment_data import RUN_DIR, load_data

OUT = Path(__file__).resolve().parents[2] / "proposal/figures"
OUT.mkdir(parents=True, exist_ok=True)
CSV = RUN_DIR / "analysis/csv"

NAVY, GOLD, RED, GREEN, GREY = "#1B3D6E", "#C9A227", "#C03030", "#1A7F37", "#8A8F98"
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 200})

d = load_data(verbose=False)
V = d["doc_vecs"]; idx = d["doc_id_to_idx"]; qv = d["query_idx_to_vec"]
qm = d["queries_meta"].set_index("query_idx"); sc = d["scores_df"]

# ── per-query triangle geometry ──────────────────────────────────────────────
rows = []
for q, g in sc.groupby("query_idx"):
    gi = idx.get(str(qm.loc[q, "gt_id"]))
    if gi is None:
        continue
    g3 = [idx[x] for x in g.loc[g.llm_score >= 1.0, "doc_id"] if x in idx]
    if not g3:
        continue
    qq, gg = qv[q], V[gi]
    c = V[g3].mean(0); c /= np.linalg.norm(c)
    # arc lengths (radians) on the unit sphere
    a = np.arccos(np.clip(c @ gg, -1, 1))    # side opposite Q  (C–GT)
    b = np.arccos(np.clip(qq @ gg, -1, 1))   # side opposite C  (Q–GT)
    cc = np.arccos(np.clip(qq @ c, -1, 1))   # side opposite GT (Q–C)
    def ang(opp, s1, s2):
        return np.degrees(np.arccos(np.clip((np.cos(opp) - np.cos(s1) * np.cos(s2)) / (np.sin(s1) * np.sin(s2)), -1, 1)))
    # Q–C coordinate frame (Euclidean chord, Q=-1, C=+1)
    u = c - qq; h = np.linalg.norm(u) / 2; u /= np.linalg.norm(u); m = (qq + c) / 2
    rel = gg - m; x = (rel @ u) / h; perp = np.linalg.norm(rel - (rel @ u) * u)
    rows.append(dict(q=q, angQ=ang(a, b, cc), angC=ang(b, a, cc), angGT=ang(cc, a, b),
                     dQGT=1 - qq @ gg, dCGT=1 - c @ gg, dQC=1 - qq @ c, x=x, perp=perp, nG3=len(g3)))
T = pd.DataFrame(rows)
print(f"triangle n={len(T)}  angles Q/C/GT = {T.angQ.mean():.1f}/{T.angC.mean():.1f}/{T.angGT.mean():.1f}  "
      f"sum={T[['angQ','angC','angGT']].sum(1).mean():.1f}  d(Q,GT)={T.dQGT.mean():.3f} d(C,GT)={T.dCGT.mean():.3f} "
      f"d(Q,C)={T.dQC.mean():.3f}  GT frame x={T.x.mean():.2f} perp={T.perp.mean():.2f}  "
      f"r(angGT,angC)={T.angGT.corr(T.angC):.3f}")

fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.7), gridspec_kw=dict(width_ratios=[1.25, 1]))
for col, lab, colr in [("angQ", r"$\angle Q$", RED), ("angC", r"$\angle C$", GREEN), ("angGT", r"$\angle GT$", GOLD)]:
    ax[0].hist(T[col], bins=np.arange(20, 110, 5), alpha=0.55, color=colr, label=f"{lab}  (mean {T[col].mean():.1f}°)")
ax[0].set_xlabel("vertex angle (degrees)"); ax[0].set_ylabel("queries"); ax[0].legend(frameon=False, fontsize=8)
ax[0].set_title("(a) Angles of the Q–C–GT spherical triangle", fontsize=9, loc="left")
ax[1].scatter(T.x, T.perp, s=9, color=NAVY, alpha=0.55, linewidths=0)
ax[1].scatter([-1], [0], marker="*", s=140, color=RED, zorder=3, label="Q")
ax[1].scatter([1], [0], marker="P", s=90, color=GREEN, zorder=3, label="C")
ax[1].scatter([T.x.mean()], [T.perp.mean()], marker="D", s=50, color=GOLD, edgecolor="k", zorder=4,
              label=f"mean GT ({T.x.mean():.2f}, {T.perp.mean():.2f})")
ax[1].axhline(0, color=GREY, lw=0.6); ax[1].set_xlim(-1.3, 1.5); ax[1].set_ylim(-0.05, 0.8)
ax[1].set_xlabel("position along the Q–C axis  (Q = −1, C = +1)"); ax[1].set_ylabel("perpendicular offset (chord distance)")
ax[1].legend(frameon=False, fontsize=7.5, loc="upper left")
ax[1].set_title("(b) Where the GT lies relative to Q and C", fontsize=9, loc="left")
fig.tight_layout(); fig.savefig(OUT / "triangle.pdf"); plt.close(fig)

# ── relevance gradient ───────────────────────────────────────────────────────
sc2 = sc.copy(); sc2["di"] = sc2.doc_id.map(idx)
grad = []
for q, g in sc2.groupby("query_idx"):
    gi = idx.get(str(qm.loc[q, "gt_id"]))
    if gi is None:
        continue
    g = g.dropna(subset=["di"]); di = g.di.astype(int).to_numpy()
    dgt = 1 - V[di] @ V[gi]
    keep = di != gi
    grad.append(pd.DataFrame({"s": g.llm_score.to_numpy()[keep], "d": dgt[keep]}))
G = pd.concat(grad)
lv = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
means = [G.d[np.isclose(G.s, s)].mean() for s in lv]
counts = [int(np.isclose(G.s, s).sum()) for s in lv]
print("gradient:", dict(zip(lv, np.round(means, 3))), counts)

fig, ax = plt.subplots(figsize=(3.6, 2.5))
ax.bar([str(s) for s in lv], means, color=[GREY, GREY, GREY, GOLD, GOLD, GREEN], width=0.65)
for i, (m, n) in enumerate(zip(means, counts)):
    ax.text(i, m + 0.015, f"{m:.2f}", ha="center", fontsize=8)
ax.set_xlabel("LLM relevance score of the document"); ax.set_ylabel("mean cosine distance to the GT")
ax.set_ylim(0, 1.0); ax.set_title("Judged relevance vs. distance to the GT", fontsize=9, loc="left")
fig.tight_layout(); fig.savefig(OUT / "gradient.pdf"); plt.close(fig)

# ── static family sweeps + adaptive prototype ───────────────────────────────
bfs = pd.read_csv(CSV / "circle_bfs/exp2_bfs_ranks.csv")
alphas = sorted({int(c.split("_")[-1]) / 100 for c in bfs.columns if c.startswith("rank_h1_")})
ks = sorted({int(c.split("_k")[1]) for c in bfs.columns if c.startswith("rank_h3_")})
h1_top1 = [(bfs[f"rank_h1_{int(round(a*100)):03d}"] == 1).mean() * 100 for a in alphas]
h1_mean = [bfs[f"rank_h1_{int(round(a*100)):03d}"].mean() for a in alphas]
h3_top1 = [(bfs[f"rank_h3_k{k:04d}"] == 1).mean() * 100 for k in ks]
b1_top1 = (bfs.rank_b1 == 1).mean() * 100; b2_top1 = (bfs.rank_b2 == 1).mean() * 100
ad = pd.read_csv(CSV / "adaptive_search/adaptive_search_results.csv")
ad_alphas = [1.0, 0.9, 0.7, 0.5, 0.3, 0.1]
ad_calls = [ad[f"calls_a{int(a*10):02d}"].mean() for a in ad_alphas]
print("H1 top1 by alpha:", dict(zip(alphas, np.round(h1_top1, 1))))
print("H3 top1 by k:", dict(zip(ks, np.round(h3_top1, 1))))
print("adaptive prototype mean calls by alpha:", dict(zip(ad_alphas, np.round(ad_calls, 2))))

fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.4))
ax[0].plot(alphas, h1_top1, "o-", color=NAVY, ms=3.5); ax[0].axhline(b1_top1, color=GREY, ls="--", lw=0.8)
ax[0].text(0.52, b1_top1 + 0.4, "B1 (query only)", fontsize=7.5, color=GREY)
ax[0].set_xlabel(r"$\alpha$  (weight on the query)"); ax[0].set_ylabel("top-1 (%)"); ax[0].set_title("(a) H1: static blend", fontsize=9, loc="left")
ax[1].semilogx(ks, h3_top1, "o-", color=NAVY, ms=3.5); ax[1].axhline(b1_top1, color=GREY, ls="--", lw=0.8); ax[1].axhline(b2_top1, color=RED, ls=":", lw=0.8)
ax[1].text(60, b2_top1 + 3.5, "B2 (centroid only)", fontsize=7.5, color=RED)
ax[1].text(60, b1_top1 - 6, "B1 (query only)", fontsize=7.5, color=GREY)
ax[1].set_xlabel("k  (window re-ranked by centroid)"); ax[1].set_ylabel("top-1 (%)"); ax[1].set_title("(b) H3: window re-rank", fontsize=9, loc="left")
ax[2].plot(ad_alphas, ad_calls, "o-", color=NAVY, ms=3.5); ax[2].axhline(ad_calls[0], color=GREY, ls="--", lw=0.8)
ax[2].set_xlabel(r"fixed $\alpha$  after first relevant hit"); ax[2].set_ylabel("documents inspected until GT")
ax[2].set_title("(c) Adaptive prototype, fixed α", fontsize=9, loc="left"); ax[2].invert_xaxis()
fig.tight_layout(); fig.savefig(OUT / "sweeps.pdf"); plt.close(fig)

# ── trilateration examples (existing light-theme PNGs) ──────────────────────
for q in ("037", "118"):
    shutil.copy(RUN_DIR / f"analysis/03_centroid_geometry/charts_vector_space/geometric/query_{q}.png", OUT / f"trilateration_q{q}.png")

# ── group sizes (G1/G2/G3) ───────────────────────────────────────────────────
ms = pd.read_csv(CSV / "group_analysis/master_summary.csv")
print("group cols:", [c for c in ms.columns][:20])
for c in ("n_g1", "n_g2", "n_g3"):
    if c in ms: print(c, round(ms[c].mean(), 2), int(ms[c].median()))
print("saved figures to", OUT)
