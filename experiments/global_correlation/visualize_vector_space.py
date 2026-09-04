"""
Vector space visualization — two chart types per query.

1. Radial density plot (all 196 queries, seconds — uses precomputed distances):
   How many pool docs lie at each cosine distance from Query / GT / Centroid.
   Vertical lines at r1 (G1 edge) and r2 (G2 edge) show why G2 >> G1.

2. UMAP 2D scatter (top 10 extreme queries — requires embedding load, ~3-4 min):
   All 5035 docs projected to 2D, colored by group membership.
   Query, GT, Centroid shown as distinct markers.

Output: analysis/charts/vector_space/
  radial/query_NNN.png   (196 files)
  umap/query_NNN.png     (10 files)

Usage:
  python visualize_vector_space.py           # both
  python visualize_vector_space.py --radial  # radial only (no model load)
  python visualize_vector_space.py --umap    # UMAP only
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

sys.path.insert(0, str(Path(__file__).resolve().parent))
from load_experiment_data import RUN_DIR

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
N_EXTREME = 10
N_BINS    = 40
GROUP_DIR = RUN_DIR / "analysis/csv/group_analysis"
SQ_PATH   = RUN_DIR / "scoring_queue.parquet"
OUT_DIR   = RUN_DIR / "analysis/charts/vector_space"

C_QUERY    = "#E53E3E"
C_GT       = "#276749"
C_CENTROID = "#2563EB"
C_G1_ONLY  = "#FC8181"
C_G2_ONLY  = "#93C5FD"
C_OVERLAP  = "#A78BFA"
C_G3_REST  = "#F6AD55"
C_NONE     = "#CBD5E0"


# ---------------------------------------------------------------------------
# Shared loaders
# ---------------------------------------------------------------------------

def load_master_summary() -> pd.DataFrame:
    ms = pd.read_csv(GROUP_DIR / "master_summary.csv")
    ms["gap"] = ms["n_g2"] - ms["n_g1"]
    return ms


def load_per_query(q_idx: int) -> pd.DataFrame:
    return pd.read_csv(GROUP_DIR / "per_query" / f"query_{q_idx:03d}.csv")


def load_query_texts() -> dict[int, str]:
    sq = pd.read_parquet(SQ_PATH, columns=["query_idx", "query_text"])
    return sq.drop_duplicates("query_idx").set_index("query_idx")["query_text"].to_dict()


def align_to_doc_order(pq: pd.DataFrame, all_doc_ids: list) -> pd.DataFrame:
    """Re-index pq rows to match the all_doc_ids ordering used by doc_vecs."""
    ref = pd.DataFrame({"doc_id": [str(d) for d in all_doc_ids]})
    pq2 = pq.copy()
    pq2["doc_id"] = pq2["doc_id"].astype(str)
    return ref.merge(pq2, on="doc_id", how="left").fillna(0)


# ---------------------------------------------------------------------------
# Radial density plot
# ---------------------------------------------------------------------------

def plot_radial(
    q_idx: int,
    pq: pd.DataFrame,
    r1: float,
    r2: float,
    n_g1: int,
    n_g2: int,
    n_g3: int,
    query_text: str,
    out_path: Path,
) -> None:
    bins = np.linspace(0, 1, N_BINS + 1)
    fig, ax = plt.subplots(figsize=(9, 4.5))

    ax.hist(pq["dist_to_query"].dropna(),      bins=bins, alpha=0.5, color=C_QUERY,    label="from Query")
    ax.hist(pq["dist_to_gt_reembed"].dropna(), bins=bins, alpha=0.5, color=C_GT,       label="from GT")
    if "dist_to_centroid" in pq.columns:
        ax.hist(pq["dist_to_centroid"].dropna(), bins=bins, alpha=0.5, color=C_CENTROID, label="from Centroid")

    ax.axvline(r1, color=C_QUERY,    lw=2.0, ls="--", label=f"r1 = {r1:.3f}  (G1 = {n_g1} docs)")
    if r2 > 0:
        ax.axvline(r2, color=C_CENTROID, lw=2.0, ls="--", label=f"r2 = {r2:.3f}  (G2 = {n_g2} docs)")

    ax.set_xlabel("Cosine distance from center point", fontsize=10)
    ax.set_ylabel("Pool docs in bin", fontsize=10)
    short = (query_text[:85] + "...") if len(query_text) > 85 else query_text
    ax.set_title(f"Query {q_idx:03d}  |  G1={n_g1}  G2={n_g2}  G3={n_g3}\n{short}", fontsize=9, loc="left")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_xlim(0, 1)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def run_radial(ms: pd.DataFrame, q_text_map: dict[int, str]) -> None:
    out = OUT_DIR / "radial"
    out.mkdir(parents=True, exist_ok=True)
    print(f"Radial density plots — {len(ms)} queries ...")
    for i, (_, mrow) in enumerate(ms.iterrows()):
        q_idx = int(mrow["query_idx"])
        plot_radial(
            q_idx=q_idx,
            pq=load_per_query(q_idx),
            r1=float(mrow["dist_query_to_gt"]),
            r2=float(mrow["dist_centroid_to_gt"]),
            n_g1=int(mrow["n_g1"]),
            n_g2=int(mrow["n_g2"]),
            n_g3=int(mrow["n_g3"]),
            query_text=q_text_map.get(q_idx, ""),
            out_path=out / f"query_{q_idx:03d}.png",
        )
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(ms)}")
    print(f"  Done — {len(list(out.glob('*.png')))} files in {out}")


# ---------------------------------------------------------------------------
# UMAP scatter
# ---------------------------------------------------------------------------

def fit_umap_on_docs(doc_vecs: np.ndarray):
    import umap as umap_lib
    print("  Fitting UMAP (5035 docs × 768 dims) ...")
    reducer = umap_lib.UMAP(
        n_components=2, n_neighbors=20, min_dist=0.1,
        metric="cosine", random_state=42, verbose=False,
    )
    reducer.fit(doc_vecs)
    print("  UMAP ready.")
    return reducer


def compute_centroid(g3_vecs: np.ndarray) -> np.ndarray | None:
    if len(g3_vecs) == 0:
        return None
    c = g3_vecs.mean(axis=0)
    norm = np.linalg.norm(c)
    return c / norm if norm > 0 else None


def plot_umap(
    q_idx: int,
    aligned_pq: pd.DataFrame,
    reducer,
    q_vec: np.ndarray,
    gt_vec: np.ndarray,
    centroid: np.ndarray | None,
    r1: float,
    r2: float,
    n_g1: int,
    n_g2: int,
    n_g3: int,
    query_text: str,
    out_path: Path,
) -> None:
    doc_xy = reducer.embedding_   # (n_docs, 2) — already fitted

    g1 = aligned_pq["in_g1"].values.astype(bool)
    g2 = aligned_pq["in_g2"].values.astype(bool)
    g3 = aligned_pq["in_g3"].values.astype(bool)

    g1_only = g1 & ~g2
    g2_only = g2 & ~g1
    both    = g1 & g2
    g3_rest = g3 & ~g1 & ~g2
    none_   = ~g1 & ~g2 & ~g3

    # Project special points into the fitted UMAP space
    special      = [q_vec, gt_vec] + ([centroid] if centroid is not None else [])
    special_xy   = reducer.transform(np.array(special))
    q_xy         = special_xy[0]
    gt_xy        = special_xy[1]
    c_xy         = special_xy[2] if centroid is not None else None

    fig, ax = plt.subplots(figsize=(8, 7))

    def sc(mask, color, label, s=15, alpha=0.4, z=2):
        if mask.sum():
            ax.scatter(doc_xy[mask, 0], doc_xy[mask, 1],
                       c=color, s=s, alpha=alpha, label=label, zorder=z, linewidths=0)

    sc(none_,   C_NONE,    f"Other ({int(none_.sum())})",    s=10, alpha=0.2, z=1)
    sc(g3_rest, C_G3_REST, f"G3 only ({int(g3_rest.sum())})", s=25, alpha=0.65)
    sc(g2_only, C_G2_ONLY, f"G2 only ({int(g2_only.sum())})", s=35, alpha=0.75, z=3)
    sc(g1_only, C_G1_ONLY, f"G1 only ({int(g1_only.sum())})", s=35, alpha=0.75, z=3)
    sc(both,    C_OVERLAP, f"G1+G2 ({int(both.sum())})",    s=45, alpha=0.85, z=4)

    ax.scatter(*q_xy,  marker="*", s=260, c=C_QUERY,    zorder=6, label="Query",    edgecolors="white", linewidths=0.6)
    ax.scatter(*gt_xy, marker="D", s=120, c=C_GT,       zorder=6, label="GT",       edgecolors="white", linewidths=0.6)
    if c_xy is not None:
        ax.scatter(*c_xy, marker="P", s=180, c=C_CENTROID, zorder=6, label="Centroid", edgecolors="white", linewidths=0.6)

    short = (query_text[:75] + "...") if len(query_text) > 75 else query_text
    ax.set_title(
        f"Query {q_idx:03d} — UMAP  |  G1={n_g1}  G2={n_g2}  G3={n_g3}  "
        f"r1={r1:.3f}  r2={r2:.3f}\n{short}",
        fontsize=9, loc="left",
    )
    ax.set_xlabel("UMAP-1", fontsize=9)
    ax.set_ylabel("UMAP-2", fontsize=9)
    ax.legend(fontsize=8, loc="best", markerscale=0.9)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def run_umap(ms: pd.DataFrame, q_text_map: dict[int, str]) -> None:
    from load_experiment_data import load_data
    out = OUT_DIR / "umap"
    out.mkdir(parents=True, exist_ok=True)

    data             = load_data()
    doc_vecs         = data["doc_vecs"]
    query_idx_to_vec = data["query_idx_to_vec"]
    doc_id_to_idx    = data["doc_id_to_idx"]
    all_doc_ids      = data["all_doc_ids"]

    reducer = fit_umap_on_docs(doc_vecs)

    top10 = ms.nlargest(N_EXTREME, "gap")
    print(f"UMAP scatter — top {N_EXTREME} extreme queries ...")
    for _, mrow in top10.iterrows():
        q_idx  = int(mrow["query_idx"])
        gt_idx = doc_id_to_idx.get(str(mrow["gt_id"]))
        if gt_idx is None:
            print(f"  Query {q_idx}: GT not in pool, skipping.")
            continue

        q_vec    = query_idx_to_vec[q_idx]
        gt_vec   = doc_vecs[gt_idx]
        pq       = load_per_query(q_idx)
        aligned  = align_to_doc_order(pq, all_doc_ids)
        g3_mask  = aligned["in_g3"].values.astype(bool)
        centroid = compute_centroid(doc_vecs[g3_mask])

        print(f"  Query {q_idx} (gap={int(mrow['gap'])}) ...")
        plot_umap(
            q_idx=q_idx,
            aligned_pq=aligned,
            reducer=reducer,
            q_vec=q_vec,
            gt_vec=gt_vec,
            centroid=centroid,
            r1=float(mrow["dist_query_to_gt"]),
            r2=float(mrow["dist_centroid_to_gt"]),
            n_g1=int(mrow["n_g1"]),
            n_g2=int(mrow["n_g2"]),
            n_g3=int(mrow["n_g3"]),
            query_text=q_text_map.get(q_idx, ""),
            out_path=out / f"query_{q_idx:03d}.png",
        )
    print(f"  Done — {len(list(out.glob('*.png')))} files in {out}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--radial", action="store_true", help="Radial density plots only")
    parser.add_argument("--umap",   action="store_true", help="UMAP scatter only")
    args = parser.parse_args()
    do_both = not args.radial and not args.umap

    ms         = load_master_summary()
    q_text_map = load_query_texts()

    if args.radial or do_both:
        run_radial(ms, q_text_map)

    if args.umap or do_both:
        run_umap(ms, q_text_map)

    print(f"\nAll outputs in: {OUT_DIR}")


if __name__ == "__main__":
    main()
