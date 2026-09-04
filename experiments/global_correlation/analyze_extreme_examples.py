"""
Extreme examples deep dive — top N queries by G2-G1 gap.

Reads pre-computed output from analyze_groups.py (no re-embedding needed).
Classifies each query into:
  Mechanism A: centroid farther from GT than query (r2 >= r1)
  Mechanism B: centroid closer to GT than query  (r2 <  r1)

Output: analysis/csv/extreme_examples/
  top10_extreme_examples.csv  — full stats for each query
  g2_only_query_NNN.csv       — docs in G2 but not G1, per query
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from load_experiment_data import RUN_DIR

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N_TOP = 10
GROUP_DIR = RUN_DIR / "analysis/csv/group_analysis"
SQ_PATH   = RUN_DIR / "scoring_queue.parquet"
OUT_DIR   = RUN_DIR / "analysis/csv/extreme_examples"


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_master_summary() -> pd.DataFrame:
    """Load group_analysis master summary and add gap column."""
    ms = pd.read_csv(GROUP_DIR / "master_summary.csv")
    ms["gap"] = ms["n_g2"] - ms["n_g1"]
    return ms


def load_query_meta() -> dict[int, tuple[str, str]]:
    """Return {query_idx: (query_text, gt_id)} from scoring_queue.parquet."""
    sq = pd.read_parquet(SQ_PATH, columns=["query_idx", "query_text", "gt_id"])
    meta = sq.drop_duplicates("query_idx").set_index("query_idx")
    return {
        int(idx): (row["query_text"], str(row["gt_id"]))
        for idx, row in meta.iterrows()
    }


def load_per_query_csv(q_idx: int) -> pd.DataFrame:
    return pd.read_csv(GROUP_DIR / "per_query" / f"query_{q_idx:03d}.csv")


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def classify_mechanism(r1: float, r2: float) -> str:
    """
    A_large_r2: centroid is at least as far from GT as the query.
                Dense document region explains G2 >> G1 despite "bad" centroid.
    B_small_r2: centroid is closer to GT; coherent G3 cluster drives the gap.
    """
    return "A_large_r2" if r2 >= r1 else "B_small_r2"


def safe_density(n_docs: int, radius: float) -> float:
    return round(n_docs / radius, 2) if radius > 0 else 0.0


def _safe_agg(series: pd.Series, fn) -> float | None:
    return round(float(fn(series)), 4) if len(series) > 0 else None


def per_query_derived_stats(pq: pd.DataFrame) -> dict:
    """
    Compute derived statistics from a per-query membership CSV.

    Groups used:
      g3      — docs with in_g3 = 1
      g2_only — docs with in_g2 = 1 and in_g1 = 0  (the "extra" docs G2 adds over G1)
      overlap — docs with in_g1 = 1 and in_g2 = 1
    """
    g3      = pq[pq["in_g3"] == 1]
    g2_only = pq[(pq["in_g2"] == 1) & (pq["in_g1"] == 0)]
    overlap = pq[(pq["in_g1"] == 1) & (pq["in_g2"] == 1)]

    dist_col = "dist_to_gt_reembed"
    return {
        "g3_mean_dist_gt":         _safe_agg(g3[dist_col],      np.mean),
        "g3_max_dist_gt":          _safe_agg(g3[dist_col],      np.max),
        "g3_spread_std":           _safe_agg(g3[dist_col],      np.std),
        "g2_only_count":           len(g2_only),
        "g2_only_mean_dist_gt":    _safe_agg(g2_only[dist_col], np.mean),
        "g2_only_mean_dist_query": _safe_agg(g2_only["dist_to_query"], np.mean)
                                   if "dist_to_query" in pq.columns else None,
        "g1_g2_overlap_count":     len(overlap),
    }


def build_summary_row(
    q_idx: int,
    mrow: pd.Series,
    query_text: str,
    pq: pd.DataFrame,
) -> dict:
    """Assemble one output row from master-summary fields + derived stats."""
    r1 = float(mrow["dist_query_to_gt"])
    r2 = float(mrow["dist_centroid_to_gt"])
    return {
        "query_idx":        q_idx,
        "query_text":       query_text,
        "rag_failed":       int(mrow["rag_failed"]),
        "n_g1":             int(mrow["n_g1"]),
        "n_g2":             int(mrow["n_g2"]),
        "n_g3":             int(mrow["n_g3"]),
        "gap_g2_minus_g1":  int(mrow["gap"]),
        "r1_dist_Q_GT":     round(r1, 4),
        "r2_dist_C_GT":     round(r2, 4),
        "r2_lt_r1":         bool(r2 < r1),
        "mechanism":        classify_mechanism(r1, r2),
        "gt_in_g1":         int(mrow["gt_in_g1"]),
        "gt_in_g2":         int(mrow["gt_in_g2"]),
        "gt_in_g3":         int(mrow["gt_in_g3"]),
        "density_g1":       safe_density(int(mrow["n_g1"]), r1),
        "density_g2":       safe_density(int(mrow["n_g2"]), r2),
        **per_query_derived_stats(pq),
    }


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def save_g2_only_csv(pq: pd.DataFrame, q_idx: int, out_dir: Path) -> None:
    """Write docs that are in G2 but not G1 — the "extra" documents G2 adds."""
    g2_only = pq[(pq["in_g2"] == 1) & (pq["in_g1"] == 0)]
    g2_only.to_csv(out_dir / f"g2_only_query_{q_idx:03d}.csv", index=False)


def print_results(rows: list[dict]) -> None:
    sep = "=" * 74
    print(f"\n{sep}")
    print("EXTREME EXAMPLES — TOP 10 LARGEST G2-G1 GAP")
    print(sep)
    hdr = (f"{'#':<3} {'Q':>4} {'G1':>4} {'G2':>4} {'G3':>4} {'Gap':>5}"
           f" {'r1':>6} {'r2':>6} {'r2<r1':>6} {'Mech':<12} {'RAG'}")
    print(hdr)
    print("-" * 74)
    for i, r in enumerate(rows, 1):
        print(
            f"{i:<3} {r['query_idx']:>4} {r['n_g1']:>4} {r['n_g2']:>4}"
            f" {r['n_g3']:>4} {r['gap_g2_minus_g1']:>5}"
            f" {r['r1_dist_Q_GT']:>6.3f} {r['r2_dist_C_GT']:>6.3f}"
            f" {'Yes' if r['r2_lt_r1'] else 'No':>6}"
            f" {r['mechanism']:<12}"
            f" {'FAIL' if r['rag_failed'] else '-'}"
        )

    mech_counts: dict[str, int] = {}
    for r in rows:
        mech_counts[r["mechanism"]] = mech_counts.get(r["mechanism"], 0) + 1

    print(f"\nMechanism distribution:")
    for mech, cnt in sorted(mech_counts.items()):
        label = ("centroid farther from GT, lands in dense doc region"
                 if mech == "A_large_r2"
                 else "centroid closer to GT, coherent G3 cluster")
        print(f"  {mech} ({cnt} queries): {label}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ms         = load_master_summary()
    query_meta = load_query_meta()
    top_n      = ms.nlargest(N_TOP, "gap")

    rows: list[dict] = []
    for _, mrow in top_n.iterrows():
        q_idx = int(mrow["query_idx"])
        query_text, _ = query_meta.get(q_idx, ("", ""))
        pq = load_per_query_csv(q_idx)

        rows.append(build_summary_row(q_idx, mrow, query_text, pq))
        save_g2_only_csv(pq, q_idx, OUT_DIR)

    pd.DataFrame(rows).to_csv(OUT_DIR / "top10_extreme_examples.csv", index=False)

    print_results(rows)
    print(f"\nOutputs written to: {OUT_DIR}")


if __name__ == "__main__":
    main()
