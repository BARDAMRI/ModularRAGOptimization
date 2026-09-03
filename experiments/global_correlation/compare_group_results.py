"""
Compare master_summary.csv from group_analysis (original) vs group_analysis_v2 (refactored).
Prints a clear pass/fail per column.
"""

from pathlib import Path
import numpy as np
import pandas as pd

RUN_DIR = (
    Path(__file__).resolve().parent.parent.parent
    / "results/cloud_runs/staged_run_2026-08-03_08-32-45"
)

orig = pd.read_csv(RUN_DIR / "analysis/csv/group_analysis/master_summary.csv")
refac = pd.read_csv(RUN_DIR / "analysis/csv/group_analysis_v2/master_summary.csv")

print(f"Original  rows: {len(orig)}")
print(f"Refactored rows: {len(refac)}")
assert len(orig) == len(refac), "Row count mismatch!"

# Align on query_idx
orig = orig.sort_values("query_idx").reset_index(drop=True)
refac = refac.sort_values("query_idx").reset_index(drop=True)
assert (orig["query_idx"].values == refac["query_idx"].values).all(), "query_idx mismatch!"

numeric_cols = [
    "n_pool", "n_g1", "n_g2", "n_g3",
    "n_g1_g2", "n_g1_g3", "n_g2_g3", "n_g1_g2_g3",
    "jaccard_g1_g2", "jaccard_g1_g3", "jaccard_g2_g3", "jaccard_g1_g2_g3",
    "gt_in_g1", "gt_in_g2", "gt_in_g3", "gt_in_all",
    "dist_query_to_gt", "dist_centroid_to_gt",
    "rag_failed",
]

string_cols = ["g1_doc_ids", "g2_doc_ids", "g3_doc_ids", "gt_id"]

print("\n--- Numeric columns ---")
all_pass = True
for col in numeric_cols:
    if col not in orig.columns or col not in refac.columns:
        print(f"  {col:<30} MISSING in one file")
        all_pass = False
        continue
    a = orig[col].values.astype(float)
    b = refac[col].values.astype(float)
    # handle NaN (e.g. dist_centroid_to_gt when G3 empty)
    nan_match = np.isnan(a) == np.isnan(b)
    val_match = np.where(np.isnan(a) | np.isnan(b), True, np.isclose(a, b, atol=1e-9))
    ok = nan_match.all() and val_match.all()
    n_diff = (~val_match).sum() + (~nan_match).sum()
    status = "PASS" if ok else f"FAIL  ({n_diff} rows differ)"
    print(f"  {col:<30} {status}")
    if not ok:
        all_pass = False
        bad = np.where(~val_match | ~nan_match)[0]
        for i in bad[:5]:
            print(f"      query {orig.loc[i,'query_idx']}: orig={a[i]:.8f}  refac={b[i]:.8f}")

print("\n--- String columns (doc_id sets) ---")
for col in string_cols:
    if col not in orig.columns or col not in refac.columns:
        continue
    mismatches = (orig[col].fillna("") != refac[col].fillna("")).sum()
    status = "PASS" if mismatches == 0 else f"FAIL  ({mismatches} rows differ)"
    print(f"  {col:<30} {status}")
    if mismatches:
        all_pass = False

print()
print("=" * 45)
print("OVERALL:", "PASS — refactored output is identical" if all_pass else "FAIL — see diffs above")
print("=" * 45)
