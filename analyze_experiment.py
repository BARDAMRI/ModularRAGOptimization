#!/usr/bin/env python3
"""
analyze_experiment.py — Post-run analysis for the global correlation experiment.

Reads experiment_results.db, produces CSVs, charts, and a summary report
in <run_dir>/analysis/.

Central question: Can LLM relevance scoring replace or supplement the retriever
when it fails to rank the ground-truth document (GT) at position 1?

Schema columns used:
  query_idx  — integer query index
  doc_id     — chroma document ID
  llm_score  — 0.0–1.0 LLM relevance score
  dist_to_gt — cosine distance from doc to GT embedding (0 = GT itself)
  is_gt      — 1 if this is the GT doc for this query
  rag_failed — 1 if actual retriever did NOT rank GT as top-1

Usage:
    python analyze_experiment.py                          # auto-detect latest run
    python analyze_experiment.py path/to/experiment_results.db
    python analyze_experiment.py --run-dir path/to/staged_run_YYYY-MM-DD
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy import stats

from analysis_common import (
    ANALYSIS_DIR,
    AnalysisLayout,
    build_per_query_overview,
    collect_run_health,
    find_latest_db,
    format_health_text_lines,
    load_scored_dataframe,
    save_run_health,
    write_analysis_index,
)

# optional sklearn — gracefully degrade if missing
try:
    from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("[WARN] scikit-learn not installed — ROC/PR curves will be skipped.")

plt.rcParams.update({"figure.dpi": 120, "font.size": 10})

# ── helpers ──────────────────────────────────────────────────────────────────

def _load_dataframe(db_path: Path) -> pd.DataFrame:
    print(f"Loading data from {db_path} …")
    df = load_scored_dataframe(db_path)
    print(f"  {len(df):,} scored rows, {df['query_idx'].nunique()} queries, "
          f"{df['is_gt'].sum()} GT rows")
    return df


def _save_fig(fig: plt.Figure, path: Path, title: str = "") -> None:
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    if title:
        print(f"  [chart] {path.name}  — {title}")
    else:
        print(f"  [chart] {path.name}")


def _save_csv(df: pd.DataFrame, path: Path, label: str = "") -> None:
    df.to_csv(path, index=False)
    print(f"  [csv]   {path.name}" + (f"  — {label}" if label else ""))


# ── per-query statistics base ─────────────────────────────────────────────────

def _build_per_query_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute GT rank (by llm_score DESC) and GT score per query.
    Returns DataFrame indexed by query_idx with:
        n_docs,         gt_llm_score, gt_dist_to_gt, gt_llm_rank, gt_llm_rank_avg,
        gt_rank_pct (percentile 0–100, 100 = best),
        rag_failed, spearman_r, spearman_p, pearson_r, pearson_p,
        best_ngt_score, score_gap, gt_score_percentile
    """
    rows = []
    for qidx, grp in df.groupby("query_idx"):
        gt_row = grp[grp["is_gt"] == 1]
        n_docs = len(grp)

        if gt_row.empty:
            continue

        gt_score = float(gt_row["llm_score"].iloc[0])
        gt_dist  = float(gt_row["dist_to_gt"].iloc[0])
        rag_fail = int(gt_row["rag_failed"].iloc[0])

        # rank by llm_score DESC (1 = highest)
        grp = grp.copy()
        grp["_rank"] = grp["llm_score"].rank(method="min", ascending=False)
        grp["_rank_avg"] = grp["llm_score"].rank(method="average", ascending=False)
        gt_rank = int(grp.loc[gt_row.index[0], "_rank"])
        gt_rank_avg = float(grp.loc[gt_row.index[0], "_rank_avg"])

        # GT score percentile among all docs in this query
        gt_pct = float(stats.percentileofscore(grp["llm_score"].values, gt_score, kind="rank"))

        # best non-GT score
        ngt = grp[grp["is_gt"] == 0]["llm_score"]
        best_ngt = float(ngt.max()) if len(ngt) > 0 else float("nan")
        score_gap = gt_score - best_ngt

        # correlations — four methods
        x_score = grp["llm_score"].values
        y_dist  = grp["dist_to_gt"].values
        y_gt    = grp["is_gt"].values.astype(float)
        if n_docs >= 5:
            sp_r,  sp_p  = stats.spearmanr(x_score, y_dist)
            pe_r,  pe_p  = stats.pearsonr(x_score, y_dist)
            kt_r,  kt_p  = stats.kendalltau(x_score, y_dist, variant="b")
            pb_r,  pb_p  = stats.pointbiserialr(y_gt, x_score)   # score separates GT from non-GT
        else:
            sp_r = sp_p = pe_r = pe_p = kt_r = kt_p = pb_r = pb_p = float("nan")

        rows.append({
            "query_idx":    qidx,
            "n_docs":       n_docs,
            "gt_llm_score": gt_score,
            "gt_dist_to_gt": gt_dist,
            "gt_llm_rank":     gt_rank,
            "gt_llm_rank_avg": gt_rank_avg,
            "gt_rank_pct":     gt_pct,
            "rag_failed":   rag_fail,
            "spearman_r":   sp_r,
            "spearman_p":   sp_p,
            "pearson_r":    pe_r,
            "pearson_p":    pe_p,
            "kendall_tau":  kt_r,
            "kendall_p":    kt_p,
            "pointbiserial_r": pb_r,
            "pointbiserial_p": pb_p,
            "best_ngt_score": best_ngt,
            "score_gap":    score_gap,
        })

    return pd.DataFrame(rows)


# ── section 1: per-query correlation ─────────────────────────────────────────

# Four correlation methods used:
#   Spearman r     — rank-based, monotonic; llm_score vs dist_to_gt
#   Pearson r      — linear; llm_score vs dist_to_gt
#   Kendall tau-b  — rank-based, robust to ties; llm_score vs dist_to_gt
#   Point-biserial — continuous vs binary; llm_score vs is_gt
#
# Spearman/Pearson/Kendall answer: "does higher LLM score predict lower distance to GT?"
# Point-biserial answers: "does LLM score separate the GT doc from all others?"
# Expected sign: negative for the first three (high score → low dist); positive for PB (GT scores higher).

def section_correlation(pqs: pd.DataFrame, layout: AnalysisLayout) -> dict:
    cols = ["query_idx", "n_docs",
            "spearman_r", "spearman_p",
            "pearson_r",  "pearson_p",
            "kendall_tau", "kendall_p",
            "pointbiserial_r", "pointbiserial_p"]
    corr = pqs[cols].copy()
    _save_csv(corr, layout.per_query_csv("per_query_correlation.csv"),
              "Spearman / Pearson / Kendall-tau / Point-biserial per query")

    valid = corr.dropna(subset=["spearman_r"])

    # per-method summary stats
    methods = [
        ("spearman_r",       "spearman_p",       "Spearman r\n(score vs dist_to_gt)",     "steelblue",     True),
        ("pearson_r",        "pearson_p",         "Pearson r\n(score vs dist_to_gt)",      "darkorange",    True),
        ("kendall_tau",      "kendall_p",         "Kendall τ-b\n(score vs dist_to_gt)",    "mediumseagreen",True),
        ("pointbiserial_r",  "pointbiserial_p",   "Point-biserial r\n(score vs is_gt)",    "mediumpurple",  False),
    ]

    summary_rows = []
    for col_r, col_p, label, _, expect_neg in methods:
        v = valid[col_r].dropna()
        p = valid[col_p].dropna()
        summary_rows.append({
            "method":         label.replace("\n", " "),
            "median":         v.median(),
            "mean":           v.mean(),
            "pct_negative":   (v < 0).mean() * 100,
            "pct_positive":   (v > 0).mean() * 100,
            "pct_sig_p05":    (p < 0.05).mean() * 100,
            "expected_sign":  "negative" if expect_neg else "positive",
        })
    summary_df = pd.DataFrame(summary_rows)
    _save_csv(summary_df, layout.global_csv("correlation_summary.csv"), "Cross-method correlation summary")

    # 2 × 2 histogram grid
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes = axes.ravel()
    for ax, (col_r, col_p, label, color, expect_neg) in zip(axes, methods):
        v = valid[col_r].dropna()
        ax.hist(v, bins=30, color=color, edgecolor="white", alpha=0.85)
        median_v = v.median()
        ax.axvline(median_v, color="red", linestyle="--",
                   label=f"median={median_v:.3f}")
        ax.axvline(0, color="black", linestyle=":", alpha=0.5)
        sig_pct = (valid[col_p].dropna() < 0.05).mean() * 100
        neg_pct = (v < 0).mean() * 100
        expected = "← expect negative" if expect_neg else "expect positive →"
        ax.set_xlabel(f"{label.split(chr(10))[0]}")
        ax.set_ylabel("Queries")
        ax.set_title(f"{label}\n"
                     f"neg={neg_pct:.1f}%  sig(p<.05)={sig_pct:.1f}%  {expected}")
        ax.legend(fontsize=8)

    fig.suptitle(
        "Per-Query Correlation: LLM Score vs Distance-to-GT  &  LLM Score vs is_GT\n"
        "Spearman / Pearson / Kendall τ-b / Point-biserial",
        fontsize=11,
    )
    plt.tight_layout()
    _save_fig(fig, layout.chart_base("correlation_histograms.png"),
              "4-method correlation histograms")

    # global (pooled) correlations across all 1M rows — summary only
    return {
        "median_spearman_r":  valid["spearman_r"].median(),
        "median_pearson_r":   valid["pearson_r"].median(),
        "median_kendall_tau": valid["kendall_tau"].median(),
        "median_pb_r":        valid["pointbiserial_r"].median(),
        "pct_negative_r":     (valid["spearman_r"] < 0).mean() * 100,
        "pct_significant_r":  (valid["spearman_p"] < 0.05).mean() * 100,
    }


# ── section 2: LLM rank of GT ────────────────────────────────────────────────

def section_gt_rank(pqs: pd.DataFrame, layout: AnalysisLayout) -> dict:
    rank_df = pqs[["query_idx", "gt_llm_rank", "gt_llm_score", "n_docs", "gt_rank_pct",
                    "rag_failed"]].copy()
    _save_csv(rank_df, layout.per_query_csv("gt_llm_rank.csv"), "GT rank by LLM score per query")

    ranks = rank_df["gt_llm_rank"].dropna().astype(int)
    p_at_1  = (ranks == 1).mean() * 100
    p_at_3  = (ranks <= 3).mean() * 100
    p_at_5  = (ranks <= 5).mean() * 100
    p_at_10 = (ranks <= 10).mean() * 100

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    # rank distribution (cap at 100 for readability)
    rank_cap = ranks.clip(upper=100)
    axes[0].hist(rank_cap, bins=50, color="steelblue", edgecolor="white")
    axes[0].set_xlabel("GT LLM Rank (capped at 100)")
    axes[0].set_ylabel("Queries")
    axes[0].set_title(f"GT Rank by LLM Score\nP@1={p_at_1:.1f}%  P@5={p_at_5:.1f}%  P@10={p_at_10:.1f}%")

    # CDF
    sorted_r = np.sort(ranks)
    cdf = np.arange(1, len(sorted_r) + 1) / len(sorted_r)
    axes[1].plot(sorted_r, cdf * 100, color="darkorange")
    for k, c in [(1, "red"), (5, "green"), (10, "purple"), (50, "gray")]:
        pct = (sorted_r <= k).mean() * 100
        axes[1].axvline(k, linestyle="--", color=c, alpha=0.6, label=f"K={k}: {pct:.1f}%")
    axes[1].set_xlabel("GT LLM Rank")
    axes[1].set_ylabel("Cumulative % of Queries")
    axes[1].set_title("CDF of GT LLM Rank")
    axes[1].set_xlim(left=0)
    axes[1].legend(fontsize=8)
    fig.suptitle("LLM Ability to Identify GT (rank 1 = perfect)")
    _save_fig(fig, layout.chart_base("gt_llm_rank.png"), "GT rank distribution + CDF")

    return {"p_at_1": p_at_1, "p_at_3": p_at_3, "p_at_5": p_at_5, "p_at_10": p_at_10}


# ── section 3: LLM recovery rates ────────────────────────────────────────────

def section_recovery(pqs: pd.DataFrame, layout: AnalysisLayout) -> dict:
    ks = [1, 3, 5, 10, 20, 50, 100, 200]
    ranks = pqs["gt_llm_rank"].dropna().astype(int)
    rows = []
    for k in ks:
        recovered = (ranks <= k).sum()
        total = len(ranks)
        rows.append({"K": k, "recovered": recovered, "total": total,
                     "recovery_rate_pct": recovered / total * 100})
    rec_df = pd.DataFrame(rows)
    _save_csv(rec_df, layout.global_csv("llm_recovery_at_k.csv"), "LLM recovery rate at K")

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar([str(k) for k in rec_df["K"]], rec_df["recovery_rate_pct"], color="steelblue", edgecolor="white")
    ax.set_xlabel("Top-K threshold")
    ax.set_ylabel("% Queries where GT is in top-K by LLM")
    ax.set_title("LLM Recovery Rate: GT in top-K\n(retriever failed on ALL queries)")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    for i, row in rec_df.iterrows():
        ax.text(i, row["recovery_rate_pct"] + 0.5, f"{row['recovery_rate_pct']:.1f}%",
                ha="center", va="bottom", fontsize=8)
    _save_fig(fig, layout.chart_base("llm_recovery_at_k.png"), "Recovery rate bar chart")

    return {"recovery_at_1": rec_df[rec_df["K"] == 1]["recovery_rate_pct"].iloc[0],
            "recovery_at_5": rec_df[rec_df["K"] == 5]["recovery_rate_pct"].iloc[0],
            "recovery_at_10": rec_df[rec_df["K"] == 10]["recovery_rate_pct"].iloc[0]}


# ── section 4: score distributions (GT vs non-GT) ────────────────────────────

def section_score_distribution(df: pd.DataFrame, layout: AnalysisLayout) -> dict:
    gt_scores  = df[df["is_gt"] == 1]["llm_score"]
    ngt_scores = df[df["is_gt"] == 0]["llm_score"]

    summary = pd.DataFrame({
        "group": ["GT", "non-GT"],
        "count": [len(gt_scores), len(ngt_scores)],
        "mean": [gt_scores.mean(), ngt_scores.mean()],
        "median": [gt_scores.median(), ngt_scores.median()],
        "std": [gt_scores.std(), ngt_scores.std()],
        "p25": [gt_scores.quantile(0.25), ngt_scores.quantile(0.25)],
        "p75": [gt_scores.quantile(0.75), ngt_scores.quantile(0.75)],
        "p90": [gt_scores.quantile(0.90), ngt_scores.quantile(0.90)],
        "p99": [gt_scores.quantile(0.99), ngt_scores.quantile(0.99)],
    })
    _save_csv(summary, layout.global_csv("score_distribution_summary.csv"), "GT vs non-GT score summary")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # overlapping histogram
    bins = np.linspace(0, 1, 41)
    axes[0].hist(ngt_scores, bins=bins, alpha=0.6, color="steelblue",
                 density=True, label=f"non-GT (n={len(ngt_scores):,})")
    axes[0].hist(gt_scores, bins=bins, alpha=0.8, color="crimson",
                 density=True, label=f"GT (n={len(gt_scores)})")
    axes[0].set_xlabel("LLM Score")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Score Distribution: GT vs non-GT")
    axes[0].legend()

    # box plot
    axes[1].boxplot([ngt_scores.sample(min(5000, len(ngt_scores)), random_state=42), gt_scores],
                    labels=["non-GT", "GT"], patch_artist=True,
                    boxprops=dict(facecolor="lightblue"))
    axes[1].set_ylabel("LLM Score")
    axes[1].set_title("Box Plot: GT vs non-GT")

    # violin
    parts = axes[2].violinplot(
        [ngt_scores.sample(min(5000, len(ngt_scores)), random_state=42).values, gt_scores.values],
        positions=[1, 2], showmedians=True
    )
    axes[2].set_xticks([1, 2])
    axes[2].set_xticklabels(["non-GT", "GT"])
    axes[2].set_ylabel("LLM Score")
    axes[2].set_title("Violin: GT vs non-GT")
    fig.suptitle("LLM Score Distribution: GT vs All Other Docs")
    _save_fig(fig, layout.chart_base("score_distributions.png"), "GT vs non-GT score distributions")

    return {"gt_mean_score": float(gt_scores.mean()), "ngt_mean_score": float(ngt_scores.mean()),
            "gt_median_score": float(gt_scores.median())}


# ── section 5: GT score percentile ───────────────────────────────────────────

def section_gt_percentile(pqs: pd.DataFrame, layout: AnalysisLayout) -> dict:
    pct_df = pqs[["query_idx", "gt_rank_pct"]].dropna().copy()
    _save_csv(pct_df, layout.per_query_csv("gt_score_percentile.csv"), "GT score percentile per query")

    vals = pct_df["gt_rank_pct"]
    above_50 = (vals >= 50).mean() * 100
    above_75 = (vals >= 75).mean() * 100
    above_90 = (vals >= 90).mean() * 100

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(vals, bins=20, color="mediumseagreen", edgecolor="white")
    ax.axvline(vals.median(), color="red", linestyle="--", label=f"median={vals.median():.1f}th pct")
    ax.set_xlabel("GT Score Percentile (100 = GT is highest scorer)")
    ax.set_ylabel("Queries")
    ax.set_title(f"GT Score Percentile Among All Docs Per Query\n"
                 f"above 50th: {above_50:.1f}%  above 75th: {above_75:.1f}%  above 90th: {above_90:.1f}%")
    ax.legend()
    _save_fig(fig, layout.chart_base("gt_score_percentile.png"), "GT score percentile histogram")

    return {"gt_pct_median": float(vals.median()), "pct_above_50": above_50,
            "pct_above_75": above_75, "pct_above_90": above_90}


# ── section 6: ROC / PR curves ───────────────────────────────────────────────

def section_roc_pr(df: pd.DataFrame, layout: AnalysisLayout) -> dict:
    if not HAS_SKLEARN:
        return {}

    labels = df["is_gt"].values
    scores = df["llm_score"].values

    # global ROC
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = roc_auc_score(labels, scores)

    # global PR
    prec, rec, _ = precision_recall_curve(labels, scores)
    ap = average_precision_score(labels, scores)
    baseline_pr = labels.mean()

    summary = pd.DataFrame({
        "metric": ["ROC-AUC", "Average Precision (PR-AUC)", "Baseline (random) PR"],
        "value": [roc_auc, ap, baseline_pr],
    })
    _save_csv(summary, layout.global_csv("roc_pr_summary.csv"), "Global ROC/PR summary")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(fpr, tpr, color="steelblue", label=f"ROC-AUC={roc_auc:.4f}")
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.4, label="random")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve — LLM as GT Detector")
    axes[0].legend()

    axes[1].plot(rec, prec, color="darkorange", label=f"AP={ap:.4f}")
    axes[1].axhline(baseline_pr, color="k", linestyle="--", alpha=0.4,
                    label=f"baseline={baseline_pr:.5f}")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve — LLM as GT Detector")
    axes[1].legend()
    fig.suptitle("LLM Score as Binary GT Classifier (global, all queries)")
    _save_fig(fig, layout.chart_base("roc_pr_curves.png"), "ROC and PR curves")

    return {"roc_auc": roc_auc, "average_precision": ap, "baseline_pr": baseline_pr}


# ── section 7: scatter (dist_to_gt vs llm_score) ─────────────────────────────

def section_scatter(df: pd.DataFrame, layout: AnalysisLayout) -> None:
    sample_n = 10_000
    ngt = df[df["is_gt"] == 0]
    gt  = df[df["is_gt"] == 1]
    ngt_sample = ngt.sample(min(sample_n, len(ngt)), random_state=42)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(ngt_sample["llm_score"], ngt_sample["dist_to_gt"],
               alpha=0.15, s=5, color="steelblue", label="non-GT")
    ax.scatter(gt["llm_score"], gt["dist_to_gt"],
               alpha=0.9, s=30, color="crimson", zorder=5, label="GT")
    ax.set_xlabel("LLM Score →  (higher = more relevant)")
    ax.set_ylabel("Distance to GT ↓  (lower = closer to GT)")
    ax.set_title(f"LLM Score vs Distance to GT\n"
                 f"(non-GT sample n={len(ngt_sample):,}, GT n={len(gt)})")
    ax.legend()
    _save_fig(fig, layout.chart_base("scatter_score_vs_dist.png"), "Scatter: LLM score vs dist-to-GT")


# ── section 8: distance quantile analysis ────────────────────────────────────

def section_distance_quantiles(df: pd.DataFrame, layout: AnalysisLayout) -> None:
    n_bins = 10
    ngt = df[df["is_gt"] == 0].copy()
    ngt["dist_bin"] = pd.qcut(ngt["dist_to_gt"], q=n_bins, labels=False, duplicates="drop")
    ngt["dist_bin_label"] = pd.qcut(ngt["dist_to_gt"], q=n_bins, duplicates="drop").astype(str)

    agg = (ngt.groupby("dist_bin")["llm_score"]
               .agg(["mean", "median", "std", "count"])
               .reset_index())
    labels = ngt.groupby("dist_bin")["dist_bin_label"].first().reset_index()
    agg = agg.merge(labels, on="dist_bin")
    _save_csv(agg, layout.global_csv("distance_quantile_analysis.csv"), "LLM score by dist quantile bucket")

    fig, ax = plt.subplots(figsize=(11, 4))
    xs = np.arange(len(agg))
    ax.bar(xs, agg["mean"], color="steelblue", alpha=0.7, label="Mean LLM Score")
    ax.plot(xs, agg["median"], "o-", color="darkorange", label="Median LLM Score")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"Q{i+1}" for i in xs], fontsize=8)
    ax.set_xlabel("Distance Decile (Q1=closest to GT, Q10=farthest)")
    ax.set_ylabel("LLM Score")
    ax.set_title("Mean/Median LLM Score by Distance-to-GT Decile\n"
                 "(non-GT docs only; expect decline from Q1→Q10 if LLM is useful)")
    ax.legend()
    _save_fig(fig, layout.chart_base("distance_quantile_analysis.png"), "LLM score by distance bucket")


# ── section 9: top-K Jaccard overlap ─────────────────────────────────────────

def section_topk_jaccard(df: pd.DataFrame, layout: AnalysisLayout) -> None:
    ks = [5, 10, 20, 50, 100]
    rows = []
    for qidx, grp in df.groupby("query_idx"):
        grp = grp.copy()
        top_dist = set(grp.nsmallest(max(ks), "dist_to_gt")["doc_id"])
        top_llm  = set(grp.nlargest(max(ks), "llm_score")["doc_id"])
        for k in ks:
            tk_dist = set(grp.nsmallest(k, "dist_to_gt")["doc_id"])
            tk_llm  = set(grp.nlargest(k, "llm_score")["doc_id"])
            inter = len(tk_dist & tk_llm)
            union = len(tk_dist | tk_llm)
            jaccard = inter / union if union > 0 else 0.0
            rows.append({"query_idx": qidx, "K": k, "jaccard": jaccard,
                         "intersection": inter})
    jac_df = pd.DataFrame(rows)
    _save_csv(jac_df, layout.per_query_csv("topk_jaccard.csv"), "Top-K Jaccard overlap per query")

    agg = jac_df.groupby("K")["jaccard"].describe().reset_index()
    _save_csv(agg, layout.global_csv("topk_jaccard_summary.csv"), "Top-K Jaccard summary")

    fig, ax = plt.subplots(figsize=(9, 4))
    data_by_k = [jac_df[jac_df["K"] == k]["jaccard"].values for k in ks]
    bp = ax.boxplot(data_by_k, labels=[str(k) for k in ks], patch_artist=True,
                    boxprops=dict(facecolor="lightsteelblue"))
    ax.set_xlabel("K")
    ax.set_ylabel("Jaccard Similarity")
    ax.set_title("Top-K Agreement: LLM Ranking vs Distance-to-GT Ranking\n"
                 "(Jaccard between top-K sets; 1.0 = perfect agreement)")
    _save_fig(fig, layout.chart_base("topk_jaccard.png"), "Top-K Jaccard box plot")


# ── section 10: score gap ────────────────────────────────────────────────────

def section_score_gap(pqs: pd.DataFrame, layout: AnalysisLayout) -> dict:
    gap_df = pqs[["query_idx", "gt_llm_score", "best_ngt_score", "score_gap"]].dropna().copy()
    _save_csv(gap_df, layout.per_query_csv("score_gap.csv"), "GT score gap vs best non-GT per query")

    gap = gap_df["score_gap"]
    pos_frac = (gap > 0).mean() * 100
    zero_frac = (gap == 0).mean() * 100

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(gap, bins=40, color="mediumorchid", edgecolor="white")
    ax.axvline(0, color="black", linestyle="--", linewidth=1.2)
    ax.axvline(gap.median(), color="red", linestyle="--",
               label=f"median={gap.median():.3f}")
    ax.set_xlabel("GT LLM Score − Best Non-GT LLM Score")
    ax.set_ylabel("Queries")
    ax.set_title(f"Score Gap: GT vs Best Non-GT\n"
                 f"GT is highest scorer in {pos_frac:.1f}% of queries | "
                 f"Tied in {zero_frac:.1f}%")
    ax.legend()
    _save_fig(fig, layout.chart_base("score_gap.png"), "Score gap histogram")

    return {"pct_gt_highest": pos_frac, "median_gap": float(gap.median()),
            "mean_gap": float(gap.mean())}


# ── section 11: per-query LLM score heatmap (top queries) ────────────────────

def section_per_query_heatmap(pqs: pd.DataFrame, layout: AnalysisLayout) -> None:
    """Heatmap: each row = query, columns = LLM rank bucket, color = GT rank."""
    rank_df = pqs[["query_idx", "gt_llm_rank", "n_docs"]].dropna().copy()
    rank_df["gt_rank_norm"] = rank_df["gt_llm_rank"] / rank_df["n_docs"]

    fig, ax = plt.subplots(figsize=(10, 5))
    sc = ax.scatter(
        rank_df["gt_llm_rank"].clip(upper=200),
        rank_df["query_idx"],
        c=rank_df["gt_rank_norm"],
        cmap="RdYlGn_r",
        s=30,
        vmin=0,
        vmax=1,
    )
    plt.colorbar(sc, ax=ax, label="Normalized GT LLM Rank (0=best)")
    ax.set_xlabel("GT LLM Rank (capped at 200)")
    ax.set_ylabel("Query Index")
    ax.set_title("Per-Query GT LLM Rank\n(green = GT near top, red = GT near bottom)")
    _save_fig(fig, layout.chart_base("per_query_gt_rank_scatter.png"), "Per-query GT rank scatter")


# ── section 12: LLM score vs query-level statistics ──────────────────────────

def section_query_level_overview(pqs: pd.DataFrame, layout: AnalysisLayout) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.ravel()

    # spearman_r vs gt_llm_rank
    valid = pqs.dropna(subset=["spearman_r", "gt_llm_rank"])
    axes[0].scatter(valid["spearman_r"], valid["gt_llm_rank"].clip(upper=200),
                    alpha=0.5, s=20, color="steelblue")
    axes[0].set_xlabel("Spearman r (llm_score vs dist_to_gt)")
    axes[0].set_ylabel("GT LLM Rank (capped 200)")
    axes[0].set_title("Correlation vs GT Rank\n(stronger neg corr → lower GT rank?)")

    # gt_llm_score vs gt_dist_to_gt
    axes[1].scatter(pqs["gt_dist_to_gt"], pqs["gt_llm_score"],
                    alpha=0.5, s=20, color="darkorange")
    axes[1].set_xlabel("GT dist_to_gt (should be ~0)")
    axes[1].set_ylabel("GT LLM Score")
    axes[1].set_title("GT Distance vs GT LLM Score")

    # score_gap vs gt_llm_rank
    valid2 = pqs.dropna(subset=["score_gap", "gt_llm_rank"])
    axes[2].scatter(valid2["score_gap"], valid2["gt_llm_rank"].clip(upper=200),
                    alpha=0.5, s=20, color="mediumseagreen")
    axes[2].axvline(0, color="red", linestyle="--", alpha=0.5)
    axes[2].set_xlabel("Score Gap (GT − best non-GT)")
    axes[2].set_ylabel("GT LLM Rank (capped 200)")
    axes[2].set_title("Score Gap vs GT Rank\n(positive gap → rank 1 expected)")

    # n_docs vs gt_llm_rank
    axes[3].scatter(pqs["n_docs"], pqs["gt_llm_rank"].clip(upper=200),
                    alpha=0.4, s=20, color="mediumpurple")
    axes[3].set_xlabel("N Docs in Pool (query)")
    axes[3].set_ylabel("GT LLM Rank (capped 200)")
    axes[3].set_title("Pool Size vs GT Rank")

    fig.suptitle("Query-Level Overview Scatter Plots")
    plt.tight_layout()
    _save_fig(fig, layout.chart_base("query_level_overview.png"), "4-panel query-level overview")


# ── section 13: MRR and P@K table ────────────────────────────────────────────

def section_mrr(pqs: pd.DataFrame, layout: AnalysisLayout) -> dict:
    ranks = pqs["gt_llm_rank"].dropna().astype(float)
    mrr = (1.0 / ranks).mean()
    ks = [1, 3, 5, 10, 20, 50, 100]
    rows = [{"metric": "MRR", "value": mrr}]
    for k in ks:
        rows.append({"metric": f"P@{k}", "value": (ranks <= k).mean()})
    mrr_df = pd.DataFrame(rows)
    _save_csv(mrr_df, layout.global_csv("mrr_and_precision_at_k.csv"), "MRR and P@K")
    return {"mrr": mrr}


# ── section 14: dist_to_gt distribution for GT docs ─────────────────────────

def section_gt_dist_sanity(df: pd.DataFrame, layout: AnalysisLayout) -> None:
    gt = df[df["is_gt"] == 1]
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.hist(gt["dist_to_gt"], bins=30, color="crimson", edgecolor="white")
    ax.set_xlabel("dist_to_gt for GT doc")
    ax.set_ylabel("Count")
    ax.set_title(f"GT Doc Distance to GT Embedding (should be ~0)\n"
                 f"mean={gt['dist_to_gt'].mean():.4f}  max={gt['dist_to_gt'].max():.4f}")
    _save_fig(fig, layout.chart_base("gt_dist_sanity.png"), "GT doc distance sanity check")


# ── section 15: score by dist_to_gt for GT docs across queries ──────────────

def section_gt_scores_scatter(pqs: pd.DataFrame, layout: AnalysisLayout) -> None:
    valid = pqs.dropna(subset=["gt_llm_score", "gt_dist_to_gt"])
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(valid["gt_llm_score"], valid["gt_dist_to_gt"],
               alpha=0.6, s=25, color="crimson")
    ax.set_xlabel("GT LLM Score")
    ax.set_ylabel("GT dist_to_gt (should be ~0)")
    ax.set_title("Per-Query GT: LLM Score vs Distance\n"
                 "(distance should cluster near 0; score variation shows LLM confidence in GT)")
    _save_fig(fig, layout.chart_base("gt_score_vs_dist.png"), "GT LLM score vs GT distance")


# ── summary report ───────────────────────────────────────────────────────────

def write_summary(
    stats_dict: dict,
    pqs: pd.DataFrame,
    df: pd.DataFrame,
    layout: AnalysisLayout,
    health: dict | None = None,
) -> None:
    n_queries = pqs["query_idx"].nunique()
    n_rows = len(df)
    rag_fail_pct = (pqs["rag_failed"] == 1).mean() * 100 if "rag_failed" in pqs.columns else 100.0

    ranks_min = pqs["gt_llm_rank"].dropna().astype(float)
    ranks_avg = pqs["gt_llm_rank_avg"].dropna().astype(float)
    mrr_optimistic = (1.0 / ranks_min).mean()
    mrr_corrected = (1.0 / ranks_avg).mean()
    p1_optimistic = (ranks_min == 1).mean() * 100
    p1_corrected = (ranks_avg <= 1).mean() * 100
    n_tied_median = float(stats_dict.get("median_n_tied_at_gt", float("nan")))
    if n_tied_median != n_tied_median:  # NaN — compute quick tie estimate
        tie_sizes = []
        for _, grp in df.groupby("query_idx"):
            gt_row = grp[grp["is_gt"] == 1]
            if gt_row.empty:
                continue
            gt_score = float(gt_row["llm_score"].iloc[0])
            if (grp["llm_score"] > gt_score).any():
                continue
            tie_sizes.append(int((grp["llm_score"] == gt_score).sum()))
        n_tied_median = float(np.median(tie_sizes)) if tie_sizes else float("nan")

    lines = [
        "=" * 70,
        "GLOBAL CORRELATION EXPERIMENT — ANALYSIS SUMMARY",
        "=" * 70,
        "",
    ]
    if health:
        lines.extend(format_health_text_lines(health))
        lines.append("")
    lines.extend([
        "DATASET",
        f"  Scored pairs      : {n_rows:,}",
        f"  Queries           : {n_queries}",
        f"  GT rows           : {int(df['is_gt'].sum())}",
        f"  Retriever failed  : {rag_fail_pct:.1f}% of queries (rag_failed=1)",
        f"  Avg LLM score     : {df['llm_score'].mean():.4f}",
        f"  Avg dist_to_gt    : {df['dist_to_gt'].mean():.4f}",
        "",
        "CORRELATION (llm_score vs dist_to_gt, expect negative; vs is_gt, expect positive)",
        f"  Median Spearman r     : {stats_dict.get('median_spearman_r', 'n/a'):.4f}",
        f"  Median Pearson r      : {stats_dict.get('median_pearson_r', 'n/a'):.4f}",
        f"  Median Kendall tau-b  : {stats_dict.get('median_kendall_tau', 'n/a'):.4f}",
        f"  Median Point-biserial : {stats_dict.get('median_pb_r', 'n/a'):.4f}  (score vs is_gt)",
        f"  % negative Spearman r : {stats_dict.get('pct_negative_r', 0):.1f}%",
        f"  % significant r       : {stats_dict.get('pct_significant_r', 0):.1f}%  (p<0.05)",
        "",
        "LLM AS GT DETECTOR",
        f"  ROC-AUC           : {stats_dict.get('roc_auc', 'n/a (sklearn missing)')}",
        f"  Avg Precision     : {stats_dict.get('average_precision', 'n/a')}",
        "",
        "LLM RANK OF GT (rank by llm_score DESC; 1 = best)",
        f"  MRR (optimistic)      : {mrr_optimistic:.4f}  (rank_min — ties share rank 1)",
        f"  MRR (tie-corrected)   : {mrr_corrected:.4f}  (rank_avg — see extended analysis)",
        f"  P@1 (optimistic)      : {p1_optimistic:.1f}%",
        f"  P@1 (tie-corrected)   : {p1_corrected:.1f}%",
        f"  P@3               : {stats_dict.get('p_at_3', 0):.1f}%",
        f"  P@5               : {stats_dict.get('p_at_5', 0):.1f}%",
        f"  P@10              : {stats_dict.get('p_at_10', 0):.1f}%",
        "",
        "LLM RECOVERY (same as P@K when rag_failed=100%; retriever missed GT on all queries)",
        f"  Recovery@1        : {stats_dict.get('recovery_at_1', 0):.1f}%  (GT in top tier, not unique)",
        f"  Recovery@5        : {stats_dict.get('recovery_at_5', 0):.1f}%",
        f"  Recovery@10       : {stats_dict.get('recovery_at_10', 0):.1f}%",
        "",
        "SCORE DISTRIBUTIONS",
        f"  GT mean score     : {stats_dict.get('gt_mean_score', 0):.4f}",
        f"  non-GT mean score : {stats_dict.get('ngt_mean_score', 0):.4f}",
        f"  GT median score   : {stats_dict.get('gt_median_score', 0):.4f}",
        "",
        "GT SCORE PERCENTILE (100 = GT has highest score in query)",
        f"  Median percentile : {stats_dict.get('gt_pct_median', 0):.1f}",
        f"  > 50th percentile : {stats_dict.get('pct_above_50', 0):.1f}%",
        f"  > 75th percentile : {stats_dict.get('pct_above_75', 0):.1f}%",
        f"  > 90th percentile : {stats_dict.get('pct_above_90', 0):.1f}%",
        "",
        "SCORE GAP (GT score − best non-GT score per query)",
        f"  GT is highest scorer : {stats_dict.get('pct_gt_highest', 0):.1f}% of queries",
        f"  Median gap        : {stats_dict.get('median_gap', 0):.4f}",
        f"  Mean gap          : {stats_dict.get('mean_gap', 0):.4f}",
        "",
        "CONCLUSION",
        textwrap.fill(
            "When the retriever fails to rank GT at position 1 (which happens in "
            f"{rag_fail_pct:.0f}% of queries here), the LLM is a strong coarse filter but "
            "not a precise ranker. Optimistic P@1="
            f"{p1_optimistic:.1f}% only means GT reaches the top score tier (often tied "
            f"with a median of {n_tied_median:.0f} other docs); tie-corrected P@1="
            f"{p1_corrected:.1f}% is the honest single-doc pick rate. "
            f"MRR drops from {mrr_optimistic:.3f} (optimistic) to {mrr_corrected:.3f} "
            "(tie-corrected). Run analyze_experiment_extended.py for full tie analysis. "
            f"Spearman median r={stats_dict.get('median_spearman_r', 0):.3f} shows LLM score "
            "aligns with proximity to GT (negative = aligned).",
            width=70,
        ),
        "",
        "=" * 70,
    ])

    report_path = layout.summary_txt("analysis_summary.txt")
    report_path.write_text("\n".join(lines))
    print(f"  [txt]   {report_path.name}")


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze global correlation experiment DB")
    parser.add_argument("db_path", nargs="?", help="Path to experiment_results.db")
    parser.add_argument("--run-dir", help="Path to run directory (DB auto-located inside)")
    args = parser.parse_args()

    if args.run_dir:
        db_path = Path(args.run_dir) / "experiment_results.db"
    elif args.db_path:
        db_path = Path(args.db_path)
    else:
        db_path = find_latest_db()

    if not db_path.exists():
        print(f"[ERROR] DB not found: {db_path}", file=sys.stderr)
        sys.exit(1)

    run_dir = db_path.parent
    layout = AnalysisLayout(run_dir / ANALYSIS_DIR).ensure()
    print(f"Run dir : {run_dir}")
    print(f"Output  : {layout.root}")
    print()

    health = collect_run_health(db_path)
    save_run_health(health, layout.root)
    if health["warnings"]:
        print("Run health warnings:")
        for w in health["warnings"]:
            print(f"  • {w}")
        print()

    df = _load_dataframe(db_path)
    print("\nBuilding per-query statistics …")
    pqs = _build_per_query_stats(df)
    print(f"  {len(pqs)} queries processed\n")

    all_stats: dict = {}

    print("── Section 1: Correlation ─────────────────────────────")
    all_stats.update(section_correlation(pqs, layout))

    print("── Section 2: GT LLM Rank ─────────────────────────────")
    all_stats.update(section_gt_rank(pqs, layout))

    print("── Section 3: Recovery Rate ───────────────────────────")
    all_stats.update(section_recovery(pqs, layout))

    print("── Section 4: Score Distributions ────────────────────")
    all_stats.update(section_score_distribution(df, layout))

    print("── Section 5: GT Score Percentile ────────────────────")
    all_stats.update(section_gt_percentile(pqs, layout))

    print("── Section 6: ROC / PR Curves ────────────────────────")
    all_stats.update(section_roc_pr(df, layout))

    print("── Section 7: Scatter (score vs dist) ────────────────")
    section_scatter(df, layout)

    print("── Section 8: Distance Quantile Analysis ─────────────")
    section_distance_quantiles(df, layout)

    print("── Section 9: Top-K Jaccard Overlap ──────────────────")
    section_topk_jaccard(df, layout)

    print("── Section 10: Score Gap ──────────────────────────────")
    all_stats.update(section_score_gap(pqs, layout))

    print("── Section 11: Per-Query GT Rank Scatter ─────────────")
    section_per_query_heatmap(pqs, layout)

    print("── Section 12: Query-Level Overview ──────────────────")
    section_query_level_overview(pqs, layout)

    print("── Section 13: MRR and P@K ───────────────────────────")
    all_stats.update(section_mrr(pqs, layout))

    print("── Section 14: GT Distance Sanity ────────────────────")
    section_gt_dist_sanity(df, layout)

    print("── Section 15: GT Score vs Distance ──────────────────")
    section_gt_scores_scatter(pqs, layout)

    print("── Summary Report ────────────────────────────────────")
    write_summary(all_stats, pqs, df, layout, health=health)

    build_per_query_overview(layout.root)
    write_analysis_index(layout.root, health)

    print(f"\nDone. All outputs in: {layout.root}")
    print(f"  Index : {layout.index_md()}")


if __name__ == "__main__":
    main()
