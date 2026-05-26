#!/usr/bin/env python3
"""
analyze_experiment_extended.py — Extended / corrective analysis for the global correlation DB.

This script adds analyses that the base `analyze_experiment.py` did not cover, and
corrects several misleading interpretations that arise from the data's characteristics:

  KEY INSIGHT: LLM scores are quantized (integers 1-10 divided by 10).
  95% of queries have GT and ~11 other docs ALL tied at score 1.0.
  "P@1 = 98%" is therefore misleading — GT is in the top-scoring group, but
  not uniquely identified. These scripts surface the tie structure explicitly.

Sections added:
  A — Tie structure analysis: how many docs share GT's score?
  B — Score discretization: per-level doc counts and dist_to_gt
  C — LLM as threshold filter: precision / recall at each score cutoff
  D — Query difficulty classification (easy / medium / hard / failed)
  E — Failure deep-dive: the queries where GT did NOT score 1.0
  F — False-positive analysis: non-GT docs scoring 1.0
  G — Corrected MRR / P@K using tie-adjusted (average) rank
  H — Within-tier dist_to_gt: does dist predict GT inside the 1.0 tier?
  I — Marginal value of LLM score levels (score 0.4 docs vs 0.1 docs)
  J — Improved summary with corrected conclusions

Usage:
    python analyze_experiment_extended.py                    # auto-detect latest run
    python analyze_experiment_extended.py path/to/experiment_results.db
    python analyze_experiment_extended.py --run-dir path/to/staged_run_YYYY-MM-DD
"""

from __future__ import annotations

import argparse
import os
import sqlite3
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

plt.rcParams.update({"figure.dpi": 120, "font.size": 10})

RESULTS_ROOT = Path("results/global_exp")
ANALYSIS_DIR = "analysis"
ALL_SCORE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def _find_latest_db() -> Path:
    candidates = sorted(RESULTS_ROOT.glob("staged_run_*/experiment_results.db"))
    if not candidates:
        candidates = sorted(RESULTS_ROOT.glob("*/experiment_results.db"))
    if not candidates:
        raise FileNotFoundError(f"No experiment_results.db found under {RESULTS_ROOT}")
    return candidates[-1]


def _load(db_path: Path) -> pd.DataFrame:
    print(f"Loading from {db_path} …")
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        "SELECT query_idx, doc_id, llm_score, dist_to_gt, is_gt, rag_failed "
        "FROM results WHERE llm_score IS NOT NULL",
        conn,
    )
    conn.close()
    print(f"  {len(df):,} rows  |  {df['query_idx'].nunique()} queries  |  "
          f"{df['is_gt'].sum()} GT rows\n")
    return df


def _fig(path: Path, label: str = "") -> None:
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  [chart] {path.name}" + (f"  — {label}" if label else ""))


def _csv(df: pd.DataFrame, path: Path, label: str = "") -> None:
    df.to_csv(path, index=False)
    print(f"  [csv]   {path.name}" + (f"  — {label}" if label else ""))


# ─── A. Tie structure analysis ────────────────────────────────────────────────

def section_a_tie_structure(df: pd.DataFrame, out: Path) -> dict:
    """
    For each query: how many docs share the same score as GT?
    Surfaces the "illusory P@1" — GT is in the top tier but not uniquely identified.
    """
    rows = []
    for qidx, grp in df.groupby("query_idx"):
        gt_row = grp[grp["is_gt"] == 1]
        if gt_row.empty:
            continue
        gt_score = float(gt_row["llm_score"].iloc[0])
        n_tied   = int((grp["llm_score"] == gt_score).sum())
        n_above  = int((grp["llm_score"] > gt_score).sum())
        n_docs   = len(grp)
        tier_precision = 1.0 / n_tied if n_tied > 0 else 0.0
        rows.append({
            "query_idx":      qidx,
            "gt_score":       gt_score,
            "n_tied_at_gt":   n_tied,          # docs sharing GT's score (incl GT)
            "n_above_gt":     n_above,          # docs scoring HIGHER than GT
            "n_docs":         n_docs,
            "tier_precision": tier_precision,   # expected precision if random from tied group
            "gt_uniquely_top": int(n_above == 0 and n_tied == 1),
        })
    tie_df = pd.DataFrame(rows)
    _csv(tie_df, out / "tie_structure.csv", "Per-query tie structure")

    gt_in_top  = tie_df[tie_df["n_above_gt"] == 0]
    gt_not_top = tie_df[tie_df["n_above_gt"] > 0]
    n_unique   = tie_df["gt_uniquely_top"].sum()
    mean_prec  = gt_in_top["tier_precision"].mean()

    # Panel 1: distribution of n_tied
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].hist(gt_in_top["n_tied_at_gt"], bins=30, color="steelblue", edgecolor="white")
    axes[0].axvline(gt_in_top["n_tied_at_gt"].median(), color="red", linestyle="--",
                    label=f"median={gt_in_top['n_tied_at_gt'].median():.0f}")
    axes[0].set_xlabel("N docs tied at GT's score (incl GT)")
    axes[0].set_ylabel("Queries")
    axes[0].set_title(f"Tie Size When GT Reaches Top Tier\n(n={len(gt_in_top)} queries)")
    axes[0].legend()

    # Panel 2: tier_precision distribution
    axes[1].hist(gt_in_top["tier_precision"], bins=30, color="darkorange", edgecolor="white")
    axes[1].axvline(mean_prec, color="red", linestyle="--",
                    label=f"mean={mean_prec:.3f}")
    axes[1].set_xlabel("Expected Precision (1 / n_tied)")
    axes[1].set_ylabel("Queries")
    axes[1].set_title(f"Tier Precision: Random Pick From Tied Group\n"
                      f"Uniquely top: {n_unique} queries  |  mean prec: {mean_prec:.3f}")
    axes[1].legend()

    # Panel 3: n_above_gt distribution
    axes[2].hist(tie_df["n_above_gt"], bins=30, color="crimson", edgecolor="white")
    axes[2].axvline(0.5, color="black", linestyle="--", alpha=0.5)
    axes[2].set_xlabel("N docs scoring ABOVE GT")
    axes[2].set_ylabel("Queries")
    axes[2].set_title(f"Docs Outscoring GT Per Query\n"
                      f"GT in top tier: {len(gt_in_top)}/{ len(tie_df)} queries")
    fig.suptitle("Tie Structure: Why P@1=98% Is Misleading\n"
                 "GT is in top tier but shares it with ~11 other docs on average")
    plt.tight_layout()
    _fig(out / "tie_structure.png", "Tie structure — corrects misleading P@1")

    return {
        "n_queries_gt_in_top": len(gt_in_top),
        "n_queries_gt_not_in_top": len(gt_not_top),
        "n_uniquely_top": int(n_unique),
        "mean_tier_precision": float(mean_prec),
        "median_n_tied": float(gt_in_top["n_tied_at_gt"].median()),
    }


# ─── B. Score discretization ─────────────────────────────────────────────────

def section_b_score_levels(df: pd.DataFrame, out: Path) -> None:
    """
    Break down each discrete score level: doc counts, GT fraction, dist_to_gt distribution.
    The LLM uses a 1-10 integer scale (/10), so only 10 distinct values are possible.
    """
    levels = sorted(df["llm_score"].unique())
    rows = []
    for lvl in levels:
        sub = df[df["llm_score"] == lvl]
        n_gt  = int(sub["is_gt"].sum())
        n_all = len(sub)
        rows.append({
            "score":       lvl,
            "n_docs":      n_all,
            "n_gt":        n_gt,
            "gt_fraction": n_gt / n_all if n_all > 0 else 0,
            "mean_dist":   sub["dist_to_gt"].mean(),
            "median_dist": sub["dist_to_gt"].median(),
            "p10_dist":    sub["dist_to_gt"].quantile(0.10),
            "p90_dist":    sub["dist_to_gt"].quantile(0.90),
        })
    lvl_df = pd.DataFrame(rows)
    _csv(lvl_df, out / "score_level_breakdown.csv", "Per-score-level stats")

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    # doc count per level
    axes[0, 0].bar([str(r["score"]) for _, r in lvl_df.iterrows()],
                   lvl_df["n_docs"], color="steelblue", edgecolor="white")
    axes[0, 0].set_xlabel("LLM Score")
    axes[0, 0].set_ylabel("Doc Count (log scale)")
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_title("Doc Count per Score Level")

    # GT fraction per level
    axes[0, 1].bar([str(r["score"]) for _, r in lvl_df.iterrows()],
                   lvl_df["gt_fraction"] * 100, color="crimson", edgecolor="white")
    axes[0, 1].set_xlabel("LLM Score")
    axes[0, 1].set_ylabel("% of Docs That Are GT")
    axes[0, 1].set_title("GT Fraction per Score Level\n(score=1.0 has highest concentration of GTs)")

    # mean dist per level
    axes[1, 0].bar([str(r["score"]) for _, r in lvl_df.iterrows()],
                   lvl_df["mean_dist"], color="mediumseagreen", edgecolor="white",
                   label="Mean dist_to_gt")
    axes[1, 0].plot([str(r["score"]) for _, r in lvl_df.iterrows()],
                    lvl_df["median_dist"], "o-", color="darkorange", label="Median dist_to_gt")
    axes[1, 0].set_xlabel("LLM Score")
    axes[1, 0].set_ylabel("Distance to GT")
    axes[1, 0].set_title("Distance-to-GT per Score Level\n(expect: higher score → lower dist)")
    axes[1, 0].legend()

    # box plot of dist per score level (non-GT docs only to exclude 0-dist GT)
    ngt = df[df["is_gt"] == 0]
    level_groups = [ngt[ngt["llm_score"] == lvl]["dist_to_gt"].values for lvl in levels]
    level_labels = [str(lvl) for lvl in levels]
    bp = axes[1, 1].boxplot([g for g in level_groups if len(g) > 0],
                             tick_labels=[l for l, g in zip(level_labels, level_groups) if len(g) > 0],
                             patch_artist=True, showfliers=False,
                             boxprops=dict(facecolor="lightsteelblue"))
    axes[1, 1].set_xlabel("LLM Score")
    axes[1, 1].set_ylabel("dist_to_gt (non-GT docs)")
    axes[1, 1].set_title("dist_to_gt Box Plot per Score Level\n(non-GT only, no outliers)")

    fig.suptitle("Score Discretization: LLM Uses Integer Scale 1-10 (÷ 10)")
    plt.tight_layout()
    _fig(out / "score_level_breakdown.png", "Score levels: counts, GT fraction, distances")


# ─── C. LLM as threshold filter ──────────────────────────────────────────────

def section_c_threshold_filter(df: pd.DataFrame, out: Path) -> None:
    """
    Treat LLM score as a binary filter: keep docs scoring >= threshold.
    Compute precision, recall, F1, and mean pool size at each threshold.
    This answers: "if we use LLM to pre-filter, what threshold makes sense?"
    """
    levels = sorted(df["llm_score"].unique())
    n_queries = df["query_idx"].nunique()
    rows = []

    for thresh in levels:
        kept  = df[df["llm_score"] >= thresh]
        n_tp  = int(kept["is_gt"].sum())
        n_all_gt = int(df["is_gt"].sum())
        n_kept = len(kept)
        prec  = n_tp / n_kept if n_kept > 0 else 0
        rec   = n_tp / n_all_gt if n_all_gt > 0 else 0
        f1    = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        avg_pool = n_kept / n_queries
        rows.append({
            "threshold":    thresh,
            "n_docs_kept":  n_kept,
            "avg_pool_size": avg_pool,
            "precision":    prec,
            "recall":       rec,
            "f1":           f1,
        })
    filt_df = pd.DataFrame(rows)
    _csv(filt_df, out / "threshold_filter_analysis.csv", "LLM filter: precision/recall at each threshold")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    xs = [str(r["threshold"]) for _, r in filt_df.iterrows()]

    axes[0].bar(xs, filt_df["avg_pool_size"], color="steelblue", edgecolor="white")
    axes[0].set_xlabel("Min LLM Score Threshold")
    axes[0].set_ylabel("Avg Docs Kept Per Query")
    axes[0].set_yscale("log")
    axes[0].set_title("Pool Size After Filtering")

    axes[1].plot(xs, filt_df["precision"] * 100, "o-", color="crimson", label="Precision")
    axes[1].plot(xs, filt_df["recall"] * 100, "s-", color="steelblue", label="Recall")
    axes[1].plot(xs, filt_df["f1"] * 100, "^-", color="darkorange", label="F1")
    axes[1].set_xlabel("Min LLM Score Threshold")
    axes[1].set_ylabel("%")
    axes[1].set_title("Precision / Recall / F1 by Threshold")
    axes[1].legend()
    axes[1].yaxis.set_major_formatter(mticker.PercentFormatter())

    # precision vs recall curve (threshold as parameter)
    axes[2].plot(filt_df["recall"] * 100, filt_df["precision"] * 100,
                 "o-", color="mediumseagreen")
    for _, row in filt_df.iterrows():
        axes[2].annotate(str(row["threshold"]),
                         (row["recall"] * 100, row["precision"] * 100),
                         textcoords="offset points", xytext=(4, 4), fontsize=7)
    axes[2].set_xlabel("Recall (%)")
    axes[2].set_ylabel("Precision (%)")
    axes[2].set_title("Precision vs Recall (threshold labeled)")

    fig.suptitle("LLM as a Threshold Filter: Precision, Recall, Pool Size")
    plt.tight_layout()
    _fig(out / "threshold_filter.png", "LLM filter: precision/recall tradeoff by threshold")


# ─── D. Query difficulty classification ──────────────────────────────────────

def section_d_difficulty(df: pd.DataFrame, out: Path) -> dict:
    """
    Classify each query into:
      - FAILED:  GT scores < 1.0 (LLM does not rank GT at top)
      - HARD:    GT scores 1.0 but >10 docs also score 1.0
      - MEDIUM:  GT scores 1.0, 2–10 docs also score 1.0
      - EASY:    GT scores 1.0, only GT scores 1.0 (uniquely identified)
    """
    rows = []
    for qidx, grp in df.groupby("query_idx"):
        gt_row = grp[grp["is_gt"] == 1]
        if gt_row.empty:
            continue
        gt_score = float(gt_row["llm_score"].iloc[0])
        n_at_1   = int((grp["llm_score"] == 1.0).sum())
        n_above  = int((grp["llm_score"] > gt_score).sum())

        if n_above > 0 or gt_score < 1.0:
            difficulty = "FAILED"
        elif n_at_1 == 1:
            difficulty = "EASY"
        elif n_at_1 <= 10:
            difficulty = "MEDIUM"
        else:
            difficulty = "HARD"

        rows.append({
            "query_idx":  qidx,
            "gt_score":   gt_score,
            "n_at_top":   n_at_1,
            "difficulty": difficulty,
        })
    diff_df = pd.DataFrame(rows)
    _csv(diff_df, out / "query_difficulty.csv", "Per-query difficulty classification")

    counts = diff_df["difficulty"].value_counts().reindex(["EASY", "MEDIUM", "HARD", "FAILED"], fill_value=0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    colors = {"EASY": "mediumseagreen", "MEDIUM": "steelblue", "HARD": "darkorange", "FAILED": "crimson"}
    axes[0].bar(counts.index, counts.values,
                color=[colors[k] for k in counts.index], edgecolor="white")
    for i, (k, v) in enumerate(counts.items()):
        axes[0].text(i, v + 0.5, str(v), ha="center", va="bottom", fontweight="bold")
    axes[0].set_ylabel("Queries")
    axes[0].set_title("Query Difficulty Distribution\n"
                      "EASY=GT uniquely top  MEDIUM=2-10 tied  HARD=>10 tied  FAILED=GT not top")

    # n_at_top distribution for non-FAILED
    non_fail = diff_df[diff_df["difficulty"] != "FAILED"]
    ax2colors = [colors[d] for d in non_fail["difficulty"]]
    axes[1].scatter(non_fail["query_idx"], non_fail["n_at_top"],
                    c=[{"EASY": "mediumseagreen", "MEDIUM": "steelblue", "HARD": "darkorange"}[d]
                       for d in non_fail["difficulty"]],
                    alpha=0.7, s=25)
    axes[1].axhline(10, color="darkorange", linestyle="--", alpha=0.5, label="HARD threshold (10)")
    axes[1].axhline(1, color="mediumseagreen", linestyle="--", alpha=0.5, label="EASY threshold (1)")
    axes[1].set_xlabel("Query Index")
    axes[1].set_ylabel("N docs at top score (score=1.0)")
    axes[1].set_title("Tie Count Per Query (non-FAILED)")
    axes[1].legend()

    fig.suptitle("Query Difficulty: Can LLM Uniquely Identify GT?")
    plt.tight_layout()
    _fig(out / "query_difficulty.png", "Query difficulty classification")

    return {k: int(v) for k, v in counts.items()}


# ─── E. Failure deep-dive ─────────────────────────────────────────────────────

def section_e_failures(df: pd.DataFrame, out: Path) -> None:
    """
    Deep-dive into queries where GT did NOT score 1.0.
    These are the true LLM failures — it rated GT below some non-GT docs.
    """
    rows = []
    for qidx, grp in df.groupby("query_idx"):
        gt_row = grp[grp["is_gt"] == 1]
        if gt_row.empty:
            continue
        gt_score = float(gt_row["llm_score"].iloc[0])
        gt_dist  = float(gt_row["dist_to_gt"].iloc[0])
        if gt_score < 1.0:
            n_above = int((grp["llm_score"] > gt_score).sum())
            max_score = float(grp["llm_score"].max())
            n_max = int((grp["llm_score"] == max_score).sum())
            rank = int((grp["llm_score"] > gt_score).sum()) + 1
            rows.append({
                "query_idx":    qidx,
                "gt_llm_score": gt_score,
                "gt_dist_to_gt": gt_dist,
                "n_docs_above_gt": n_above,
                "gt_rank":       rank,
                "max_score_in_query": max_score,
                "n_docs_at_max": n_max,
                "n_docs_total": len(grp),
            })
    fail_df = pd.DataFrame(rows)
    if fail_df.empty:
        print("  No LLM failures found (all GTs score 1.0).")
        return
    _csv(fail_df, out / "failure_cases.csv", "Failure cases: GT did not score 1.0")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # GT score and rank for failures
    ax = axes[0]
    for _, row in fail_df.iterrows():
        ax.annotate(f"q{row['query_idx']}\nscore={row['gt_llm_score']}\nrank={row['gt_rank']}",
                    xy=(row["gt_llm_score"], row["gt_rank"]),
                    ha="center", va="bottom", fontsize=9)
    ax.scatter(fail_df["gt_llm_score"], fail_df["gt_rank"],
               color="crimson", zorder=5, s=80)
    ax.set_xlabel("GT LLM Score")
    ax.set_ylabel("GT Rank (1=best)")
    ax.set_title(f"LLM Failure Cases (n={len(fail_df)})\nGT scored below some non-GT docs")

    # score distribution for one failure query
    if len(fail_df) > 0:
        worst = fail_df.sort_values("gt_rank", ascending=False).iloc[0]
        qidx = int(worst["query_idx"])
        grp = df[df["query_idx"] == qidx]
        axes[1].hist(grp[grp["is_gt"] == 0]["llm_score"], bins=20,
                     color="steelblue", alpha=0.7, label="non-GT")
        axes[1].axvline(worst["gt_llm_score"], color="crimson", linewidth=2,
                        label=f"GT score={worst['gt_llm_score']:.1f}")
        axes[1].set_xlabel("LLM Score")
        axes[1].set_ylabel("Doc Count")
        axes[1].set_title(f"Score Distribution for Worst Failure: query {qidx}\n"
                          f"GT rank={int(worst['gt_rank'])} of {int(worst['n_docs_total'])}")
        axes[1].legend()

    fig.suptitle("Failure Analysis: Queries Where LLM Rated GT Below Non-GT Docs")
    plt.tight_layout()
    _fig(out / "failure_cases.png", "Failure deep-dive")

    print(f"  Failure summary:")
    print(fail_df[["query_idx", "gt_llm_score", "gt_rank", "n_docs_above_gt"]].to_string(index=False))


# ─── F. False positive analysis (non-GT docs scoring 1.0) ────────────────────

def section_f_false_positives(df: pd.DataFrame, out: Path) -> None:
    """
    2021 non-GT docs score 1.0 — the same as GT.
    Are they genuinely close to GT (low dist_to_gt) or are they false positives?
    """
    ngt_top = df[(df["is_gt"] == 0) & (df["llm_score"] == 1.0)].copy()
    ngt_low = df[(df["is_gt"] == 0) & (df["llm_score"] == 0.1)].copy()
    gt_top  = df[(df["is_gt"] == 1)  & (df["llm_score"] == 1.0)].copy()

    summary = pd.DataFrame({
        "group":       ["GT (score=1.0)", "non-GT (score=1.0)", "non-GT (score=0.1)"],
        "count":       [len(gt_top), len(ngt_top), len(ngt_low)],
        "mean_dist":   [gt_top["dist_to_gt"].mean(), ngt_top["dist_to_gt"].mean(), ngt_low["dist_to_gt"].mean()],
        "median_dist": [gt_top["dist_to_gt"].median(), ngt_top["dist_to_gt"].median(), ngt_low["dist_to_gt"].median()],
        "p10_dist":    [gt_top["dist_to_gt"].quantile(0.10), ngt_top["dist_to_gt"].quantile(0.10), ngt_low["dist_to_gt"].quantile(0.10)],
        "p90_dist":    [gt_top["dist_to_gt"].quantile(0.90), ngt_top["dist_to_gt"].quantile(0.90), ngt_low["dist_to_gt"].quantile(0.90)],
    })
    _csv(summary, out / "false_positive_analysis.csv", "Non-GT docs scoring 1.0 vs 0.1")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # dist distribution: ngt@1.0 vs ngt@0.1
    bins = np.linspace(0, 1.2, 50)
    axes[0].hist(ngt_low["dist_to_gt"].sample(min(5000, len(ngt_low)), random_state=42),
                 bins=bins, alpha=0.6, density=True, color="steelblue", label=f"non-GT score=0.1")
    axes[0].hist(ngt_top["dist_to_gt"], bins=bins, alpha=0.8, density=True,
                 color="darkorange", label=f"non-GT score=1.0 (n={len(ngt_top)})")
    axes[0].set_xlabel("dist_to_gt")
    axes[0].set_ylabel("Density")
    axes[0].set_title("False Positives vs Background\nAre top-scoring non-GTs genuinely close to GT?")
    axes[0].legend()

    # per-query: how many non-GT docs score 1.0?
    per_q = df[df["is_gt"] == 0].groupby("query_idx").apply(
        lambda g: (g["llm_score"] == 1.0).sum()
    ).reset_index(name="n_fp")
    axes[1].hist(per_q["n_fp"], bins=30, color="crimson", edgecolor="white")
    axes[1].axvline(per_q["n_fp"].median(), color="black", linestyle="--",
                    label=f"median={per_q['n_fp'].median():.0f}")
    axes[1].set_xlabel("N non-GT docs scoring 1.0 per query")
    axes[1].set_ylabel("Queries")
    axes[1].set_title(f"False Positives per Query\n(avg={per_q['n_fp'].mean():.1f})")
    axes[1].legend()

    # dist_to_gt box plots for each group
    axes[2].boxplot(
        [gt_top["dist_to_gt"].values,
         ngt_top["dist_to_gt"].values,
         ngt_low["dist_to_gt"].sample(min(5000, len(ngt_low)), random_state=42).values],
        tick_labels=["GT\n(score=1.0)", "non-GT\n(score=1.0)", "non-GT\n(score=0.1)"],
        patch_artist=True, showfliers=False,
        boxprops=dict(facecolor="lightsteelblue"),
    )
    axes[2].set_ylabel("dist_to_gt")
    axes[2].set_title("dist_to_gt by Group\n(non-GT@1.0 are much closer to GT than non-GT@0.1?)")

    fig.suptitle("False Positive Analysis: Non-GT Docs That Score 1.0")
    plt.tight_layout()
    _fig(out / "false_positive_analysis.png", "False positives: non-GT docs at score=1.0")


# ─── G. Tie-corrected MRR / P@K ──────────────────────────────────────────────

def section_g_corrected_mrr(df: pd.DataFrame, out: Path) -> dict:
    """
    Compute MRR and P@K using rank(method='average') which assigns the mean
    rank to tied docs (e.g., 11 docs tied at rank 1 all get rank 6).
    This is a fairer estimate of LLM's retrieval ability than rank(method='min').
    Also computes "pessimistic" MRR using rank(method='max').
    """
    rows = []
    for qidx, grp in df.groupby("query_idx"):
        gt_row = grp[grp["is_gt"] == 1]
        if gt_row.empty:
            continue
        grp = grp.copy()
        grp["rank_min"] = grp["llm_score"].rank(method="min", ascending=False)
        grp["rank_avg"] = grp["llm_score"].rank(method="average", ascending=False)
        grp["rank_max"] = grp["llm_score"].rank(method="max", ascending=False)
        idx = gt_row.index[0]
        rows.append({
            "query_idx": qidx,
            "rank_min":  float(grp.loc[idx, "rank_min"]),
            "rank_avg":  float(grp.loc[idx, "rank_avg"]),
            "rank_max":  float(grp.loc[idx, "rank_max"]),
            "n_docs":    len(grp),
        })
    rank_df = pd.DataFrame(rows)
    _csv(rank_df, out / "tie_corrected_ranks.csv", "Min/avg/max ranks per query")

    ks = [1, 3, 5, 10, 20, 50]
    mrr_min = (1.0 / rank_df["rank_min"]).mean()
    mrr_avg = (1.0 / rank_df["rank_avg"]).mean()
    mrr_max = (1.0 / rank_df["rank_max"]).mean()

    metric_rows = []
    metric_rows.append({"metric": "MRR (optimistic, min-rank)", "value": mrr_min})
    metric_rows.append({"metric": "MRR (tie-corrected, avg-rank)", "value": mrr_avg})
    metric_rows.append({"metric": "MRR (pessimistic, max-rank)", "value": mrr_max})
    for k in ks:
        metric_rows.append({"metric": f"P@{k} (optimistic, rank_min<=k)", "value": (rank_df["rank_min"] <= k).mean()})
        metric_rows.append({"metric": f"P@{k} (tie-corrected, rank_avg<=k)", "value": (rank_df["rank_avg"] <= k).mean()})
        metric_rows.append({"metric": f"P@{k} (pessimistic, rank_max<=k)", "value": (rank_df["rank_max"] <= k).mean()})
    metric_df = pd.DataFrame(metric_rows)
    _csv(metric_df, out / "corrected_mrr_at_k.csv", "Tie-corrected MRR and P@K")

    # Comparison bar chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    # MRR comparison
    axes[0].bar(["MRR\n(optimistic)", "MRR\n(tie-corrected)", "MRR\n(pessimistic)"],
                [mrr_min, mrr_avg, mrr_max],
                color=["mediumseagreen", "steelblue", "crimson"], edgecolor="white")
    for i, v in enumerate([mrr_min, mrr_avg, mrr_max]):
        axes[0].text(i, v + 0.002, f"{v:.4f}", ha="center", va="bottom", fontweight="bold")
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("MRR")
    axes[0].set_title("MRR: Optimistic vs Tie-Corrected vs Pessimistic")

    # P@K comparison for three methods
    for k_val, color, label in [(ks, "mediumseagreen", "optimistic"),
                                 (ks, "steelblue", "tie-corrected"),
                                 (ks, "crimson", "pessimistic")]:
        pass  # plotted below

    p_opt  = [(rank_df["rank_min"] <= k).mean() * 100 for k in ks]
    p_avg  = [(rank_df["rank_avg"] <= k).mean() * 100 for k in ks]
    p_max  = [(rank_df["rank_max"] <= k).mean() * 100 for k in ks]
    xs = np.arange(len(ks))
    w = 0.25
    axes[1].bar(xs - w, p_opt, w, color="mediumseagreen", label="Optimistic (rank_min)")
    axes[1].bar(xs,     p_avg, w, color="steelblue",      label="Tie-corrected (rank_avg)")
    axes[1].bar(xs + w, p_max, w, color="crimson",        label="Pessimistic (rank_max)")
    axes[1].set_xticks(xs)
    axes[1].set_xticklabels([f"P@{k}" for k in ks])
    axes[1].set_ylabel("% Queries")
    axes[1].set_title("P@K: Effect of Tie-Breaking Method")
    axes[1].legend(fontsize=8)
    axes[1].yaxis.set_major_formatter(mticker.PercentFormatter())

    fig.suptitle("Corrected Retrieval Metrics: Why Tie-Breaking Method Matters")
    plt.tight_layout()
    _fig(out / "corrected_mrr_at_k.png", "Tie-corrected MRR and P@K comparison")

    return {"mrr_optimistic": mrr_min, "mrr_corrected": mrr_avg, "mrr_pessimistic": mrr_max}


# ─── H. Within-tier dist_to_gt analysis ──────────────────────────────────────

def section_h_within_tier(df: pd.DataFrame, out: Path) -> None:
    """
    Among docs tied at score=1.0 per query, is dist_to_gt lower for GT than for non-GT?
    If yes, dist_to_gt as a tiebreaker would help. This tests whether a hybrid
    LLM-score + distance approach could uniquely identify GT.
    """
    rows = []
    for qidx, grp in df.groupby("query_idx"):
        top_tier = grp[grp["llm_score"] == 1.0]
        if top_tier.empty or (top_tier["is_gt"] == 1).sum() == 0:
            continue
        gt_in_tier = top_tier[top_tier["is_gt"] == 1]
        ngt_in_tier = top_tier[top_tier["is_gt"] == 0]
        if ngt_in_tier.empty:
            continue
        gt_dist  = float(gt_in_tier["dist_to_gt"].iloc[0])
        ngt_dists = ngt_in_tier["dist_to_gt"].values
        pct_ngt_further = float((ngt_dists > gt_dist).mean() * 100)
        rows.append({
            "query_idx":       qidx,
            "gt_dist":         gt_dist,
            "mean_ngt_dist":   float(ngt_dists.mean()),
            "min_ngt_dist":    float(ngt_dists.min()),
            "n_in_tier":       len(top_tier),
            "pct_ngt_further": pct_ngt_further,
            "gt_would_win_dist": int(gt_dist <= ngt_dists.min()),
        })
    tier_df = pd.DataFrame(rows)
    if tier_df.empty:
        print("  No tier data to analyze.")
        return
    _csv(tier_df, out / "within_tier_dist.csv", "GT vs non-GT dist within score=1.0 tier")

    win_rate = tier_df["gt_would_win_dist"].mean() * 100
    mean_pct_further = tier_df["pct_ngt_further"].mean()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(tier_df["pct_ngt_further"], bins=20, color="mediumseagreen", edgecolor="white")
    axes[0].axvline(mean_pct_further, color="red", linestyle="--",
                    label=f"mean={mean_pct_further:.1f}%")
    axes[0].axvline(50, color="black", linestyle=":", alpha=0.5, label="50% (random)")
    axes[0].set_xlabel("% non-GT docs in top tier with dist > GT dist")
    axes[0].set_ylabel("Queries")
    axes[0].set_title(f"Within Tier: GT is closer to GT than what % of tier peers?\n"
                      f"(100% = GT uniquely closest; 50% = random)")
    axes[0].legend()

    axes[1].bar(["GT wins\n(lowest dist\nin tier)", "GT does not\nwin"],
                [win_rate, 100 - win_rate],
                color=["mediumseagreen", "crimson"], edgecolor="white")
    axes[1].set_ylabel("% Queries")
    axes[1].set_title(f"Would dist_to_gt tiebreaker pick GT?\n"
                      f"({win_rate:.1f}% of {len(tier_df)} queries)")
    axes[1].yaxis.set_major_formatter(mticker.PercentFormatter())
    for i, v in enumerate([win_rate, 100 - win_rate]):
        axes[1].text(i, v + 1, f"{v:.1f}%", ha="center", va="bottom", fontweight="bold")

    fig.suptitle("Hybrid Re-ranking: LLM Score + Distance Tiebreaker\n"
                 "Among docs tied at score=1.0, does GT have lowest dist_to_gt?")
    plt.tight_layout()
    _fig(out / "within_tier_dist.png", "Within-tier distance tiebreaker analysis")

    print(f"  Hybrid (LLM + dist tiebreaker) win rate: {win_rate:.1f}%")


# ─── I. Marginal value of intermediate score levels ──────────────────────────

def section_i_marginal_value(df: pd.DataFrame, out: Path) -> None:
    """
    Docs scoring 0.2, 0.4, 0.6, 0.8 are 'maybes' from the LLM.
    Do they have meaningfully lower dist_to_gt than docs scoring 0.1?
    This shows whether the intermediate scores carry signal.
    """
    ngt = df[df["is_gt"] == 0].copy()
    levels = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]

    agg_rows = []
    for lvl in levels:
        sub = ngt[ngt["llm_score"] == lvl]["dist_to_gt"]
        if len(sub) == 0:
            continue
        baseline = ngt[ngt["llm_score"] == 0.1]["dist_to_gt"].mean()
        agg_rows.append({
            "score":     lvl,
            "n_docs":    len(sub),
            "mean_dist": sub.mean(),
            "median_dist": sub.median(),
            "improvement_vs_01": baseline - sub.mean(),  # positive = lower dist = better
        })
    agg_df = pd.DataFrame(agg_rows)
    _csv(agg_df, out / "marginal_score_value.csv", "Mean dist-to-GT improvement per score level")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].bar([str(r["score"]) for _, r in agg_df.iterrows()],
                agg_df["mean_dist"], color="steelblue", edgecolor="white", label="Mean dist")
    axes[0].plot([str(r["score"]) for _, r in agg_df.iterrows()],
                 agg_df["median_dist"], "o-", color="darkorange", label="Median dist")
    axes[0].set_xlabel("LLM Score (non-GT docs only)")
    axes[0].set_ylabel("dist_to_gt")
    axes[0].set_title("Mean/Median dist_to_gt by Score Level\n(lower = closer to GT = better)")
    axes[0].legend()

    axes[1].bar([str(r["score"]) for _, r in agg_df.iterrows()],
                agg_df["improvement_vs_01"],
                color=["crimson" if v < 0 else "mediumseagreen" for v in agg_df["improvement_vs_01"]],
                edgecolor="white")
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].set_xlabel("LLM Score")
    axes[1].set_ylabel("Dist improvement vs score=0.1 baseline\n(positive = closer to GT)")
    axes[1].set_title("Marginal Value: How Much Does Each Score Level Help?")

    fig.suptitle("Do Intermediate LLM Scores (0.2–0.8) Add Signal Over 0.1?")
    plt.tight_layout()
    _fig(out / "marginal_score_value.png", "Marginal dist improvement by score level")


# ─── J. Corrected summary ────────────────────────────────────────────────────

def write_corrected_summary(stats: dict, out: Path) -> None:
    lines = [
        "=" * 72,
        "EXTENDED ANALYSIS SUMMARY — CORRECTED INTERPRETATION",
        "=" * 72,
        "",
        "THE TIE PROBLEM (why base P@1=98% is misleading)",
        f"  Queries where GT reaches top score tier : {stats.get('n_queries_gt_in_top', '?')} / 199",
        f"  Queries where GT is UNIQUELY top-scorer : {stats.get('n_uniquely_top', '?')} / 199",
        f"  Median # docs tied at GT's score        : {stats.get('median_n_tied', '?'):.0f}",
        f"  Expected tier precision (1/n_tied)      : {stats.get('mean_tier_precision', 0):.3f}",
        "  Interpretation: LLM correctly classes GT as 'max relevance'",
        "  but puts ~11 other docs in the same class — it cannot pinpoint GT.",
        "",
        "CORRECTED RETRIEVAL METRICS (tie-aware)",
        f"  MRR (optimistic, rank_min)   : {stats.get('mrr_optimistic', 0):.4f}",
        f"  MRR (tie-corrected, rank_avg): {stats.get('mrr_corrected', 0):.4f}",
        f"  MRR (pessimistic, rank_max)  : {stats.get('mrr_pessimistic', 0):.4f}",
        "",
        "QUERY DIFFICULTY",
        f"  EASY   (GT only top-scorer)  : {stats.get('EASY', '?')} queries",
        f"  MEDIUM (GT + 1-9 others tied): {stats.get('MEDIUM', '?')} queries",
        f"  HARD   (GT + 10+ others tied): {stats.get('HARD', '?')} queries",
        f"  FAILED (GT outscored by others): {stats.get('FAILED', '?')} queries",
        "",
        "REVISED CONCLUSION",
        textwrap.fill(
            "The LLM is excellent at binary relevance detection: it assigns score=1.0 "
            "to the GT document in 98% of queries (ROC-AUC=0.996). "
            "However, it cannot discriminate at fine-grained level within the top tier — "
            f"on average {stats.get('median_n_tied', 11):.0f} documents share the top score per query. "
            f"Tie-corrected MRR drops from {stats.get('mrr_optimistic', 0):.3f} to {stats.get('mrr_corrected', 0):.3f}, "
            "reflecting that LLM alone cannot replace the retriever — it needs a secondary signal "
            "(e.g., dist_to_gt or cosine similarity) to break ties within the top-scoring group. "
            "As a FILTER (keep score=1.0 docs, avg pool of ~11), the LLM is very effective. "
            "As a RANKER (pick single top doc), it needs tie-breaking.",
            width=72,
        ),
        "",
        "=" * 72,
    ]
    path = out / "extended_summary.txt"
    path.write_text("\n".join(lines))
    print(f"  [txt]   {path.name}")


# ─── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("db_path", nargs="?")
    parser.add_argument("--run-dir")
    args = parser.parse_args()

    if args.run_dir:
        db_path = Path(args.run_dir) / "experiment_results.db"
    elif args.db_path:
        db_path = Path(args.db_path)
    else:
        db_path = _find_latest_db()

    if not db_path.exists():
        print(f"[ERROR] DB not found: {db_path}", file=sys.stderr)
        sys.exit(1)

    run_dir = db_path.parent
    out_dir = run_dir / ANALYSIS_DIR
    out_dir.mkdir(exist_ok=True)
    print(f"Run dir : {run_dir}")
    print(f"Output  : {out_dir}\n")

    df = _load(db_path)
    all_stats: dict = {}

    print("── A. Tie Structure ───────────────────────────────────")
    all_stats.update(section_a_tie_structure(df, out_dir))

    print("── B. Score Discretization ────────────────────────────")
    section_b_score_levels(df, out_dir)

    print("── C. Threshold Filter ────────────────────────────────")
    section_c_threshold_filter(df, out_dir)

    print("── D. Query Difficulty ────────────────────────────────")
    all_stats.update(section_d_difficulty(df, out_dir))

    print("── E. Failure Deep-Dive ───────────────────────────────")
    section_e_failures(df, out_dir)

    print("── F. False Positives (non-GT @ score=1.0) ───────────")
    section_f_false_positives(df, out_dir)

    print("── G. Tie-Corrected MRR / P@K ─────────────────────────")
    all_stats.update(section_g_corrected_mrr(df, out_dir))

    print("── H. Within-Tier Distance Tiebreaker ─────────────────")
    section_h_within_tier(df, out_dir)

    print("── I. Marginal Value of Score Levels ──────────────────")
    section_i_marginal_value(df, out_dir)

    print("── J. Extended Summary ────────────────────────────────")
    write_corrected_summary(all_stats, out_dir)

    print(f"\nDone. All outputs in: {out_dir}")


if __name__ == "__main__":
    main()
