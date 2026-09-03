"""
Validates that analyze_groups_v2 produces identical results to the original.
Runs both implementations on the same 10 queries using shared embeddings.
"""

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

MAX_QUERIES = 10

RUN_DIR = (
    Path(__file__).resolve().parent.parent.parent
    / "results/cloud_runs/staged_run_2026-08-03_08-32-45"
)
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"


# ---------------------------------------------------------------------------
# Helpers — ORIGINAL implementation
# ---------------------------------------------------------------------------

def cosine_dist_orig(a, b):
    return float(1.0 - np.dot(a, b))


def compute_groups_orig(q_vec, gt_vec, vecs_q, doc_ids_q, q_scores):
    dist_q_gt = cosine_dist_orig(q_vec, gt_vec)
    dist_doc_to_query = 1.0 - vecs_q @ q_vec

    g1_flags = (dist_doc_to_query <= dist_q_gt).astype(int)
    g1_doc_ids = set(np.array(doc_ids_q)[g1_flags.astype(bool)])

    g3_flags = (q_scores["llm_score"].values == 1.0).astype(int)
    g3_doc_ids = set(np.array(doc_ids_q)[g3_flags.astype(bool)])

    g3_indices = np.where(g3_flags)[0]
    if len(g3_indices) > 0:
        g3_vecs = vecs_q[g3_indices]
        centroid = g3_vecs.mean(axis=0)
        centroid_norm = centroid / np.linalg.norm(centroid)
        dist_centroid_gt = cosine_dist_orig(centroid_norm, gt_vec)
        dist_doc_to_centroid = 1.0 - vecs_q @ centroid_norm
        g2_flags = (dist_doc_to_centroid <= dist_centroid_gt).astype(int)
        g2_doc_ids = set(np.array(doc_ids_q)[g2_flags.astype(bool)])
    else:
        centroid_norm = None
        dist_centroid_gt = None
        g2_flags = np.zeros(len(doc_ids_q), dtype=int)
        g2_doc_ids = set()

    return dict(
        dist_q_gt=dist_q_gt,
        dist_centroid_gt=dist_centroid_gt,
        g1=g1_doc_ids, g2=g2_doc_ids, g3=g3_doc_ids,
        g1_flags=g1_flags, g2_flags=g2_flags, g3_flags=g3_flags,
    )


# ---------------------------------------------------------------------------
# Helpers — V2 (refactored) implementation
# ---------------------------------------------------------------------------

def cosine_dist_v2(a, b):
    return float(1.0 - np.dot(a, b))


def docs_in_circle(vecs_q, doc_ids, center_vec, radius):
    distances = 1.0 - vecs_q @ center_vec
    flags = (distances <= radius).astype(int)
    return flags, set(np.array(doc_ids)[flags.astype(bool)])


def compute_g3_v2(q_scores, doc_ids_q):
    flags = (q_scores["llm_score"].values == 1.0).astype(int)
    return flags, set(np.array(doc_ids_q)[flags.astype(bool)])


def compute_centroid_v2(vecs_q, g3_flags):
    indices = np.where(g3_flags)[0]
    if len(indices) == 0:
        return None
    centroid = vecs_q[indices].mean(axis=0)
    norm = np.linalg.norm(centroid)
    return centroid / norm if norm > 0 else None


def compute_groups_v2(q_vec, gt_vec, vecs_q, doc_ids_q, q_scores):
    dist_q_gt = cosine_dist_v2(q_vec, gt_vec)

    g1_flags, g1_doc_ids = docs_in_circle(vecs_q, doc_ids_q, q_vec, dist_q_gt)
    g3_flags, g3_doc_ids = compute_g3_v2(q_scores, doc_ids_q)

    centroid_norm = compute_centroid_v2(vecs_q, g3_flags)
    if centroid_norm is not None:
        dist_centroid_gt = cosine_dist_v2(centroid_norm, gt_vec)
        g2_flags, g2_doc_ids = docs_in_circle(vecs_q, doc_ids_q, centroid_norm, dist_centroid_gt)
    else:
        dist_centroid_gt = None
        g2_flags = np.zeros(len(doc_ids_q), dtype=int)
        g2_doc_ids = set()

    return dict(
        dist_q_gt=dist_q_gt,
        dist_centroid_gt=dist_centroid_gt,
        g1=g1_doc_ids, g2=g2_doc_ids, g3=g3_doc_ids,
        g1_flags=g1_flags, g2_flags=g2_flags, g3_flags=g3_flags,
    )


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def extract_doc_text(s):
    try:
        d = json.loads(s)
        raw = d.get("contexts") or d.get("CONTEXTS") or ""
        return " ".join(str(c) for c in raw if c) if isinstance(raw, list) else str(raw)
    except Exception:
        return str(s)


print("Loading data …")
sq = pd.read_parquet(RUN_DIR / "scoring_queue.parquet",
                     columns=["query_idx", "query_text", "gt_id", "doc_id", "doc_text", "is_gt"])

conn = sqlite3.connect(RUN_DIR / "experiment_results.db")
scores_df = pd.read_sql(
    "SELECT query_idx, doc_id, llm_score, rag_failed FROM results", conn)
conn.close()

queries_meta = (sq[["query_idx", "query_text", "gt_id"]]
                .drop_duplicates("query_idx")
                .sort_values("query_idx")
                .reset_index(drop=True)
                .head(MAX_QUERIES))

pool_docs = sq[["doc_id", "doc_text"]].drop_duplicates("doc_id").reset_index(drop=True)

# Restrict pool to only docs appearing in the first MAX_QUERIES queries
relevant_q_idxs = set(queries_meta["query_idx"].tolist())
relevant_doc_ids = set(sq[sq["query_idx"].isin(relevant_q_idxs)]["doc_id"].tolist())
# Also need GT docs
gt_ids = set(queries_meta["gt_id"].tolist())
relevant_doc_ids |= gt_ids
pool_docs = pool_docs[pool_docs["doc_id"].isin(relevant_doc_ids)].reset_index(drop=True)

print(f"Queries: {len(queries_meta)}, Pool docs for these queries: {len(pool_docs)}")

print("Embedding …")
model = SentenceTransformer(MODEL_NAME)
query_vecs = model.encode(queries_meta["query_text"].tolist(),
                          batch_size=64, normalize_embeddings=True, show_progress_bar=True)
doc_texts = [extract_doc_text(t) for t in pool_docs["doc_text"].tolist()]
doc_vecs = model.encode(doc_texts, batch_size=64, normalize_embeddings=True, show_progress_bar=True)

q_idx_to_vec = dict(zip(queries_meta["query_idx"].tolist(), query_vecs))
doc_id_to_vec = dict(zip(pool_docs["doc_id"].tolist(), doc_vecs))

# ---------------------------------------------------------------------------
# Run both implementations and compare
# ---------------------------------------------------------------------------

print("\n=== Running comparison ===\n")
all_pass = True

for _, qrow in queries_meta.iterrows():
    q_idx = int(qrow["query_idx"])
    gt_id = str(qrow["gt_id"])
    q_vec = q_idx_to_vec[q_idx]
    gt_vec = doc_id_to_vec.get(gt_id)
    if gt_vec is None:
        print(f"  query {q_idx}: SKIP (no GT embedding)")
        continue

    q_scores = scores_df[scores_df["query_idx"] == q_idx].copy()
    q_scores = q_scores[q_scores["doc_id"].isin(doc_id_to_vec)].copy()
    doc_ids_q = q_scores["doc_id"].tolist()
    if not doc_ids_q:
        continue
    vecs_q = np.stack([doc_id_to_vec[d] for d in doc_ids_q])

    orig = compute_groups_orig(q_vec, gt_vec, vecs_q, doc_ids_q, q_scores)
    v2   = compute_groups_v2  (q_vec, gt_vec, vecs_q, doc_ids_q, q_scores)

    checks = {
        "dist_q_gt":       (orig["dist_q_gt"],       v2["dist_q_gt"]),
        "dist_centroid_gt":(orig["dist_centroid_gt"], v2["dist_centroid_gt"]),
        "G1 set":          (orig["g1"],               v2["g1"]),
        "G2 set":          (orig["g2"],               v2["g2"]),
        "G3 set":          (orig["g3"],               v2["g3"]),
    }

    query_pass = True
    for name, (a, b) in checks.items():
        if isinstance(a, set):
            ok = (a == b)
        elif a is None and b is None:
            ok = True
        elif a is None or b is None:
            ok = False
        else:
            ok = abs(a - b) < 1e-9
        if not ok:
            print(f"  query {q_idx} | {name}: FAIL  orig={a}  v2={b}")
            query_pass = False
            all_pass = False

    status = "PASS" if query_pass else "FAIL"
    g1, g2, g3 = len(orig["g1"]), len(orig["g2"]), len(orig["g3"])
    print(f"  query {q_idx}: {status}  G1={g1} G2={g2} G3={g3}")

print()
print("=" * 45)
print("OVERALL:", "PASS — refactored code is identical to original" if all_pass
      else "FAIL — see diffs above")
print("=" * 45)
