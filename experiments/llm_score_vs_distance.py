import csv
import os
import re
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from google import genai

from configurations.config import GEMINI_API_KEY, RUN_RANDOM_SCENARIO
from scripts.qa_data_set_loader import load_qa_queries
from utility.distance_metrics import DistanceMetric
from utility.logger import logger
from vector_db.trilateration_retriever import cosine_distance

client = genai.Client(api_key=GEMINI_API_KEY)


# --------------------------------------------------------------
def sample_random_docs_by_pmid(vector_db, n):
    """
    Sample ~n random documents by selecting random PMIDs directly from Chroma.
    Returns a list of TextNode objects with embeddings attached.
    """
    import random
    from llama_index.core.schema import TextNode

    # Peek metadata only (fast)
    num_of_samples = vector_db.collection.count()
    raw = vector_db.collection.get(include=["metadatas"])
    all_pmid = [node["PMID"] for node in raw["metadatas"]]

    selected_pmids = random.sample(all_pmid, min(n, num_of_samples))

    sampled_nodes = []

    for pmid in selected_pmids:
        try:
            res = vector_db.collection.get(
                where={"PMID": pmid},
                include=["documents", "embeddings", "metadatas"]
            )
            if res and res.get("documents") and res["documents"][0]:
                doc = res["documents"][0]
                emb = res["embeddings"][0]
                meta = res["metadatas"][0]
                node = TextNode(text=doc, metadata=meta, id_=str(pmid))
                node.embedding = np.array(emb, dtype=np.float32)
                sampled_nodes.append(node)
        except Exception as e:
            logger.warning(f"⚠️ sample_random_docs_by_pmid(): Error retrieving PMID {pmid}: {e}")
            continue

    return sampled_nodes


# --------------------------------------------------------------
# LLM Score vs Distance Scatter Experiment
# --------------------------------------------------------------

import time


def gemini_score(query, document, max_retries=3):
    prompt = f"""
### ROLE
You are an expert scientific relevance assessor.

### TASK
Given a QUERY and a DOCUMENT excerpt, output a single relevance score.

### OUTPUT FORMAT (STRICT)
- Output ONLY one number from this list (no extra text):
  0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0

### SCORING GUIDE
- 1.0 = Directly answers the query OR provides the exact evidence needed.
- 0.9 = Extremely relevant; minor missing detail.
- 0.8 = Strongly relevant; key concepts and context match.
- 0.7 = Relevant; supports the query but not perfectly aligned.
- 0.6 = Moderately relevant; same subtopic, partial support.
- 0.5 = Same field; different context/population/setting.
- 0.4 = Weak relevance; only a small overlap of terms/concepts.
- 0.3 = Mostly unrelated; superficial keyword overlap.
- 0.2 = Nearly irrelevant.
- 0.1 = Barely related.
- 0.0 = Completely irrelevant.

### DECISION RULES
- Judge semantic relevance, not writing quality.
- If the DOCUMENT contradicts the QUERY or is off-topic, score <= 0.2.
- If the DOCUMENT is about the right general domain but wrong specific question, score around 0.4–0.6.
- If the DOCUMENT provides direct evidence, score >= 0.8.

---
QUERY:
{query}

---
DOCUMENT (excerpt):
{document[:8000]}
"""

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt
            )
            text = response.text.strip()

            m = re.search(r"\b(0\.[0-9]|1\.0|0\.0|1)\b", text)
            if m:
                val = float(m.group(1))
                # Normalize rare "1" to "1.0" and clamp
                val = 1.0 if val == 1 else val
                return min(1.0, max(0.0, val))
            return 0.0

        except Exception as e:
            if "503" in str(e) or "overloaded" in str(e).lower():
                wait_time = (attempt + 1) * 5
                print(f"⚠️ Server overloaded (503). Retrying in {wait_time}s... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
            else:
                print(f"⚠️ gemini_score exception: {e}")
                return 0.0

    return 0.0


def crossencoder_score(cross_encoder_model, query: str, document: str) -> float:
    import numpy as np
    try:
        pair = [(query, document)]
        raw_output = cross_encoder_model.predict(pair)

        if isinstance(raw_output, (np.ndarray, list)):
            logit = float(raw_output[0])
        else:
            logit = float(raw_output)

        return logit
    except Exception as e:
        print(f"⚠️ crossencoder_score error: {e}")
        return -10.0


def run_llm_score_vs_distance_scatter_experiment(
        vector_db,
        embedding_model,
        llm_model,
        llm_tokenizer,
        cross_encoder_model,
        output_dir="results/llm_scatter"):
    """
    For each query:
      • Scenario A (Optional): random docs -> Save to separate CSV/Plot
      • Scenario B (Always): closest docs to query -> Save to separate CSV/Plot
    """

    queries = load_qa_queries(3)
    num_of_docs = 200
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    run_output_dir = os.path.join(output_dir, f"gemini_llm_score_vs_distance_{timestamp}")
    os.makedirs(run_output_dir, exist_ok=True)

    scenario_random_nodes = []
    if RUN_RANDOM_SCENARIO:
        logger.info(f"📚 Scenario A: Sampling {num_of_docs} random documents by PMIDs...")
        scenario_random_nodes = sample_random_docs_by_pmid(vector_db, num_of_docs)
        if not scenario_random_nodes:
            logger.error("❌ Could not sample any random documents. Skipping Scenario A.")
        else:
            logger.info(f"📘 Loaded {len(scenario_random_nodes)} random documents.")

    metric = getattr(vector_db, "distance_metric", None)
    if metric == DistanceMetric.COSINE:
        metric_func = cosine_distance
    elif metric == DistanceMetric.EUCLIDEAN:
        metric_func = lambda a, b: np.linalg.norm(a - b)
    elif metric == DistanceMetric.INNER_PRODUCT:
        metric_func = lambda a, b: -np.dot(a, b)
    else:
        metric_func = cosine_distance

    # ==================== Loop per Query ====================
    for q_index, entry in enumerate(queries, start=1):
        query = entry.get("question")
        gt_id = entry.get("pubid") or entry.get("PMID") or entry.get("pmid")

        if gt_id is None:
            logger.warning(f"⚠️ Query {q_index} has no pubid/PMID; skipping.")
            continue

        gt_id_str = str(gt_id)

        try:
            gt_node_res = vector_db.collection.get(where={"PMID": gt_id}, include=["embeddings"])
            embs = gt_node_res.get("embeddings")
            if embs is None or len(embs) == 0:
                logger.warning(f"⚠️ Ground-truth PMID {gt_id_str} not found in DB; skipping.")
                continue
            gt_emb = np.array(embs[0], dtype=np.float32)
        except Exception as e:
            logger.error(f"❌ Error fetching GT for query {q_index}: {e}")
            continue

        logger.info(f"\n🔍 Processing query {q_index}: {query[:60]} ...")

        def process_and_plot(nodes, scenario_name, color_llm):
            if not nodes:
                return

            rows = []
            data_llm = {'x': [], 'y': []}
            data_ce = {'x': [], 'y': []}

            for node in nodes:
                doc_id = _extract_doc_id(node)
                doc_emb = np.array(node.embedding, dtype=np.float32)
                doc_text = node.get_content() if hasattr(node, "get_content") else node.text

                llm_s = gemini_score(query, doc_text)
                ce_s = crossencoder_score(cross_encoder_model, query, doc_text) if cross_encoder_model else None

                dist = metric_func(gt_emb, doc_emb)

                rows.append([q_index, gt_id_str, doc_id, llm_s, ce_s if ce_s is not None else "", dist])

                data_llm['x'].append(llm_s)
                data_llm['y'].append(dist)

                if ce_s is not None:
                    data_ce['x'].append(ce_s)
                    data_ce['y'].append(dist)

                time.sleep(0.5)

            csv_name = f"query_{q_index:02d}_{scenario_name}.csv"
            with open(os.path.join(run_output_dir, csv_name), "w", newline="", encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["query_idx", "query_gt_id", "doc_id", "llm_score", "cross_score", "distance"])
                writer.writerows(rows)

            plt.figure(figsize=(8, 6))
            plt.scatter(data_llm['x'], data_llm['y'], alpha=0.6, color=color_llm, s=15, label="Gemini Score")

            if data_ce['x']:
                plt.scatter(data_ce['x'], data_ce['y'], alpha=0.6, color='orange', s=15, marker='x',
                            label="CrossEncoder Score")

            plt.title(f"Query {q_index}: {scenario_name.capitalize()} Docs\n(GT Distance)")
            plt.xlabel("Score (LLM: 0-1, CE: Logits)")
            plt.ylabel("Vector Distance to GT")
            plt.legend()
            plt.grid(True, alpha=0.3)

            plot_name = f"query_{q_index:02d}_{scenario_name}.png"
            plt.savefig(os.path.join(run_output_dir, plot_name), dpi=200)
            plt.close()

        if RUN_RANDOM_SCENARIO and scenario_random_nodes:
            logger.info(f"📊 Processing Scenario A (Random) for Query {q_index}...")
            process_and_plot(scenario_random_nodes, "random", "gray")

        logger.info(f"🎯 Processing Scenario B (Closest) for Query {q_index}...")
        query_emb = embedding_model.get_text_embedding(query)
        query_emb = np.array(query_emb, dtype=np.float32)

        closest_hits = vector_db.retrieve(query_emb, top_k=num_of_docs)
        closest_nodes = [hit.node for hit in closest_hits]

        process_and_plot(closest_nodes, "closest", "blue")

        logger.info(f"✅ Completed Query {q_index}")

    logger.info("🎉 Experiment completed.")


def _extract_doc_id(node):
    """Helper to safely extract doc_id from node or metadata."""
    doc_id = getattr(node, "id_", None)
    if not doc_id and hasattr(node, "metadata") and node.metadata:
        for key in ["PMID", "pmid", "pubid"]:
            if key in node.metadata:
                doc_id = str(node.metadata[key])
                break
    return str(doc_id) if doc_id else "unknown"


################################################################################################
# Retriever Rank vs Distance Experiment
################################################################################################

def run_retriever_rank_vs_distance_experiment(
        vector_db,
        embedding_model,
        num_queries=200,
        output_dir="results/top_k_rank_vs_distance"
):
    """
    For each query:
      1. Retrieve top_k=100 docs using vector_db.retrieve(query_text, top_k=100).
      2. Extract ground-truth PMID from query entry.
      3. Identify ground-truth doc in retrieved set (if present).
      4. For each retrieved doc: compute rank, distance to GT, mark GT, mark rank-1.
      5. Save per-query CSV: rank, distance, is_gt, is_rank_1.
      6. Save per-query scatter plot: x=rank, y=distance, mark rank-1, mark GT.
      7. All outputs under output_dir/rank_vs_distance_<timestamp>/
    """

    # ==================== Initialization ====================
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = os.path.join(output_dir, f"rank_vs_distance_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)

    queries = load_qa_queries(num_queries)

    summary_stats = []
    # ==================== For each Query ====================
    for q_idx, entry in enumerate(queries, start=1):
        # ---- Retrieve query text and GT PMID ----
        query_text = entry.get("question")
        gt_id = entry.get("PMID") or entry.get("pmid") or entry.get("pubid")
        if gt_id is None:
            logger.warning(f"⚠️ Query {q_idx} missing ground-truth PMID/pubid/pmid; skipping.")
            continue
        gt_id_str = str(gt_id)
        # ---- Fetch GT embedding from DB by metadata PMID ----
        try:
            gt_res = vector_db.collection.get(
                where={"PMID": gt_id},
                include=["embeddings"]
            )
        except Exception as e:
            logger.warning(f"⚠️ Failed to fetch GT embedding for PMID={gt_id_str}: {e}")
            continue

        embs = gt_res.get("embeddings")
        if embs is None or len(embs) == 0 or embs[0] is None:
            logger.warning(f"⚠️ Ground-truth PMID {gt_id_str} not found in DB; skipping query.")
            continue

        # Chroma returns a list-of-embeddings; take the first match
        gt_emb = np.array(embs[0], dtype=np.float32)

        # ==================== Retrieval ====================
        if not query_text:
            logger.warning(f"⚠️ Query {q_idx} missing question text; skipping.")
            continue

        try:
            emb = embedding_model.get_text_embedding(query_text)
            query_emb = np.array(gt_emb, dtype=np.float32)
            retrieved_hits = vector_db.retrieve(query_emb, top_k=100)
        except Exception as e:
            logger.error(f"❌ Retrieval failed for query {q_idx}: {e}")
            continue
        # Each hit: .node (with .id_, .embedding, etc.)
        retrieved_nodes = []
        for hit in retrieved_hits:
            node = getattr(hit, "node", None)
            if node is not None:
                retrieved_nodes.append(node)
        if not retrieved_nodes:
            logger.warning(f"⚠️ No nodes retrieved for query {q_idx}.")
            continue

        # ==================== Distance Computation ====================
        # Use vector_db's metric if available
        metric = getattr(vector_db, "distance_metric", None)
        if metric == DistanceMetric.COSINE:
            metric_func = cosine_distance
        elif metric == DistanceMetric.EUCLIDEAN:
            metric_func = lambda a, b: np.linalg.norm(a - b)
        elif metric == DistanceMetric.INNER_PRODUCT:
            metric_func = lambda a, b: -np.dot(a, b)
        else:
            metric_func = cosine_distance

        # ==================== Identify GT Doc in Retrieved ====================
        # Prepare data for CSV and plotting
        csv_rows = []
        distances = []
        gt_idx = None

        for idx, node in enumerate(retrieved_nodes):
            # Determine doc_id for traceability
            doc_id = getattr(node, "id_", None)
            if not doc_id:
                doc_id = None
                if hasattr(node, "metadata") and node.metadata:
                    for key in ["PMID", "pmid", "pubid"]:
                        if key in node.metadata:
                            doc_id = str(node.metadata[key])
                            break
            doc_id = str(doc_id) if doc_id is not None else ""
            node_pmid = (
                node.metadata.get("PMID") if hasattr(node, "metadata") and node.metadata else None
            )
            if node.embedding is None:
                csv_rows.append([q_idx, gt_id_str, doc_id, idx + 1, None, 0, int(idx == 0)])
                distances.append(np.nan)
                continue
            node_emb = np.array(node.embedding, dtype=np.float32)
            dist = metric_func(gt_emb, node_emb)
            is_gt = False
            # Compare as string
            if node_pmid is not None and str(node_pmid).strip() == gt_id_str.strip():
                gt_idx = idx
                is_gt = True

            csv_rows.append([q_idx, gt_id_str, doc_id, idx + 1, dist, int(is_gt), int(idx == 0)])
            distances.append(dist)
        is_found = (gt_idx is not None)
        rank_pos = gt_idx + 1 if is_found else -1
        is_rank_1 = (rank_pos == 1)

        summary_stats.append({
            "query_idx": q_idx,
            "gt_id": gt_id_str,
            "found": is_found,
            "rank": rank_pos,
            "is_rank_1": is_rank_1
        })

        # ==================== Save CSV ====================
        csv_path = os.path.join(out_dir, f"query_{q_idx:02d}_rank_vs_distance.csv")
        with open(csv_path, "w", newline="", encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["query_idx", "query_gt_id", "doc_id", "rank", "distance", "is_gt", "is_rank_1"])
            writer.writerows(csv_rows)

        # ==================== Plotting ====================
        plt.figure(figsize=(7, 5))
        ranks = np.arange(1, len(distances) + 1)
        plt.scatter(ranks, distances, s=16, color="blue", label="Retrieved Docs")

        # Mark rank 1
        if distances and not np.isnan(distances[0]):
            plt.scatter([1], [distances[0]], s=80, color="green", marker="*", label="Rank 1", zorder=5,
                        edgecolor="black")

        # Mark GT if present
        if gt_idx is not None:
            gt_rank = gt_idx + 1
            gt_dist = distances[gt_idx]
            if gt_rank == 1:
                # Both GT and rank 1: double marker (filled + outlined)
                plt.scatter([gt_rank], [gt_dist], s=160, facecolors='none', edgecolors='red', marker="o", linewidths=2,
                            label="GT & Rank 1", zorder=6)
            else:
                plt.scatter([gt_rank], [gt_dist], s=100, color="red", marker="o", edgecolor="black",
                            label="Ground Truth", zorder=6)

        plt.xlabel("Rank")
        plt.ylabel("Distance to Ground Truth")
        plt.title(f"Query {q_idx}: Rank vs Distance")
        plt.grid(alpha=0.4)
        plt.legend()
        plt.tight_layout()
        plot_path = os.path.join(out_dir, f"query_{q_idx:02d}_rank_vs_distance.png")
        plt.savefig(plot_path, dpi=200)
        plt.close()

    if summary_stats:
        total = len(summary_stats)
        count_rank_1 = sum(1 for s in summary_stats if s["is_rank_1"])
        count_found = sum(1 for s in summary_stats if s["found"])

        accuracy = (count_rank_1 / total) * 100
        recall = (count_found / total) * 100

        with open(os.path.join(out_dir, "summary_report.txt"), "w", encoding='utf-8') as f:
            f.write(f"Total Queries: {total}\n")
            f.write(f"GT Found (Recall): {count_found} ({recall:.2f}%)\n")
            f.write(f"GT at Rank 1 (Accuracy): {count_rank_1} ({accuracy:.2f}%)\n")

        with open(os.path.join(out_dir, "summary_all_queries.csv"), "w", newline="", encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["query_idx", "gt_id", "found", "rank", "is_rank_1"])
            for s in summary_stats:
                writer.writerow([s["query_idx"], s["gt_id"], s["found"], s["rank"], s["is_rank_1"]])

        logger.info(f"🏆 Final Results -> Accuracy@1: {accuracy:.2f}% | Recall: {recall:.2f}%")
