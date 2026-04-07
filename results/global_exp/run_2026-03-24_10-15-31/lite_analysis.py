import sqlite3
import pandas as pd


def export_lite_analysis(db_path, output_path):
    conn = sqlite3.connect(db_path)

    query_stats = """
    SELECT 
        query_idx,
        AVG(CASE WHEN is_gt = 1 THEN llm_score END) as gt_avg_score,
        AVG(CASE WHEN is_gt = 0 THEN llm_score END) as non_gt_avg_score,
        MIN(CASE WHEN is_gt = 1 THEN dist_to_gt END) as gt_dist, -- אמור להיות 0
        COUNT(*) as total_docs
    FROM results
    GROUP BY query_idx
    """
    df_stats = pd.read_sql(query_stats, conn)

    query_outliers = """
    SELECT * FROM (
        SELECT 
            query_idx, doc_id, llm_score, dist_to_gt, is_gt,
            RANK() OVER (PARTITION BY query_idx ORDER BY llm_score DESC, dist_to_gt ASC) as llm_rank
        FROM results
        WHERE is_gt = 0
    ) WHERE llm_rank <= 3
    """
    df_outliers = pd.read_sql(query_outliers, conn)

    with pd.ExcelWriter(output_path) as writer:
        df_stats.to_excel(writer, sheet_name='SummaryStats', index=False)
        df_outliers.to_excel(writer, sheet_name='TopNonGT', index=False)

    conn.close()
    print(f"Done! Please upload {output_path}")


export_lite_analysis('experiment_results.db', 'lite_analysis.xlsx')