#!/bin/bash
#SBATCH --job-name=pubmed-export
#SBATCH --partition=main
#SBATCH --time=0-01:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/export_%J.out
#SBATCH --error=logs/export_%J.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=bardamri1702@gmail.com

set -euo pipefail

PROJECT_DIR="$HOME/ModularRAGOptimization"
CONDA_ENV="rag_env"

echo "========================================"
echo "  Export: Safety-net artifacts"
echo "  Job ID : $SLURM_JOB_ID"
echo "  Node   : $(hostname)"
echo "  Start  : $(date)"
echo "========================================"

module load anaconda 2>/dev/null || module load anaconda3 2>/dev/null
source activate "${CONDA_ENV}"

cd "$PROJECT_DIR"

# Find the latest Phase-1 run dir
LATEST_RUN=$(ls -td results/cluster_run/staged_run_* 2>/dev/null | head -1)
if [ -z "$LATEST_RUN" ]; then
    echo "ERROR: No staged_run_* directory found under results/cluster_run/"
    exit 1
fi
if [ ! -f "$LATEST_RUN/scoring_queue.parquet" ]; then
    echo "ERROR: scoring_queue.parquet missing in $LATEST_RUN"
    exit 1
fi
echo "Run dir: $LATEST_RUN"

echo ""
echo "--- Starting export_pool_artifacts.py ---"

python export_pool_artifacts.py \
    --run-dir "$LATEST_RUN" \
    --chroma-source pubmed_full \
    2>&1

EXIT_CODE=$?

echo ""
echo "========================================"
echo "  Export complete (exit $EXIT_CODE)"
echo "  End: $(date)"
echo "========================================"

if [ $EXIT_CODE -ne 0 ]; then
    echo "ERROR: export_pool_artifacts.py failed with exit code $EXIT_CODE"
    exit $EXIT_CODE
fi

# List exported artifacts
echo ""
echo "Artifacts in $LATEST_RUN :"
for f in scoring_queue.parquet pool_embeddings.npy pool_doc_ids.json \
          pool_metadata.parquet gt_embeddings.npy gt_doc_ids.json run_metadata.json; do
    if [ -f "$LATEST_RUN/$f" ]; then
        SIZE=$(du -sh "$LATEST_RUN/$f" | cut -f1)
        echo "  $SIZE  $f"
    else
        echo "  MISSING: $f"
    fi
done

echo ""
echo "Download command (run from your local machine):"
echo "  scp -r bardamri@slurm.bgu.ac.il:$LATEST_RUN/ ./cluster_results/"
