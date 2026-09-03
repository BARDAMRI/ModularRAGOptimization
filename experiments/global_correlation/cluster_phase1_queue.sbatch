#!/bin/bash
#SBATCH --job-name=pubmed-queue
#SBATCH --partition=main
#SBATCH --time=0-04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=logs/phase1_queue_%J.out
#SBATCH --error=logs/phase1_queue_%J.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=bardamri1702@gmail.com

set -euo pipefail

PROJECT_DIR="$HOME/ModularRAGOptimization"
CONDA_ENV="rag_env"

echo "========================================"
echo "  Phase 1: Build Scoring Queue"
echo "  Job ID : $SLURM_JOB_ID"
echo "  Node   : $(hostname)"
echo "  Start  : $(date)"
echo "========================================"

module load anaconda 2>/dev/null || module load anaconda3 2>/dev/null
source activate "${CONDA_ENV}"

cd "$PROJECT_DIR"

# Verify Chroma DB exists from Phase 0
if [ ! -d "storage/chroma/pubmed_full/cosine" ]; then
    echo "ERROR: Chroma DB not found at storage/chroma/pubmed_full/cosine"
    echo "       Make sure Phase 0 completed successfully before running Phase 1."
    exit 1
fi
echo "Chroma DB found: $(du -sh storage/chroma/pubmed_full | cut -f1)"

echo ""
echo "--- Starting build_scoring_queue.py ---"

python build_scoring_queue.py \
    --chroma-source pubmed_full \
    --num-queries 199 \
    --max-ranked-gt 100 \
    --max-ranked-query 100 \
    --output-dir results/cluster_run \
    2>&1

EXIT_CODE=$?

echo ""
echo "========================================"
echo "  Phase 1 complete (exit $EXIT_CODE)"
echo "  End: $(date)"
echo "========================================"

if [ $EXIT_CODE -ne 0 ]; then
    echo "ERROR: build_scoring_queue.py failed with exit code $EXIT_CODE"
    exit $EXIT_CODE
fi

# Show result summary
LATEST_RUN=$(ls -td results/cluster_run/staged_run_* 2>/dev/null | head -1)
if [ -n "$LATEST_RUN" ] && [ -f "$LATEST_RUN/scoring_queue.parquet" ]; then
    echo "Scoring queue: $LATEST_RUN/scoring_queue.parquet"
    echo "Queue size:    $(du -sh $LATEST_RUN/scoring_queue.parquet | cut -f1)"
fi
