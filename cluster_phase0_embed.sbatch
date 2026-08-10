#!/bin/bash
#SBATCH --job-name=pubmed-embed
#SBATCH --partition=gpu
#SBATCH --time=0-06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:rtx_3090:1
#
# To use a different GPU (comment/uncomment):
# #SBATCH --gres=gpu:rtx_4090:1
# #SBATCH --gres=gpu:rtx_6000:1
# #SBATCH --gres=gpu:rtx_2080:1
#
#SBATCH --output=logs/phase0_embed_%J.out
#SBATCH --error=logs/phase0_embed_%J.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=bardamri1702@gmail.com

set -euo pipefail

PROJECT_DIR="$HOME/ModularRAGOptimization"
CONDA_ENV="rag_env"

echo "========================================"
echo "  Phase 0: PubMed Embedding"
echo "  Job ID : $SLURM_JOB_ID"
echo "  Node   : $(hostname)"
echo "  Start  : $(date)"
echo "========================================"

# ── Environment ──────────────────────────────────────────────────────────────
module load anaconda 2>/dev/null || module load anaconda3 2>/dev/null
source activate "${CONDA_ENV}"

cd "$PROJECT_DIR"

# ── GPU check ─────────────────────────────────────────────────────────────────
if ! nvidia-smi &>/dev/null; then
    echo "ERROR: No GPU detected. Aborting — embedding 36M docs on CPU would take days."
    exit 1
fi
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"

# ── HuggingFace cache → local scratch (fast SSD, auto-cleaned after job) ─────
# TMPDIR is set by Slurm to a per-job local scratch directory on the compute node.
export HF_HOME="${TMPDIR}/hf_cache"
export HF_HUB_CACHE="${TMPDIR}/hf_cache"
mkdir -p "$HF_HOME"
echo "HF cache: $HF_HOME  ($(df -h $TMPDIR | awk 'NR==2{print $4}') free)"

# Chroma DB persists to home (survives job end)
echo "Chroma DB: $PROJECT_DIR/storage/chroma/pubmed_full/cosine/"
df -h "$HOME" | awk 'NR==2{print "Home disk: " $3 " used / " $2 " total (" $5 " full)"}'

# ── Run Phase 0 ──────────────────────────────────────────────────────────────
echo ""
echo "--- Starting embed_full_pubmed.py ---"
echo "Tip: tail -f logs/phase0_embed_${SLURM_JOB_ID}.out  to watch progress"

python embed_full_pubmed.py \
    --embed-batch 256 \
    --resume \
    2>&1

EXIT_CODE=$?

echo ""
echo "========================================"
echo "  Phase 0 complete (exit $EXIT_CODE)"
echo "  End: $(date)"
echo "========================================"

if [ $EXIT_CODE -ne 0 ]; then
    echo "ERROR: embed_full_pubmed.py failed with exit code $EXIT_CODE"
    exit $EXIT_CODE
fi

# Print Chroma DB size for the record
if [ -d "$PROJECT_DIR/storage/chroma/pubmed_full" ]; then
    echo "Chroma DB size: $(du -sh $PROJECT_DIR/storage/chroma/pubmed_full | cut -f1)"
fi

echo "Phase 1 job can now be submitted or was auto-submitted via cluster_submit.sh"
