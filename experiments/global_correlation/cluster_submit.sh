#!/usr/bin/env bash
# cluster_submit.sh — Submit Phase 0 → Phase 1 → Export as a dependency chain
# ===========================================================================
# Run from the BGU cluster login node:
#   cd ~/ModularRAGOptimization
#   bash cluster_submit.sh
#
# Options:
#   --skip-phase0   Skip embedding (Chroma DB already exists)
#   --phase0-only   Submit only Phase 0
#   --phase1-only   Submit Phase 1 + Export (Chroma DB must exist)

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
SKIP_PHASE0=false
PHASE0_ONLY=false
PHASE1_ONLY=false

for arg in "$@"; do
    case $arg in
        --skip-phase0)   SKIP_PHASE0=true   ;;
        --phase0-only)   PHASE0_ONLY=true   ;;
        --phase1-only)   PHASE1_ONLY=true   ;;
        *) echo "Unknown arg: $arg"; exit 1  ;;
    esac
done

cd "$PROJECT_DIR"
mkdir -p logs

echo "========================================"
echo "  ModularRAG Cluster Job Submission"
echo "  $(date)"
echo "  Project: $PROJECT_DIR"
echo "========================================"

# ── Phase 0: Embedding (GPU, ~3-6h) ─────────────────────────────────────────
if $SKIP_PHASE0 || $PHASE1_ONLY; then
    echo ""
    echo "[SKIP] Phase 0 — using existing Chroma DB"
    PHASE0_JOB_ID=""
    if [ ! -d "storage/chroma/pubmed_full/cosine" ]; then
        echo "ERROR: --skip-phase0 used but Chroma DB not found."
        echo "       Run without --skip-phase0 to embed first."
        exit 1
    fi
    echo "Chroma DB: $(du -sh storage/chroma/pubmed_full | cut -f1)"
else
    echo ""
    echo "--- Submitting Phase 0 (embedding, GPU) ---"
    PHASE0_OUT=$(sbatch cluster_phase0_embed.sbatch)
    PHASE0_JOB_ID=$(echo "$PHASE0_OUT" | awk '{print $NF}')
    echo "Phase 0 job ID: $PHASE0_JOB_ID"
    echo "  Log: logs/phase0_embed_${PHASE0_JOB_ID}.out"
fi

if $PHASE0_ONLY; then
    echo ""
    echo "Phase 0 submitted. Re-run with --skip-phase0 after it completes."
    echo "Monitor: squeue -u \$USER"
    exit 0
fi

# ── Phase 1: Build queue (CPU, ~2-4h) ────────────────────────────────────────
echo ""
echo "--- Submitting Phase 1 (queue building, CPU) ---"
if [ -n "${PHASE0_JOB_ID:-}" ]; then
    PHASE1_OUT=$(sbatch --dependency=afterok:${PHASE0_JOB_ID} cluster_phase1_queue.sbatch)
else
    PHASE1_OUT=$(sbatch cluster_phase1_queue.sbatch)
fi
PHASE1_JOB_ID=$(echo "$PHASE1_OUT" | awk '{print $NF}')
echo "Phase 1 job ID: $PHASE1_JOB_ID"
echo "  Log: logs/phase1_queue_${PHASE1_JOB_ID}.out"

# ── Export: Artifacts (CPU, ~30min) ──────────────────────────────────────────
echo ""
echo "--- Submitting Export (artifacts, CPU) ---"
EXPORT_OUT=$(sbatch --dependency=afterok:${PHASE1_JOB_ID} cluster_export.sbatch)
EXPORT_JOB_ID=$(echo "$EXPORT_OUT" | awk '{print $NF}')
echo "Export job ID: $EXPORT_JOB_ID"
echo "  Log: logs/export_${EXPORT_JOB_ID}.out"

# ── Summary ──────────────────────────────────────────────────────────────────
echo ""
echo "========================================"
echo "  Jobs submitted successfully"
echo ""
if [ -n "${PHASE0_JOB_ID:-}" ]; then
echo "  Phase 0 (embed)   : $PHASE0_JOB_ID  [GPU, up to 6h]"
fi
echo "  Phase 1 (queue)   : $PHASE1_JOB_ID  [CPU, up to 4h]"
echo "  Export (artifacts): $EXPORT_JOB_ID  [CPU, up to 1.5h]"
echo ""
echo "Monitor all: squeue -u \$USER"
echo "Cancel all:  scancel $PHASE0_JOB_ID $PHASE1_JOB_ID $EXPORT_JOB_ID"
echo ""
echo "Phase 1 and Export will only start if the previous phase succeeds."
echo ""
echo "When complete, download from your local machine:"
echo "  scp -r bardamri@slurm.bgu.ac.il:~/ModularRAGOptimization/results/cluster_run/ ./"
echo "========================================"
