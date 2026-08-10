#!/usr/bin/env bash
# cluster_setup.sh — One-time environment setup on BGU CIS HPC cluster
# =====================================================================
# Run once from the login node (slurm.bgu.ac.il) BEFORE submitting jobs.
#
# Usage:
#   scp -r /path/to/ModularRAGOptimization bardamri@slurm.bgu.ac.il:~/
#   ssh bardamri@slurm.bgu.ac.il
#   cd ~/ModularRAGOptimization
#   bash cluster_setup.sh

set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
ok()   { echo -e "${GREEN}[OK]${NC}   $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
fail() { echo -e "${RED}[FAIL]${NC} $*"; exit 1; }

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONDA_ENV="rag_env"

echo "========================================"
echo "  ModularRAG Cluster Setup"
echo "  $(date)"
echo "  Project: $PROJECT_DIR"
echo "  Conda env: $CONDA_ENV"
echo "========================================"

# 1. Verify we're on the cluster (not accidentally on a compute node)
hostname | grep -q "slurm" \
    || warn "Not on login node (hostname: $(hostname)) — proceeding anyway"

# 2. Load Anaconda module
echo ""
echo "--- Loading Anaconda ---"
module load anaconda 2>/dev/null \
    || module load anaconda3 2>/dev/null \
    || fail "Cannot load anaconda module. Run 'module avail' to check available modules."
ok "Anaconda loaded: $(conda --version)"

# 3. Create or update conda environment
echo ""
echo "--- Conda environment ---"
if conda env list | grep -q "^${CONDA_ENV} "; then
    ok "Environment '${CONDA_ENV}' already exists — updating"
    conda activate "${CONDA_ENV}" 2>/dev/null || source activate "${CONDA_ENV}"
else
    echo "Creating conda environment '${CONDA_ENV}' (Python 3.10)..."
    conda create -y -n "${CONDA_ENV}" python=3.10
    conda activate "${CONDA_ENV}" 2>/dev/null || source activate "${CONDA_ENV}"
    ok "Environment created"
fi

# 4. Install PyTorch (CUDA 11.8 — matches cluster CUDA 11.x)
echo ""
echo "--- Installing PyTorch ---"
pip install --upgrade pip -q
pip install -q \
    "torch>=2.0.0" \
    --index-url https://download.pytorch.org/whl/cu118 \
    || pip install -q "torch>=2.0.0"   # fallback: CPU-only (shouldn't happen on GPU node)
ok "PyTorch installed: $(python -c 'import torch; print(torch.__version__)')"

# 5. Install project dependencies
echo ""
echo "--- Installing project dependencies ---"
pip install -q \
    "sentence-transformers>=2.6.0" \
    "chromadb==1.0.15" \
    "datasets>=2.14.0" \
    "huggingface_hub>=0.20.0" \
    pandas \
    pyarrow \
    numpy \
    tqdm \
    python-dotenv \
    scipy \
    requests

pip install -q \
    "llama-index-core>=0.10.0" \
    "llama-index-embeddings-huggingface" \
    "llama-index-vector-stores-chroma"

if [ -f "$PROJECT_DIR/requirements.txt" ]; then
    pip install -q -r "$PROJECT_DIR/requirements.txt" || true
fi
ok "All dependencies installed"

# 6. Create storage directories
echo ""
echo "--- Storage directories ---"
mkdir -p "$PROJECT_DIR/storage/chroma/pubmed_full/cosine"
mkdir -p "$PROJECT_DIR/results/cluster_run"
mkdir -p "$PROJECT_DIR/logs"
ok "Directories created"

# 7. Storage quota check
echo ""
echo "--- Storage quota ---"
USED=$(du -sh "$HOME" 2>/dev/null | cut -f1 || echo "unknown")
AVAIL=$(df -h "$HOME" 2>/dev/null | awk 'NR==2 {print $4}' || echo "unknown")
echo "  Home used:      $USED"
echo "  Home available: $AVAIL"
warn "The full PubMed Chroma DB needs ~200-300 GB."
warn "If quota is too small, ask sysadmin (hpc@bgu.ac.il) for /storage/ allocation."
warn "Or set CHROMA_STORAGE_OVERRIDE env var in sbatch to point elsewhere."

# 8. Quick import check
echo ""
echo "--- Import check ---"
python - << 'PYEOF'
import sys
sys.path.insert(0, ".")
tests = [
    ("chromadb",                   lambda: __import__("chromadb").__version__),
    ("sentence_transformers",      lambda: __import__("sentence_transformers").__version__),
    ("torch",                      lambda: __import__("torch").__version__),
    ("datasets",                   lambda: __import__("datasets").__version__),
    ("llama_index.core",           lambda: "ok"),
]
for name, fn in tests:
    try:
        v = fn()
        print(f"  {name:<35} OK  {v}")
    except Exception as e:
        print(f"  {name:<35} FAIL  {e}")
PYEOF

echo ""
echo "========================================"
echo "  Setup complete!  $(date)"
echo "========================================"
echo ""
echo "Next steps:"
echo "  1. Copy your .env file:  scp .env bardamri@slurm.bgu.ac.il:~/ModularRAGOptimization/"
echo "  2. Submit jobs:          bash cluster_submit.sh"
echo ""
echo "Monitor jobs:    squeue -u \$USER"
echo "GPU status:      sinfo -p gpu"
