#!/usr/bin/env bash
# setup_cloud.sh — Bootstrap ModularRAGOptimization on a fresh AWS EC2 instance
# =============================================================================
# Target:  Ubuntu 22.04 Deep Learning AMI (GPU) on AWS
# Called:  Automatically by run_cloud_phase1.py  OR  manually for debugging
#
# Manual usage:
#   scp -i key.pem setup_cloud.sh ubuntu@<IP>:/tmp/
#   ssh -i key.pem ubuntu@<IP> "PROJECT_DIR=/home/ubuntu/ModularRAGOptimization bash /tmp/setup_cloud.sh"

set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
ok()   { echo -e "${GREEN}[OK]${NC}   $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
fail() { echo -e "${RED}[FAIL]${NC} $*"; exit 1; }

PROJECT_DIR="${PROJECT_DIR:-/home/ubuntu/ModularRAGOptimization}"

echo "========================================"
echo "  ModularRAG Cloud Bootstrap"
echo "  $(date)"
echo "  Project: $PROJECT_DIR"
echo "========================================"

# ---------------------------------------------------------------------------
# 1. System checks
# ---------------------------------------------------------------------------
echo ""
echo "--- System checks ---"

if nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "unknown")
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1 || echo "unknown")
    ok "GPU: $GPU_NAME  ($GPU_MEM)"
else
    warn "nvidia-smi not found — embedding will run on CPU (very slow for 36M docs)"
fi

ok "Python: $(python3 --version)"
ok "Disk free: $(df -h "$HOME" | awk 'NR==2 {print $4}')"
ok "RAM: $(awk '/MemTotal/ {printf "%.0f GB", $2/1024/1024}' /proc/meminfo)"

# ---------------------------------------------------------------------------
# 2. System packages (minimal)
# ---------------------------------------------------------------------------
echo ""
echo "--- System packages ---"
sudo apt-get update -qq
sudo apt-get install -y -qq git wget curl unzip htop
ok "System packages ready"

# ---------------------------------------------------------------------------
# 3. Python dependencies
# ---------------------------------------------------------------------------
echo ""
echo "--- Python dependencies ---"

pip install --upgrade pip -q

# Core ML stack
pip install -q \
    "torch>=2.0.0" \
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
    requests \
    boto3 \
    paramiko

# llama-index (version-pinned to match requirements.txt)
pip install -q \
    "llama-index-core>=0.10.0" \
    "llama-index-embeddings-huggingface" \
    "llama-index-vector-stores-chroma"

# Project requirements (non-breaking; ignore conflicts)
if [ -f "$PROJECT_DIR/requirements.txt" ]; then
    pip install -q -r "$PROJECT_DIR/requirements.txt" || true
    ok "requirements.txt installed"
fi

ok "Python dependencies installed"

# ---------------------------------------------------------------------------
# 4. HuggingFace cache on large volume
# ---------------------------------------------------------------------------
echo ""
echo "--- HF cache ---"

# If EBS is mounted at /data, use it to avoid filling /home
if mountpoint -q /data 2>/dev/null || [ -d /data ]; then
    export HF_HOME=/data/hf_cache
    export HF_HUB_CACHE=/data/hf_cache
    mkdir -p /data/hf_cache
    grep -q "HF_HOME" ~/.bashrc 2>/dev/null || echo "export HF_HOME=/data/hf_cache" >> ~/.bashrc
    grep -q "HF_HUB_CACHE" ~/.bashrc 2>/dev/null || echo "export HF_HUB_CACHE=/data/hf_cache" >> ~/.bashrc
    ok "HF cache → /data/hf_cache"
else
    mkdir -p "$HOME/.cache/huggingface"
    ok "HF cache → $HOME/.cache/huggingface"
fi

# ---------------------------------------------------------------------------
# 5. Validate project structure
# ---------------------------------------------------------------------------
echo ""
echo "--- Project validation ---"

cd "$PROJECT_DIR" || fail "Project dir not found: $PROJECT_DIR  (was it uploaded?)"

for f in embed_full_pubmed.py build_scoring_queue.py export_pool_artifacts.py score_from_queue.py configurations/config.py; do
    [ -f "$f" ] && ok "$f" || warn "$f NOT FOUND"
done

# Python syntax check
python3 - << 'PYEOF'
import sys
scripts = [
    "embed_full_pubmed.py",
    "build_scoring_queue.py",
    "export_pool_artifacts.py",
    "score_from_queue.py",
]
for s in scripts:
    try:
        with open(s) as fh:
            compile(fh.read(), s, "exec")
        print(f"  syntax OK: {s}")
    except (SyntaxError, FileNotFoundError) as e:
        print(f"  WARNING: {s}: {e}")
PYEOF
ok "Syntax check done"

# Import check
python3 - << 'PYEOF'
import sys
sys.path.insert(0, ".")
errors = []
try:
    from configurations.config import HF_MODEL_NAME, GLOBAL_EXPERIMENT_OUTPUT_DIR
    print(f"  config     OK  model={HF_MODEL_NAME}")
except Exception as e:
    errors.append(f"config: {e}")

try:
    import chromadb
    print(f"  chromadb   OK  v{chromadb.__version__}")
except Exception as e:
    errors.append(f"chromadb: {e}")

try:
    from sentence_transformers import SentenceTransformer
    print("  sentence-transformers  OK")
except Exception as e:
    errors.append(f"sentence_transformers: {e}")

try:
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  torch      OK  v{torch.__version__}  device={dev}")
except Exception as e:
    errors.append(f"torch: {e}")

if errors:
    print("WARNINGS:", errors)
PYEOF
ok "Import check done"

# Storage dirs
mkdir -p storage/chroma/pubmed_full/cosine
mkdir -p results/cloud_run
ok "Storage directories created"

# ---------------------------------------------------------------------------
# 6. .env check
# ---------------------------------------------------------------------------
echo ""
echo "--- .env ---"

if [ -f .env ]; then
    ok ".env present"
    grep -v "^#" .env | grep -q "INFERENCE_API_KEY=.\+" \
        && ok "INFERENCE_API_KEY is set" \
        || warn "INFERENCE_API_KEY not set (Phase 2 won't work)"
    grep -v "^#" .env | grep -q "INFERENCE_API_URL=.\+" \
        && ok "INFERENCE_API_URL is set" \
        || warn "INFERENCE_API_URL not set (Phase 2 won't work)"
else
    warn ".env not found — Phase 2 credentials not configured"
fi

# ---------------------------------------------------------------------------
# 7. Quick GPU embedding smoke test
# ---------------------------------------------------------------------------
echo ""
echo "--- Embedding smoke test (10 docs) ---"

python3 - << 'PYEOF'
import sys, time
sys.path.insert(0, ".")
from sentence_transformers import SentenceTransformer
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
t0 = time.time()
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device=device)
texts = ["PubMed biomedical research abstract sample " + str(i) for i in range(10)]
embs  = model.encode(texts, batch_size=10, show_progress_bar=False, normalize_embeddings=True)
elapsed = time.time() - t0
print(f"  10 docs in {elapsed:.2f}s | shape={embs.shape} | device={device}")
PYEOF
ok "Embedding smoke test passed"

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
echo ""
echo "========================================"
echo "  Bootstrap complete!  $(date)"
echo "========================================"
echo ""
echo "Commands to run in order:"
echo ""
echo "  cd $PROJECT_DIR"
echo ""
echo "  # Phase 0 — embed 36M PubMed (~1.5h on GPU)"
echo "  python3 embed_full_pubmed.py --embed-batch 256 2>&1 | tee /tmp/embed.log"
echo ""
echo "  # Phase 1 — build scoring queue (~1h on CPU)"
echo "  python3 build_scoring_queue.py \\"
echo "    --chroma-source pubmed_full \\"
echo "    --num-queries 199 \\"
echo "    --max-ranked-gt 100 \\"
echo "    --max-ranked-query 100 \\"
echo "    --output-dir results/cloud_run \\"
echo "    2>&1 | tee /tmp/phase1.log"
echo ""
echo "  # Export — safety-net artifacts (embeddings + metadata)"
echo "  python3 export_pool_artifacts.py \\"
echo "    --run-dir results/cloud_run/staged_run_XXXX \\"
echo "    --chroma-source pubmed_full"
echo ""
echo "  # Then download everything locally:"
echo "  scp -i key.pem -r ubuntu@\$(hostname -I | awk '{print \$1}'):$PROJECT_DIR/results/cloud_run/staged_run_XXXX/ ."
echo "========================================"
