#!/usr/bin/env python3
"""
run_cloud_phase1.py — Fully-automated AWS cloud Phase 1
========================================================
Launches a GPU instance on AWS, embeds full PubMed (36M docs), builds the
scoring queue, exports a rich safety-net bundle, and downloads everything
back to your local machine.

After this script completes you can run Phase 2 locally:
    python score_from_queue.py --queue results/cloud_runs/<session>/scoring_queue.parquet

Prerequisites (run once):
    pip install boto3 paramiko
    aws configure                     # or export AWS_* env vars
    # Your IAM user needs:
    #   ec2:RunInstances / TerminateInstances / DescribeInstances
    #   ec2:CreateKeyPair / DeleteKeyPair / DescribeKeyPairs
    #   ec2:CreateSecurityGroup / AuthorizeSecurityGroupIngress / DescribeSecurityGroups
    #   ec2:DescribeImages / ec2:DescribeVpcs / ec2:DescribeSubnets

Usage:
    python run_cloud_phase1.py                        # full pipeline
    python run_cloud_phase1.py --check                # prerequisites only
    python run_cloud_phase1.py --skip-embedding       # skip Phase 0 (Chroma exists on instance)
    python run_cloud_phase1.py --resume SESSION_DIR   # resume after failure
    python run_cloud_phase1.py --dry-run              # plan without creating resources
    python run_cloud_phase1.py --region eu-west-1 --instance-type p3.2xlarge
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tarfile
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR  = PROJECT_ROOT / "results" / "cloud_runs"

# ---------------------------------------------------------------------------
# Tuneable configuration
# ---------------------------------------------------------------------------

DEFAULTS = dict(
    region             = "us-east-1",
    instance_type      = "g4dn.xlarge",   # 1×T4 16 GB, 4 vCPU
    key_name           = "modular-rag-key",
    security_group     = "modular-rag-phase1-sg",
    remote_user        = "ubuntu",
    remote_dir         = "/home/ubuntu/ModularRAGOptimization",
    ebs_size_gb        = 500,
    num_queries        = 199,
    max_ranked_gt      = 100,
    max_ranked_query   = 100,
    embed_batch        = 2048,           # GPU batch for embed_and_rank
    pubmed_dataset     = "MedRAG/pubmed",
    embed_model        = "sentence-transformers/all-mpnet-base-v2",
    spot               = False,          # set True to use Spot (~60 % cheaper)
)

# Files we need from the project (excludes heavy data / results / storage)
UPLOAD_INCLUDE = [
    "configurations/",
    "experiments/",
    "scripts/",
    "utility/",
    "vector_db/",
    "matrics/",
    "QACollection/",
    "embed_full_pubmed.py",
    "embed_and_rank.py",
    "build_scoring_queue.py",
    "score_from_queue.py",
    "export_pool_artifacts.py",
    "requirements.txt",
    ".env",
]

# ---------------------------------------------------------------------------
# Remote bootstrap script (runs as ubuntu on the EC2 instance)
# ---------------------------------------------------------------------------

BOOTSTRAP_SH = r"""#!/usr/bin/env bash
set -euo pipefail
echo "=== Bootstrap started ==="

nvidia-smi || echo "WARNING: nvidia-smi not available"
python3 --version

cd {remote_dir}
pip install -q --upgrade pip

# Pin chromadb — must match local project (build_scoring_queue.py imports it)
pip install -q "chromadb==1.0.15"

# Core ML stack
pip install -q \
    "torch>=2.0.0" \
    "sentence-transformers>=2.6.0" \
    "datasets>=2.14.0" \
    "huggingface_hub>=0.20.0" \
    pyarrow pandas numpy tqdm scipy requests

# llama-index (needed by build_scoring_queue.py and export_pool_artifacts.py)
pip install -q \
    "llama-index-core>=0.10.0" \
    "llama-index-embeddings-huggingface" \
    "llama-index-vector-stores-chroma"

# Project requirements last (so pinned versions above are not downgraded)
pip install -q -r requirements.txt || true

echo "=== Bootstrap complete ==="
"""

# ---------------------------------------------------------------------------
# Remote export script — extracts embeddings + rich metadata from Chroma
# (upload + run after build_scoring_queue.py completes)
# ---------------------------------------------------------------------------

EXPORT_SCRIPT_CONTENT = r'''#!/usr/bin/env python3
"""
export_pool_artifacts.py — Safety-net artifact exporter (runs on remote)
=========================================================================
After build_scoring_queue.py, this script reads the Chroma DB and the
scoring_queue.parquet to export a self-contained bundle:

    scoring_queue.parquet    — main Phase-2 input (already exists)
    pool_embeddings.npy      — float32 [n_unique_docs × dim]
    pool_doc_ids.json        — ordered list of doc IDs matching rows in .npy
    pool_metadata.parquet    — PMID, title, text per pool doc
    gt_embeddings.npy        — float32 [n_queries × dim]
    gt_doc_ids.json          — GT doc ID per query (index = query_idx - 1)
    run_metadata.json        — full provenance record
"""

from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from configurations.config import HF_MODEL_NAME
from utility.logger import logger


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--chroma-source", default="pubmed_full")
    args = ap.parse_args()

    run_dir: Path = args.run_dir.resolve()
    queue_path = run_dir / "scoring_queue.parquet"
    if not queue_path.exists():
        sys.exit(f"ERROR: scoring_queue.parquet not found in {run_dir}")

    logger.info(f"[EXPORT] Loading queue: {queue_path}")
    df = pd.read_parquet(queue_path)
    unique_doc_ids = df["doc_id"].unique().tolist()
    n_unique = len(unique_doc_ids)
    logger.info(f"[EXPORT] Pool: {n_unique:,} unique docs across {df['query_idx'].nunique()} queries")

    # --- Open Chroma ---
    from vector_db.chroma_vector_db import ChromaVectorDB
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from utility.distance_metrics import DistanceMetric

    logger.info("[EXPORT] Opening Chroma...")
    emb_model = HuggingFaceEmbedding(model_name=HF_MODEL_NAME)
    vdb = ChromaVectorDB(
        source_path=args.chroma_source,
        embedding_model=emb_model,
        distance_metric=DistanceMetric.COSINE,
    )
    collection = vdb.collection
    logger.info(f"[EXPORT] Collection has {collection.count():,} docs")

    # --- Fetch pool embeddings + metadata ---
    logger.info("[EXPORT] Fetching pool doc embeddings and metadata...")
    CHUNK = 400
    all_ids:   list[str]       = []
    all_embs:  list[np.ndarray]= []
    meta_rows: list[dict]      = []

    for start in range(0, n_unique, CHUNK):
        chunk_ids = unique_doc_ids[start: start + CHUNK]
        raw = collection.get(
            ids=chunk_ids,
            include=["embeddings", "documents", "metadatas"],
        )
        for did, emb, doc, meta in zip(
            raw["ids"], raw["embeddings"], raw["documents"], raw["metadatas"]
        ):
            if emb is None:
                continue
            all_ids.append(did)
            all_embs.append(np.array(emb, dtype=np.float32))
            pmid = meta.get("PMID") or meta.get("pubid")
            title = meta.get("Title") or meta.get("title") or ""
            meta_rows.append({
                "doc_id": did,
                "pmid": pmid,
                "title": title,
                "text_snippet": str(doc)[:500] if doc else "",
            })

        if (start // CHUNK) % 10 == 0:
            logger.info(f"[EXPORT]   {start + len(chunk_ids):,}/{n_unique:,} docs fetched")

    emb_matrix = np.stack(all_embs) if all_embs else np.empty((0, 768), dtype=np.float32)
    logger.info(f"[EXPORT] Embeddings matrix: {emb_matrix.shape}")

    # --- Fetch GT embeddings ---
    logger.info("[EXPORT] Fetching GT embeddings per query...")
    from experiments.global_correlation_experiment import (
        _gt_id_str, _gt_pmid_int_for_chroma, _get_gt_embedding_dual_path,
    )
    from scripts.qa_data_set_loader import load_qa_queries

    queries_path = run_dir / "queries.json"
    if queries_path.exists():
        queries = json.loads(queries_path.read_text())
    else:
        queries = load_qa_queries(df["query_idx"].max())

    gt_embs:   list[np.ndarray] = []
    gt_doc_ids: list[str]       = []

    for entry in queries:
        gt_id   = _gt_id_str(entry)
        gt_pmid = _gt_pmid_int_for_chroma(entry)
        gt_emb  = _get_gt_embedding_dual_path(collection, gt_id, gt_pmid)
        if gt_emb is not None:
            gt_embs.append(gt_emb.astype(np.float32))
        else:
            gt_embs.append(np.zeros(emb_matrix.shape[1] if emb_matrix.ndim > 1 else 768, dtype=np.float32))
        gt_doc_ids.append(gt_id)

    gt_matrix = np.stack(gt_embs) if gt_embs else np.empty((0, 768), dtype=np.float32)
    logger.info(f"[EXPORT] GT embeddings matrix: {gt_matrix.shape}")

    # --- Save artifacts ---
    logger.info("[EXPORT] Saving artifacts...")

    np.save(run_dir / "pool_embeddings.npy", emb_matrix)
    logger.info(f"[EXPORT] Saved pool_embeddings.npy  {emb_matrix.nbytes / 1e6:.1f} MB")

    (run_dir / "pool_doc_ids.json").write_text(json.dumps(all_ids))
    logger.info(f"[EXPORT] Saved pool_doc_ids.json  ({len(all_ids):,} ids)")

    meta_df = pd.DataFrame(meta_rows)
    meta_df.to_parquet(run_dir / "pool_metadata.parquet", index=False, compression="snappy")
    logger.info(f"[EXPORT] Saved pool_metadata.parquet  ({len(meta_df):,} rows)")

    np.save(run_dir / "gt_embeddings.npy", gt_matrix)
    (run_dir / "gt_doc_ids.json").write_text(json.dumps(gt_doc_ids))
    logger.info(f"[EXPORT] Saved gt_embeddings.npy + gt_doc_ids.json")

    # --- Run metadata ---
    run_meta = {
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "chroma_source": args.chroma_source,
        "chroma_total_docs": collection.count(),
        "pool_unique_docs": n_unique,
        "pool_docs_with_embeddings": len(all_ids),
        "n_queries": len(queries),
        "gt_embeddings_found": int((gt_matrix != 0).any(axis=1).sum()),
        "embedding_model": HF_MODEL_NAME,
        "embedding_dim": int(emb_matrix.shape[1]) if emb_matrix.ndim > 1 else None,
        "queue_rows": int(len(df)),
        "queue_is_gt_count": int(df["is_gt"].sum()),
        "queue_rag_failed_count": int(df["rag_failed"].sum()),
        "queue_dist_to_gt_mean": float(df["dist_to_gt"].mean()),
        "queue_dist_to_gt_std":  float(df["dist_to_gt"].std()),
        "artifacts": [
            "scoring_queue.parquet",
            "pool_embeddings.npy",
            "pool_doc_ids.json",
            "pool_metadata.parquet",
            "gt_embeddings.npy",
            "gt_doc_ids.json",
            "run_metadata.json",
            "queries.json",
            "correlation_pool_manifest.json",
        ],
    }
    (run_dir / "run_metadata.json").write_text(json.dumps(run_meta, indent=2))
    logger.info("[EXPORT] Saved run_metadata.json")

    print(f"\n=== Export complete ===")
    print(f"  Pool docs:        {n_unique:,}")
    print(f"  Embedding matrix: {emb_matrix.shape}  ({emb_matrix.nbytes/1e6:.0f} MB)")
    print(f"  GT matrix:        {gt_matrix.shape}")
    print(f"  Output dir:       {run_dir}")


if __name__ == "__main__":
    from datetime import datetime, timezone
    main()
'''

# ---------------------------------------------------------------------------
# SSH wrapper
# ---------------------------------------------------------------------------

class SSHClient:
    def __init__(self, host: str, user: str, key_path: str):
        import paramiko
        self._client = paramiko.SSHClient()
        self._client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self._host = host
        self._user = user
        self._key_path = key_path

    def connect(self, timeout: int = 300) -> None:
        import paramiko
        deadline = time.time() + timeout
        last_exc = None
        while time.time() < deadline:
            try:
                self._client.connect(
                    self._host,
                    username=self._user,
                    key_filename=self._key_path,
                    timeout=15,
                    banner_timeout=30,
                )
                print(f"  SSH connected to {self._host}")
                return
            except Exception as exc:
                last_exc = exc
                time.sleep(10)
        raise RuntimeError(f"SSH connection timed out after {timeout}s: {last_exc}")

    def run(self, cmd: str, check: bool = True, timeout: int = 21600) -> tuple[str, str]:
        """Execute command; return (stdout, stderr). Streams stdout live."""
        _, stdout, stderr = self._client.exec_command(cmd, get_pty=True, timeout=timeout)
        out_lines = []
        for line in iter(stdout.readline, ""):
            print(f"  [remote] {line}", end="")
            out_lines.append(line)
        out = "".join(out_lines)
        err = stderr.read().decode(errors="replace")
        rc  = stdout.channel.recv_exit_status()
        if check and rc != 0:
            raise RuntimeError(f"Remote command failed (rc={rc}):\n  CMD: {cmd}\n  STDERR: {err[-2000:]}")
        return out, err

    def put(self, local: str | Path, remote: str) -> None:
        sftp = self._client.open_sftp()
        sftp.put(str(local), remote)
        sftp.close()

    def put_string(self, content: str, remote: str) -> None:
        import io
        sftp = self._client.open_sftp()
        sftp.putfo(io.BytesIO(content.encode()), remote)
        sftp.close()

    def get(self, remote: str, local: str | Path) -> None:
        sftp = self._client.open_sftp()
        sftp.get(remote, str(local))
        sftp.close()

    def get_dir(self, remote_dir: str, local_dir: Path) -> None:
        """Recursively download a directory."""
        sftp = self._client.open_sftp()
        self._get_dir_recursive(sftp, remote_dir, local_dir)
        sftp.close()

    def _get_dir_recursive(self, sftp, remote: str, local: Path) -> None:
        import stat as _stat
        local.mkdir(parents=True, exist_ok=True)
        for attr in sftp.listdir_attr(remote):
            rpath = f"{remote}/{attr.filename}"
            lpath = local / attr.filename
            if _stat.S_ISDIR(attr.st_mode):
                self._get_dir_recursive(sftp, rpath, lpath)
            else:
                sftp.get(rpath, str(lpath))

    def close(self) -> None:
        self._client.close()


# ---------------------------------------------------------------------------
# AWS helpers
# ---------------------------------------------------------------------------

def _boto3_ec2(region: str):
    import boto3
    return boto3.client("ec2", region_name=region)


def find_dlami(ec2, region: str) -> str:
    """Return the latest Deep Learning AMI (Ubuntu 22.04, GPU) image ID."""
    resp = ec2.describe_images(
        Owners=["amazon"],
        Filters=[
            {"Name": "name",         "Values": ["Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)*"]},
            {"Name": "state",        "Values": ["available"]},
            {"Name": "architecture", "Values": ["x86_64"]},
        ],
    )
    imgs = sorted(resp["Images"], key=lambda x: x["CreationDate"], reverse=True)
    if not imgs:
        # Fallback: plain Ubuntu 22.04
        resp2 = ec2.describe_images(
            Owners=["099720109477"],  # Canonical
            Filters=[
                {"Name": "name",   "Values": ["ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*"]},
                {"Name": "state",  "Values": ["available"]},
            ],
        )
        imgs = sorted(resp2["Images"], key=lambda x: x["CreationDate"], reverse=True)
    if not imgs:
        raise RuntimeError("No suitable AMI found. Specify --ami explicitly.")
    ami = imgs[0]["ImageId"]
    print(f"  AMI: {ami} ({imgs[0]['Name'][:70]})")
    return ami


def ensure_key_pair(ec2, key_name: str, key_path: Path) -> str:
    """Create key pair if it doesn't exist; save PEM locally. Returns key_name."""
    existing = ec2.describe_key_pairs(Filters=[{"Name": "key-name", "Values": [key_name]}])
    if existing["KeyPairs"]:
        if not key_path.exists():
            # Fall back to ~/.ssh/<key_name>.pem
            ssh_fallback = Path.home() / ".ssh" / f"{key_name}.pem"
            if ssh_fallback.exists():
                import shutil
                key_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ssh_fallback, key_path)
                key_path.chmod(0o600)
                print(f"  Key pair '{key_name}' copied from {ssh_fallback}")
            else:
                raise FileNotFoundError(
                    f"Key pair '{key_name}' exists in AWS but PEM not found locally at {key_path} or {ssh_fallback}.\n"
                    f"Either delete the key pair in AWS Console or put the PEM at {key_path}."
                )
        print(f"  Key pair '{key_name}' already exists.")
        return key_name

    print(f"  Creating key pair '{key_name}'...")
    resp = ec2.create_key_pair(KeyName=key_name)
    key_path.parent.mkdir(parents=True, exist_ok=True)
    key_path.write_text(resp["KeyMaterial"])
    key_path.chmod(0o600)
    print(f"  PEM saved to: {key_path}")
    return key_name


def ensure_security_group(ec2, sg_name: str, region: str) -> str:
    """Create SG with SSH ingress if not present. Returns group ID."""
    existing = ec2.describe_security_groups(Filters=[{"Name": "group-name", "Values": [sg_name]}])
    if existing["SecurityGroups"]:
        gid = existing["SecurityGroups"][0]["GroupId"]
        print(f"  Security group '{sg_name}' exists: {gid}")
        return gid

    print(f"  Creating security group '{sg_name}'...")
    # Find default VPC
    vpcs = ec2.describe_vpcs(Filters=[{"Name": "isDefault", "Values": ["true"]}])
    if not vpcs["Vpcs"]:
        raise RuntimeError("No default VPC found. Create one or specify VpcId.")
    vpc_id = vpcs["Vpcs"][0]["VpcId"]

    resp = ec2.create_security_group(
        GroupName=sg_name,
        Description="ModularRAG Phase1 - SSH only",
        VpcId=vpc_id,
    )
    gid = resp["GroupId"]

    ec2.authorize_security_group_ingress(
        GroupId=gid,
        IpPermissions=[{
            "IpProtocol": "tcp", "FromPort": 22, "ToPort": 22,
            "IpRanges": [{"CidrIp": "0.0.0.0/0", "Description": "SSH access"}],
        }],
    )
    print(f"  Created security group {gid}")
    return gid


def launch_instance(ec2, cfg: dict, ami: str, sg_id: str) -> dict:
    """Launch EC2 instance. Returns instance dict."""
    block_devs = [{
        "DeviceName": "/dev/sda1",
        "Ebs": {
            "VolumeSize": cfg["ebs_size_gb"],
            "VolumeType": "gp3",
            "DeleteOnTermination": True,
        },
    }]

    kwargs = dict(
        ImageId            = ami,
        InstanceType       = cfg["instance_type"],
        KeyName            = cfg["key_name"],
        SecurityGroupIds   = [sg_id],
        BlockDeviceMappings= block_devs,
        MinCount=1, MaxCount=1,
        TagSpecifications  = [{
            "ResourceType": "instance",
            "Tags": [
                {"Key": "Name",    "Value": "modular-rag-phase1"},
                {"Key": "Project", "Value": "ModularRAGOptimization"},
            ],
        }],
    )

    if cfg["spot"]:
        kwargs["InstanceMarketOptions"] = {
            "MarketType": "spot",
            "SpotOptions": {"SpotInstanceType": "one-time", "InstanceInterruptionBehavior": "terminate"},
        }

    resp = ec2.run_instances(**kwargs)
    inst = resp["Instances"][0]
    iid  = inst["InstanceId"]
    print(f"  Launched instance: {iid}")
    return inst


def wait_for_instance(ec2, instance_id: str, timeout: int = 600) -> str:
    """Wait until instance is running and has a public IP. Returns public IP."""
    print(f"  Waiting for instance {instance_id} to be ready...")
    waiter = ec2.get_waiter("instance_running")
    waiter.wait(InstanceIds=[instance_id])

    deadline = time.time() + timeout
    while time.time() < deadline:
        resp = ec2.describe_instances(InstanceIds=[instance_id])
        inst = resp["Reservations"][0]["Instances"][0]
        ip   = inst.get("PublicIpAddress")
        if ip:
            print(f"  Instance running: {ip}")
            return ip
        time.sleep(5)
    raise RuntimeError("Instance running but no public IP assigned.")


def terminate_instance(ec2, instance_id: str) -> None:
    ec2.terminate_instances(InstanceIds=[instance_id])
    print(f"  Terminated instance: {instance_id}")


# ---------------------------------------------------------------------------
# Project upload
# ---------------------------------------------------------------------------

def upload_project(ssh: SSHClient, remote_dir: str) -> None:
    """Create a tarball of source files and upload + extract on remote."""
    print("  Creating project tarball...")
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
        tmp_path = tmp.name

    with tarfile.open(tmp_path, "w:gz") as tar:
        for item in UPLOAD_INCLUDE:
            local_path = PROJECT_ROOT / item
            if local_path.exists():
                arcname = item.rstrip("/")
                tar.add(str(local_path), arcname=arcname)
            else:
                print(f"    [SKIP] Not found locally: {item}")

    size_mb = Path(tmp_path).stat().st_size / 1e6
    print(f"  Uploading tarball ({size_mb:.1f} MB)...")
    ssh.put(tmp_path, "/tmp/project.tar.gz")
    Path(tmp_path).unlink(missing_ok=True)

    ssh.run(f"mkdir -p {remote_dir} && tar -xzf /tmp/project.tar.gz -C {remote_dir} && echo OK")
    print("  Project uploaded and extracted.")


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def step_bootstrap(ssh: SSHClient, remote_dir: str) -> None:
    print("\n[BOOTSTRAP] Installing dependencies...")
    script = BOOTSTRAP_SH.format(remote_dir=remote_dir)
    ssh.put_string(script, "/tmp/bootstrap.sh")
    ssh.run("chmod +x /tmp/bootstrap.sh && /tmp/bootstrap.sh", timeout=1800)


def _pubmed_schema_flags(dataset: str) -> str:
    """Return --schema / field flags appropriate for the chosen dataset."""
    if "MedRAG" in dataset or "medrag" in dataset.lower():
        # MedRAG/pubmed flat Parquet fields (confirmed by dry-run probe):
        #   id      = file-reference like "pubmed23n0001_0" — NOT the PMID
        #   PMID    = actual integer PubMed ID  ← use this
        #   title   = article title
        #   content = abstract text
        return "--schema flat --pmid-field PMID --title-field title --abstract-field content"
    if "uiyunkim" in dataset.lower():
        return "--schema flat --pmid-field PMID --title-field ArticleTitle --abstract-field AbstractText"
    # ncbi/pubmed (legacy): nested XML via custom schema
    return "--schema ncbi_pubmed"


def step_embed_and_rank(ssh: SSHClient, cfg: dict, remote_dir: str) -> str:
    """
    Phase 0+1 replacement: stream all PubMed docs, embed on GPU, build
    scoring_queue.parquet directly — no Chroma/HNSW index needed.
    Returns remote run_dir path.
    """
    print("\n[PHASE 0+1] Streaming embed + rank (no HNSW)...")

    local_data = f"{remote_dir}/data/pubmed_parquet_slim"

    # Check if slim Parquet files already exist on EC2
    check_out, _ = ssh.run(
        f"ls {local_data}/*.parquet 2>/dev/null | wc -l", check=False, timeout=15
    )
    n_existing = int(check_out.strip() or "0")

    if n_existing < 100:
        # Need to download/convert the dataset first
        print(f"  [PHASE 0+1] Only {n_existing} Parquet files found — downloading dataset...")
        pubmed_local = f"{remote_dir}/data/pubmed_local"
        dl_cmd = f"""python3 - <<'PYEOF'
import time, sys
from huggingface_hub import snapshot_download
repo = '{cfg["pubmed_dataset"]}'
dst  = '{pubmed_local}'
for attempt in range(8):
    try:
        snapshot_download(repo_id=repo, repo_type='dataset',
                          local_dir=dst, ignore_patterns=['*.md','*.txt'],
                          max_workers=2)
        print('Download complete.')
        sys.exit(0)
    except Exception as exc:
        wait = 60 * (attempt + 1)
        print(f'Attempt {{attempt+1}} failed: {{exc}}. Retrying in {{wait}}s...')
        time.sleep(wait)
print('Download failed after 8 attempts.', file=sys.stderr)
sys.exit(1)
PYEOF"""
        ssh.run(dl_cmd, timeout=7200)
        print(f"  [PHASE 0+1] Dataset downloaded to {pubmed_local}")
        # slim Parquet conversion will happen inside embed_and_rank (via embed_full_pubmed path)
        # but embed_and_rank reads from pubmed_parquet_slim directly.
        # Run embed_full_pubmed dry-run to trigger JSONL→Parquet conversion:
        schema_flags = _pubmed_schema_flags(cfg["pubmed_dataset"])
        convert_cmd = (
            f"cd {remote_dir} && python3 embed_full_pubmed.py"
            f" --dataset {pubmed_local}"
            f" {schema_flags}"
            f" --dry-run 0"
            f" --no-resume"
            f" 2>&1 | head -100"
        )
        print("  [PHASE 0+1] Triggering JSONL→Parquet conversion via embed_full_pubmed dry-run...")
        ssh.run(convert_cmd, timeout=10800, check=False)
    else:
        print(f"  [PHASE 0+1] Found {n_existing} slim Parquet files — skipping download")

    # Kill any leftover embed session
    ssh.run("tmux kill-session -t phase0 2>/dev/null || true", check=False, timeout=10)

    rank_args = (
        f" --dataset {local_data}"
        f" --pmid-field PMID"
        f" --title-field title"
        f" --abstract-field content"
        f" --model {cfg['embed_model']}"
        f" --embed-batch {cfg['embed_batch']}"
        f" --top-k {cfg['max_ranked_query']}"
        f" --num-queries {cfg['num_queries']}"
        f" --output-dir {remote_dir}/results/cloud_run"
    )

    ssh.run(
        f"tmux new-session -d -s phase0 "
        f"'cd {remote_dir} && set -o pipefail && python3 embed_and_rank.py{rank_args} "
        f"2>&1 | tee /tmp/embed_log.txt; echo $? > /tmp/embed_exit.txt'",
        timeout=30,
    )
    print("  [PHASE 0+1] Detached in tmux. Polling every 5 min...")
    print(f"  SSH in: tmux attach -t phase0")

    import time as _time
    while True:
        _time.sleep(300)
        out, _ = ssh.run(
            "tmux has-session -t phase0 2>/dev/null && echo RUNNING || echo DONE",
            check=False, timeout=15,
        )
        if "DONE" in out:
            break
        tail, _ = ssh.run("tail -3 /tmp/embed_log.txt 2>/dev/null", check=False, timeout=10)
        for line in tail.strip().splitlines():
            print(f"  [remote] {line}")

    code, _ = ssh.run("cat /tmp/embed_exit.txt 2>/dev/null || echo 0", check=False, timeout=10)
    if code.strip() not in ("", "0"):
        raise RuntimeError(f"embed_and_rank failed (exit {code.strip()}). Check /tmp/embed_log.txt")

    # Find the run_dir
    out2, _ = ssh.run(
        f"ls -td {remote_dir}/results/cloud_run/staged_run_*/ 2>/dev/null | head -1",
        check=False, timeout=10,
    )
    remote_run_dir = out2.strip()
    if not remote_run_dir:
        raise RuntimeError("Could not find run_dir after embed_and_rank. Check /tmp/embed_log.txt")
    print(f"  [PHASE 0+1] Complete. Run dir: {remote_run_dir}")
    return remote_run_dir


def step_phase0_embed(ssh: SSHClient, cfg: dict, remote_dir: str) -> None:
    print("\n[PHASE 0] Embedding full PubMed into Chroma (GPU)...")
    schema_flags = _pubmed_schema_flags(cfg["pubmed_dataset"])

    # Quick smoke-test: stream 5 docs to verify dataset is reachable before
    # committing to the 8-hour full run.
    verify_cmd = (
        f"cd {remote_dir} && python3 embed_full_pubmed.py"
        f" --dataset {cfg['pubmed_dataset']}"
        f" {schema_flags}"
        f" --dry-run 5"
        f" --no-resume"
        f" 2>&1"
    )
    print("  [PHASE 0] Verifying dataset streams (dry-run 5 docs)...")
    try:
        out, _ = ssh.run(verify_cmd, timeout=300, check=True)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Dataset smoke-test failed for '{cfg['pubmed_dataset']}'. "
            f"Check dataset name / schema flags.\n{exc}"
        ) from exc
    print("  [PHASE 0] Dataset verified. Wiping any stale Chroma data before full embed...")
    ssh.run(
        f"rm -rf {remote_dir}/storage/chroma/pubmed_full"
        f" && rm -f {remote_dir}/.embed_checkpoint.json",
        timeout=60,
    )

    # Download dataset to EBS so GPU doesn't wait on network during embedding
    local_data = f"{remote_dir}/data/pubmed_local"
    print(f"  [PHASE 0] Downloading {cfg['pubmed_dataset']} to EBS (avoids streaming bottleneck)...")
    # snapshot_download is resumable: already-downloaded files are skipped on retry.
    # max_workers=2 avoids HuggingFace 429 rate-limit; retry up to 8 times.
    dl_script = (
        "import time; "
        "from huggingface_hub import snapshot_download; "
        f"repo='{cfg['pubmed_dataset']}'; dst='{local_data}'; "
        "ok=False; "
        "[\n"
        "  (snapshot_download(repo_id=repo,repo_type='dataset',"
        "local_dir=dst,ignore_patterns=['*.md','*.txt'],max_workers=2), "
        "setattr(__builtins__,'_ok',True)) or True "
        "  if not getattr(__builtins__,'_ok',False) else None "
        "  for _ in range(8)"
        "]"
    )
    # Simpler retry loop written as a heredoc to avoid quoting hell
    dl_cmd = f"""python3 - <<'PYEOF'
import time, sys
from huggingface_hub import snapshot_download
repo = '{cfg["pubmed_dataset"]}'
dst  = '{local_data}'
for attempt in range(8):
    try:
        snapshot_download(repo_id=repo, repo_type='dataset',
                          local_dir=dst, ignore_patterns=['*.md','*.txt'],
                          max_workers=2)
        print('Download complete.')
        sys.exit(0)
    except Exception as exc:
        wait = 60 * (attempt + 1)
        print(f'Attempt {{attempt+1}} failed: {{exc}}. Retrying in {{wait}}s...')
        time.sleep(wait)
print('Download failed after 8 attempts.', file=sys.stderr)
sys.exit(1)
PYEOF"""
    ssh.run(dl_cmd, timeout=7200)  # 2 h max (includes retries)
    print("  [PHASE 0] Download complete. Starting full embed from local files...")

    embed_args = (
        f" --dataset {local_data}"
        f" {schema_flags}"
        f" --model {cfg['embed_model']}"
        f" --embed-batch {cfg['embed_batch']}"
    )
    # Kill any leftover session, then launch embed detached inside tmux.
    # The process will keep running even if the SSH connection drops.
    ssh.run("tmux kill-session -t phase0 2>/dev/null || true", check=False, timeout=10)
    ssh.run(
        f"tmux new-session -d -s phase0 "
        f"'cd {remote_dir} && set -o pipefail && python3 embed_full_pubmed.py{embed_args} "
        f"2>&1 | tee /tmp/embed_log.txt; echo $? > /tmp/embed_exit.txt'",
        timeout=30,
    )
    print("  [PHASE 0] Embed detached. Polling every 2 min (safe to lose connection)...")
    print(f"  [PHASE 0] SSH in and run: tmux attach -t phase0  (to watch live)")

    # Poll until the tmux session ends (embed finished or crashed)
    import time as _time
    while True:
        _time.sleep(120)
        out, _ = ssh.run(
            "tmux has-session -t phase0 2>/dev/null && echo RUNNING || echo DONE",
            check=False, timeout=15,
        )
        if "DONE" in out:
            break
        tail, _ = ssh.run("tail -3 /tmp/embed_log.txt 2>/dev/null", check=False, timeout=10)
        for line in tail.strip().splitlines():
            print(f"  [remote] {line}")

    # Check exit code written by the tmux command
    code, _ = ssh.run("cat /tmp/embed_exit.txt 2>/dev/null || echo 0", check=False, timeout=10)
    if code.strip() not in ("", "0"):
        raise RuntimeError(f"Phase 0 embed failed (exit {code.strip()}). Check /tmp/embed_log.txt")


def step_phase1_queue(ssh: SSHClient, cfg: dict, remote_dir: str) -> str:
    """Build scoring queue. Returns remote run_dir path."""
    print("\n[PHASE 1] Building scoring queue from Chroma...")
    cmd = (
        f"cd {remote_dir} && set -o pipefail && python3 build_scoring_queue.py"
        f" --chroma-source pubmed_full"
        f" --num-queries {cfg['num_queries']}"
        f" --max-ranked-gt {cfg['max_ranked_gt']}"
        f" --max-ranked-query {cfg['max_ranked_query']}"
        f" --output-dir results/cloud_run"
        f" 2>&1 | tee /tmp/phase1_log.txt"
    )
    out, _ = ssh.run(cmd, timeout=14400, check=True)  # 4 h max; raises on non-zero exit

    # Extract run_dir from output
    remote_run_dir = None
    for line in out.splitlines():
        if "Scoring queue ready:" in line:
            parquet_path = line.split("Scoring queue ready:")[-1].strip()
            remote_run_dir = str(Path(parquet_path).parent)
            break
    if remote_run_dir is None:
        # Fallback: find most recent dir
        out2, _ = ssh.run(f"ls -td {remote_dir}/results/cloud_run/staged_run_*/ | head -1")
        remote_run_dir = out2.strip()

    print(f"  Remote run dir: {remote_run_dir}")
    return remote_run_dir


def step_export_artifacts(ssh: SSHClient, remote_run_dir: str, remote_dir: str, chroma_source: str) -> None:
    print("\n[EXPORT] Exporting rich metadata and embeddings...")
    cmd = (
        f"cd {remote_dir} && python3 export_pool_artifacts.py"
        f" --run-dir {remote_run_dir}"
        f" --chroma-source {chroma_source}"
        f" 2>&1 | tee /tmp/export_log.txt"
    )
    ssh.run(cmd, timeout=7200)


def step_download(ssh: SSHClient, remote_run_dir: str, session_dir: Path) -> None:
    """Download all artifacts from remote to local session_dir."""
    print(f"\n[DOWNLOAD] Downloading artifacts to {session_dir}...")
    session_dir.mkdir(parents=True, exist_ok=True)

    artifacts = [
        "scoring_queue.parquet",
        "pool_embeddings.npy",
        "pool_doc_ids.json",
        "pool_metadata.parquet",
        "gt_embeddings.npy",
        "gt_doc_ids.json",
        "run_metadata.json",
        "queries.json",
        "correlation_pool_manifest.json",
        "experiment_params.json",
    ]

    for fname in artifacts:
        remote_path = f"{remote_run_dir}/{fname}"
        local_path  = session_dir / fname
        try:
            ssh.get(remote_path, local_path)
            size = local_path.stat().st_size / 1e6
            print(f"  Downloaded: {fname}  ({size:.1f} MB)")
        except FileNotFoundError:
            print(f"  [SKIP] Not found on remote: {fname}")
        except Exception as exc:
            print(f"  [WARN] Failed to download {fname}: {exc}")

    # Also download logs for debugging
    for log in ["/tmp/embed_log.txt", "/tmp/phase1_log.txt", "/tmp/export_log.txt"]:
        try:
            ssh.get(log, session_dir / Path(log).name)
        except Exception:
            pass


def validate_artifacts(session_dir: Path) -> bool:
    """Validate the downloaded artifacts are complete and consistent."""
    print("\n[VALIDATE] Checking artifacts...")
    import numpy as np
    import pandas as pd

    ok = True

    queue_path = session_dir / "scoring_queue.parquet"
    if not queue_path.exists():
        print("  ERROR: scoring_queue.parquet missing — Phase 2 cannot run!")
        return False
    df = pd.read_parquet(queue_path)
    print(f"  scoring_queue.parquet: {len(df):,} rows, {df['query_idx'].nunique()} queries, {df['doc_id'].nunique():,} unique docs")

    emb_path = session_dir / "pool_embeddings.npy"
    if emb_path.exists():
        emb = np.load(emb_path)
        print(f"  pool_embeddings.npy:   {emb.shape}  ({emb.nbytes/1e6:.0f} MB)")
    else:
        print("  WARN: pool_embeddings.npy missing (safety net incomplete)")
        ok = False

    gt_path = session_dir / "gt_embeddings.npy"
    if gt_path.exists():
        gt = np.load(gt_path)
        print(f"  gt_embeddings.npy:     {gt.shape}")
    else:
        print("  WARN: gt_embeddings.npy missing")

    meta_path = session_dir / "pool_metadata.parquet"
    if meta_path.exists():
        meta = pd.read_parquet(meta_path)
        print(f"  pool_metadata.parquet: {len(meta):,} docs")
    else:
        print("  WARN: pool_metadata.parquet missing")

    run_meta_path = session_dir / "run_metadata.json"
    if run_meta_path.exists():
        rm = json.loads(run_meta_path.read_text())
        print(f"  run_metadata.json:     chroma_source={rm.get('chroma_source')},"
              f" pool={rm.get('pool_unique_docs'):,} docs,"
              f" dim={rm.get('embedding_dim')}")

    if ok:
        print(f"\n  All good. Run Phase 2 with:")
        print(f"    python score_from_queue.py --queue {queue_path}")
    return ok


# ---------------------------------------------------------------------------
# Session state (for resume)
# ---------------------------------------------------------------------------

def save_state(session_dir: Path, state: dict) -> None:
    (session_dir / "session_state.json").write_text(json.dumps(state, indent=2))


def load_state(session_dir: Path) -> dict:
    p = session_dir / "session_state.json"
    return json.loads(p.read_text()) if p.exists() else {}


# ---------------------------------------------------------------------------
# Prerequisites check
# ---------------------------------------------------------------------------

def check_prerequisites() -> bool:
    ok = True
    print("Checking prerequisites...\n")

    # boto3
    try:
        import boto3
        print(f"  [OK] boto3 {boto3.__version__}")
    except ImportError:
        print("  [FAIL] boto3 not installed — run: pip install boto3")
        ok = False

    # paramiko
    try:
        import paramiko
        print(f"  [OK] paramiko {paramiko.__version__}")
    except ImportError:
        print("  [FAIL] paramiko not installed — run: pip install paramiko")
        ok = False

    # AWS credentials
    try:
        import boto3
        sts = boto3.client("sts")
        identity = sts.get_caller_identity()
        print(f"  [OK] AWS credentials: account={identity['Account']}, arn={identity['Arn']}")
    except Exception as exc:
        print(f"  [FAIL] AWS credentials not configured: {exc}")
        print("         Run: aws configure")
        ok = False

    # .env
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        from dotenv import dotenv_values
        env = dotenv_values(env_path)
        has_gemini = bool(env.get("GEMINI_API_KEY"))
        print(f"  [{'OK' if has_gemini else 'WARN'}] .env GEMINI_API_KEY: {'set' if has_gemini else 'NOT SET (Phase 2 scoring will fail)'}")
    else:
        print(f"  [WARN] .env not found at {env_path} — Phase 2 credentials won't be uploaded")

    print()
    return ok


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(args) -> None:
    cfg = {**DEFAULTS}
    cfg.update({k: v for k, v in vars(args).items() if v is not None and k in DEFAULTS})

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    session_dir = RESULTS_DIR / f"session_{ts}"

    if args.resume:
        session_dir = Path(args.resume).resolve()
        state = load_state(session_dir)
        if getattr(args, "fresh", False):
            # Keep AWS resource IDs but reset all phase-completion flags so
            # the pipeline re-runs every step on the same instance.
            kept = {k: state[k] for k in ("instance_id", "public_ip", "ami", "sg_id") if k in state}
            state = kept
            save_state(session_dir, state)
            print(f"Resuming session (--fresh: phase flags cleared): {session_dir}")
        else:
            print(f"Resuming session: {session_dir}")
        print(f"  State: {state}")
    else:
        session_dir.mkdir(parents=True, exist_ok=True)
        state = {}

    key_path = session_dir / f"{cfg['key_name']}.pem"

    # --- Check prerequisites ---
    if not check_prerequisites():
        sys.exit("Prerequisites not met. Fix above issues and re-run.")

    if args.dry_run:
        print("Dry-run: prerequisites OK. Would launch:")
        print(f"  Region:        {cfg['region']}")
        print(f"  Instance:      {cfg['instance_type']} {'(Spot)' if cfg['spot'] else '(On-Demand)'}")
        print(f"  EBS:           {cfg['ebs_size_gb']} GB")
        print(f"  Queries:       {cfg['num_queries']}")
        print(f"  Max ranked:    GT={cfg['max_ranked_gt']}  Query={cfg['max_ranked_query']}")
        print(f"  Session dir:   {session_dir}")
        return

    ec2 = _boto3_ec2(cfg["region"])

    # --- AWS resources ---
    if not state.get("instance_id"):
        print(f"\n[AWS] Setting up resources in {cfg['region']}...")
        ami   = find_dlami(ec2, cfg["region"])
        ensure_key_pair(ec2, cfg["key_name"], key_path)
        sg_id = ensure_security_group(ec2, cfg["security_group"], cfg["region"])

        inst  = launch_instance(ec2, cfg, ami, sg_id)
        state["instance_id"] = inst["InstanceId"]
        state["ami"]         = ami
        state["sg_id"]       = sg_id
        save_state(session_dir, state)

    instance_id = state["instance_id"]

    if not state.get("public_ip"):
        pub_ip = wait_for_instance(ec2, instance_id)
        state["public_ip"] = pub_ip
        save_state(session_dir, state)

    pub_ip   = state["public_ip"]
    remote_dir = cfg["remote_dir"]

    print(f"\n  Instance: {instance_id}  IP: {pub_ip}")
    print(f"  SSH:  ssh -i {key_path} ubuntu@{pub_ip}")

    # --- SSH ---
    print(f"\n[SSH] Connecting (may take 2-3 min for instance to boot)...")
    ssh = SSHClient(pub_ip, cfg["remote_user"], str(key_path))
    ssh.connect(timeout=300)

    try:
        # --- Bootstrap ---
        if not state.get("bootstrapped"):
            upload_project(ssh, remote_dir)
            step_bootstrap(ssh, remote_dir)
            state["bootstrapped"] = True
            save_state(session_dir, state)

        # --- Phase 0+1: streaming embed + rank (no HNSW) ---
        if not state.get("queue_done") and not args.skip_embedding:
            remote_run_dir = step_embed_and_rank(ssh, cfg, remote_dir)
            state["embedding_done"]   = True
            state["queue_done"]       = True
            state["remote_run_dir"]   = remote_run_dir
            save_state(session_dir, state)
        elif args.skip_embedding:
            print("\n[PHASE 0+1] Skipped (--skip-embedding)")

        remote_run_dir = state["remote_run_dir"]

        # Export artifacts skipped — no Chroma index in streaming-rank mode.
        # scoring_queue.parquet is the direct output.
        if not state.get("export_done"):
            print("\n[EXPORT] Skipped (streaming-rank mode produces no Chroma index)")
            state["export_done"] = True
            save_state(session_dir, state)

        # --- Download ---
        if not state.get("download_done"):
            step_download(ssh, remote_run_dir, session_dir)
            state["download_done"] = True
            save_state(session_dir, state)

    except Exception as exc:
        print(f"\n[ERROR] Pipeline failed: {exc}")
        print(f"  Instance {instance_id} kept alive for debugging.")
        print(f"  SSH:  ssh -i {key_path} ubuntu@{pub_ip}")
        print(f"  Resume: python run_cloud_phase1.py --resume {session_dir}")
        save_state(session_dir, state)
        raise

    finally:
        ssh.close()

    # --- Validate ---
    validate_artifacts(session_dir)

    # --- Terminate ---
    if not args.keep_instance:
        print(f"\n[CLEANUP] Terminating instance {instance_id}...")
        terminate_instance(ec2, instance_id)
        state["terminated"] = True
        save_state(session_dir, state)
    else:
        print(f"\n[CLEANUP] Instance kept alive (--keep-instance).")
        print(f"  SSH:  ssh -i {key_path} ubuntu@{pub_ip}")

    print(f"\n{'='*60}")
    print(f"Phase 1 complete. Session: {session_dir}")
    print(f"\nNext step — run Phase 2 locally:")
    print(f"  python score_from_queue.py --queue {session_dir}/scoring_queue.parquet")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Automated AWS Phase 1: embed PubMed + build scoring queue",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--region",          default=None, help=f"AWS region (default: {DEFAULTS['region']})")
    ap.add_argument("--instance-type",   default=None, help=f"EC2 instance type (default: {DEFAULTS['instance_type']})")
    ap.add_argument("--ebs-size-gb",     default=None, type=int, help=f"EBS volume size GB (default: {DEFAULTS['ebs_size_gb']})")
    ap.add_argument("--spot",            action="store_true", help="Use Spot instance (~60 %% cheaper)")
    ap.add_argument("--num-queries",     default=None, type=int)
    ap.add_argument("--max-ranked-gt",   default=None, type=int)
    ap.add_argument("--max-ranked-query",default=None, type=int)
    ap.add_argument("--embed-batch",     default=None, type=int)
    ap.add_argument("--ami",             default=None, help="Override AMI ID (auto-detected if omitted)")
    ap.add_argument("--skip-embedding",  action="store_true", help="Skip Phase 0 (Chroma already built on instance)")
    ap.add_argument("--keep-instance",   action="store_true", help="Don't terminate instance after completion")
    ap.add_argument("--resume",          default=None, metavar="SESSION_DIR", help="Resume interrupted session")
    ap.add_argument("--dry-run",         action="store_true", help="Validate prerequisites only, don't create resources")
    ap.add_argument("--check",           action="store_true", help="Check prerequisites and exit")
    ap.add_argument("--fresh",           action="store_true",
                    help="When used with --resume: clear phase-completion flags so every step re-runs "
                         "(keeps the instance, does NOT start a new AWS session)")
    args = ap.parse_args()

    if args.check:
        ok = check_prerequisites()
        sys.exit(0 if ok else 1)

    run_pipeline(args)


if __name__ == "__main__":
    main()
