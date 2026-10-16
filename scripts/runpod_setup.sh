#!/usr/bin/env bash
# One-shot environment bootstrap for a RunPod GPU pod (or any bare Ubuntu+CUDA box).
#
#   cd /workspace && git clone https://github.com/ViswanathGanapathy/wander-gfm.git && cd wander-gfm
#   bash scripts/runpod_setup.sh
#
# What it does (idempotent, safe to re-run):
#   1. apt: g++ / python headers / tmux / rsync (graph-walker is a C++ extension)
#   2. python env: reuses an existing conda env named $CONDA_ENV if conda is present,
#      otherwise a venv at $VENV_DIR (default /workspace/venv). Keeps the image's
#      torch if it already has CUDA; otherwise installs torch from PyPI.
#   3. pip: pybind11+numpy first, graph-walker with --no-build-isolation, then
#      requirements.txt (the documented workaround for graph-walker/setup.py), relbench.
#   4. checkpoint: downloads pretrained_wander.pt from OSF into checkpoints/pretrained_wander/
#      unless it is already there (override with CKPT_URL / skip with SKIP_CKPT=1).
#   5. smoke test: CUDA visible, graph_walker import, dataset registry import.
#
# Knobs: CONDA_ENV=wander VENV_DIR=/workspace/venv SKIP_APT=1 SKIP_CKPT=1 CKPT_URL=...
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
CONDA_ENV="${CONDA_ENV:-wander}"
VENV_DIR="${VENV_DIR:-/workspace/venv}"
CKPT_URL="${CKPT_URL:-https://osf.io/download/78nwh/?view_only=a490e7fd33ef4ec2903cd2dbdbf5db15}"
CKPT_PATH="${REPO_ROOT}/checkpoints/pretrained_wander/pretrained_wander.pt"

log() { printf '\n=== %s ===\n' "$*"; }

# ---------------------------------------------------------------- 1. system deps
if [[ "${SKIP_APT:-0}" != "1" ]] && command -v apt-get >/dev/null 2>&1; then
  log "apt packages"
  export DEBIAN_FRONTEND=noninteractive
  (apt-get update -qq && apt-get install -y -qq --no-install-recommends \
      build-essential g++ python3-dev tmux rsync git curl ca-certificates) \
    || echo "WARN: apt-get failed (not root?). Continue; g++ and python headers must exist."
fi

# ---------------------------------------------------------------- 2. python env
if command -v conda >/dev/null 2>&1; then
  log "conda env ${CONDA_ENV}"
  CONDA_BASE="$(conda info --base)"
  # shellcheck source=/dev/null
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  if ! conda env list | awk '{print $1}' | grep -qx "${CONDA_ENV}"; then
    conda create -y -n "${CONDA_ENV}" python=3.11 pip
  fi
  conda activate "${CONDA_ENV}"
  ACTIVATE_HINT="conda activate ${CONDA_ENV}"
else
  log "venv ${VENV_DIR}"
  if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    python3 -m venv --system-site-packages "${VENV_DIR}"
  fi
  # shellcheck source=/dev/null
  source "${VENV_DIR}/bin/activate"
  ACTIVATE_HINT="source ${VENV_DIR}/bin/activate"
fi
PY=python
"$PY" -m pip install -q --upgrade pip wheel

# ---------------------------------------------------------------- 3. pip deps
log "torch"
if "$PY" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  echo "keeping image torch: $("$PY" -c 'import torch; print(torch.__version__)')"
else
  "$PY" -m pip install -q torch
fi

log "graph-walker (C++ extension) and requirements"
"$PY" -m pip install -q pybind11 numpy
"$PY" -m pip install -q --no-build-isolation -e ./graph-walker
# requirements.txt lists -e ./graph-walker again; already satisfied. torch stays pinned to the image.
"$PY" -m pip install -q -r requirements.txt
"$PY" -m pip install -q relbench

mkdir -p logs raw_data checkpoints results/eval

# ---------------------------------------------------------------- 4. checkpoint
if [[ "${SKIP_CKPT:-0}" != "1" ]]; then
  if [[ -s "${CKPT_PATH}" ]]; then
    log "checkpoint present: ${CKPT_PATH}"
  else
    log "downloading checkpoint (≈145 MB) -> ${CKPT_PATH}"
    mkdir -p "$(dirname "${CKPT_PATH}")"
    curl -L --fail --retry 3 -o "${CKPT_PATH}.part" "${CKPT_URL}"
    mv "${CKPT_PATH}.part" "${CKPT_PATH}"
  fi
  ls -la "${CKPT_PATH}"
fi

# ---------------------------------------------------------------- 5. smoke test
log "smoke test"
export WANDB_MODE=disabled
"$PY" - <<'PYEOF'
import torch, networkx as nx
print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
import graph_walker
w = graph_walker.random_walks(nx.karate_club_graph(), n_walks=2, walk_len=5)
print("graph_walker ok", getattr(w, "shape", None))
from data.datasets import DATASET_REGISTRY
print("registry:", len(DATASET_REGISTRY), "datasets;",
      [k for k in DATASET_REGISTRY if k.startswith("RELBENCH")][:3], "...")
import relbench; print("relbench", relbench.__version__)
PYEOF

grep -q "WANDB_MODE=disabled" ~/.bashrc 2>/dev/null || echo 'export WANDB_MODE=disabled' >> ~/.bashrc

cat <<EOF

Done. Next (in tmux: 'tmux new -s wander'):
  ${ACTIVATE_HINT}
  cd ${REPO_ROOT} && export WANDB_MODE=disabled
  # sanity check on CORA (downloads ~2 MB, a few minutes):
  EVAL_SEEDS=0 NC_SPLITS=first DATASETS=CORA bash slurm_scripts/eval/queue_eval_nc.sh
  # convert RelBench tasks, then evaluate them:
  python scripts/data/convert_relbench.py --dataset rel-f1 --task driver-dnf
  python scripts/data/convert_relbench.py --dataset rel-trial --task study-outcome --max-nodes 300000
  bash slurm_scripts/eval/queue_eval_relbench.sh
  # results: results/eval/<run>/<DATASET>/metrics.json and eval_summary.txt
EOF
