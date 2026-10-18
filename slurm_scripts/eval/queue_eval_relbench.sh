#!/usr/bin/env bash
# Zero-shot NC eval of the pretrained checkpoint on converted RelBench tasks.
# Thin wrapper over queue_eval_nc.sh with RelBench-appropriate defaults:
#
#   * DROP_CONSTANT_TRAIN_FEATURES=0 and FINAL_INDUCTIVE_ZSCORE=0 — only the task's
#     entity table is labeled, so "constant on train nodes" / "train-node z-score"
#     would delete or blow up every feature owned by another table. The converter
#     already z-scores per table (feature_layout=shared).
#   * DATASETS defaults to every RELBENCH_* key whose cache exists under
#     raw_data/RelBench (build caches with scripts/data/convert_relbench.py).
#   * one split (RelBench has a single temporal split), seeds 0:1:2.
#
# Usage (from the repo root):
#   python scripts/data/convert_relbench.py --dataset rel-f1 --task driver-dnf
#   bash slurm_scripts/eval/queue_eval_relbench.sh
#   DATASETS="RELBENCH_F1_DRIVER_DNF" EVAL_SEEDS=0 bash slurm_scripts/eval/queue_eval_relbench.sh
#   IGNORE_EDGE_TYPES=1 bash slurm_scripts/eval/queue_eval_relbench.sh   # ablation: untyped edges
#   IGNORE_FEATURES=1  bash slurm_scripts/eval/queue_eval_relbench.sh    # ablation: structure only
# Do NOT sbatch this file.

set -euo pipefail

_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${_here}/../.." && pwd)"
cd "${REPO_ROOT}"

if [[ -z "${DATASETS:-}" ]]; then
  DATASETS="$(python3 - <<'PY'
import os, sys
sys.path.insert(0, os.getcwd())
from data.datasets.relbench_nc import RELBENCH_CLASSIFICATION_TASKS, VARIANTS, cache_path, discover_tags, registry_key
data_dir = os.environ.get("RAW_DATA_DIR", "raw_data")
tags = [None] + discover_tags(data_dir)
keys = [registry_key(d, t, m, b, g) for d, t in RELBENCH_CLASSIFICATION_TASKS for m, b in VARIANTS for g in tags
        if cache_path(data_dir, d, t, m, b, g).is_file()]
print(" ".join(keys))
PY
)"
  if [[ -z "${DATASETS}" ]]; then
    echo "ERROR: no converted RelBench tasks under raw_data/RelBench." >&2
    echo "Run: python scripts/data/convert_relbench.py --dataset rel-f1 --task driver-dnf" >&2
    exit 1
  fi
fi

export DATASETS
export DROP_CONSTANT_TRAIN_FEATURES="${DROP_CONSTANT_TRAIN_FEATURES:-0}"
export FINAL_INDUCTIVE_ZSCORE="${FINAL_INDUCTIVE_ZSCORE:-0}"
export NC_SPLITS="${NC_SPLITS:-first}"
export EVAL_SEEDS="${EVAL_SEEDS:-0:1:2}"
export EVAL_RUN_DIR="${EVAL_RUN_DIR:-${REPO_ROOT}/results/eval/relbench_$(date +%y%m%d_%H%M)}"
if [[ -n "${IGNORE_EDGE_TYPES:-}" ]]; then
  # eval_one_dataset.sh reads IGNORE_EDGE_TYPES from the environment directly.
  export IGNORE_EDGE_TYPES
fi

echo "=== RelBench zero-shot NC eval: ${DATASETS} (drop_constant=${DROP_CONSTANT_TRAIN_FEATURES}, final_zscore=${FINAL_INDUCTIVE_ZSCORE}, ignore_edge_types=${IGNORE_EDGE_TYPES:-0}, ignore_features=${IGNORE_FEATURES:-0}) ==="
exec bash "${_here}/queue_eval_nc.sh" "$@"
