# Running Wander evals on RunPod

The laptop VM (15 GB RAM, 16 GB VRAM) is enough for rel-f1 and the small paper
graphs. Everything else in `queue_eval_nc.sh` asks for 128–256 GB of host RAM, and the
full RelBench databases have millions of rows. This is the recipe for a RunPod pod.

## 1. Pick a pod

RunPod has no scheduler: `submit.sh` sees no `sbatch` and runs each job in the
foreground, one dataset after another. Size the pod for the single largest job you
plan to run. In the deploy dialog look at the **RAM** column, not only the GPU.

| What you want to run | GPU | Host RAM | Notes |
|---|---|---|---|
| rel-f1, CORA/CITESEER/PUBMED, RelBench tasks subsampled to ≤100k nodes | any 24 GB (RTX 4090 / A5000) | ≥ 48 GB | cheapest; `--max-nodes 100000` |
| RelBench tasks subsampled to 200–500k nodes, mid NC graphs (CO_CS, WIKI_CS, ROMAN_EMPIRE) | A100 80 GB / L40S 48 GB | ≥ 128 GB | matches `MID_NC_MEM` |
| Full rel-trial / rel-event, large NC graphs (CO_PHYSICS, HM_CATEGORIES, CITY_*) | H100 SXM 80 GB (or 2×) | ≥ 256 GB | matches `LARGE_NC_MEM`; `SLURM_NUM_GPUS=2` shards phase B |
| rel-stack / rel-amazon / rel-hm / rel-avito **unsubsampled** | don't | — | 6–30 M nodes; always `--max-nodes` |

Template: **RunPod PyTorch 2.x** (Ubuntu 22.04, CUDA 12.x, Python 3.11). Any image with
a CUDA-enabled torch works; the setup script keeps the image's torch.

Storage:
* **Network volume** (same datacenter as the pod) mounted at `/workspace`, 100–200 GB.
  Put the repo, `raw_data/`, `checkpoints/` and `results/` there so you can stop the
  pod between runs and resume. Pod container disk is wiped on termination.
* RelBench databases land in `~/.cache/huggingface` by default. Either set
  `HF_HOME=/workspace/hf` (recommended) or accept re-downloading on a fresh pod.

Expose SSH (public key) when deploying; the web terminal drops long jobs.

## 2. Bootstrap

```bash
ssh root@<pod-ip> -p <port> -i ~/.ssh/<key>
export HF_HOME=/workspace/hf WANDB_MODE=disabled
cd /workspace
git clone https://github.com/ViswanathGanapathy/wander-gfm.git wander-gfm   # private: see "Cloning a private repo" below
cd wander-gfm
bash scripts/runpod_setup.sh                          # ~5 min; idempotent
```

`scripts/runpod_setup.sh` installs g++/python headers, creates a venv (or a conda env
named `wander` if conda exists on the image), installs `pybind11 numpy` → graph-walker
with `--no-build-isolation` → `requirements.txt` → `relbench`, downloads the checkpoint
from OSF into `checkpoints/pretrained_wander/pretrained_wander.pt`, and runs a CUDA /
graph-walker / registry smoke test. If the OSF link ever stops working, copy the file
from your laptop instead:

```bash
# on the laptop
scp -P <port> -i ~/.ssh/<key> ~/wander-gfm/checkpoints/pretrained_wander/pretrained_wander.pt \
    root@<pod-ip>:/workspace/wander-gfm/checkpoints/pretrained_wander/
```

Open a `tmux` session for everything below (`tmux new -s wander`, reattach with
`tmux a -t wander`).

### Cloning a private repo on the pod

The repo is private, so the pod needs a credential. Easiest: a fine-grained
GitHub token with *Contents: read* on this repo, then

```bash
git clone https://<TOKEN>@github.com/ViswanathGanapathy/wander-gfm.git /workspace/wander-gfm
```

or install `gh` on the pod and run `gh auth login` first. To skip all of this,
make the repo public: `gh repo edit ViswanathGanapathy/wander-gfm --visibility public`.
Baseline submodules under `third_party/` are not needed for Wander evals; do not
`--recurse-submodules`.

## 3. Sanity check

```bash
source /workspace/venv/bin/activate   # or: conda activate wander
cd /workspace/wander-gfm && export WANDB_MODE=disabled HF_HOME=/workspace/hf
EVAL_SEEDS=0 NC_SPLITS=first DATASETS=CORA bash slurm_scripts/eval/queue_eval_nc.sh
cat results/eval/nc_*/CORA/metrics.json
```

Expect accuracy in the high 70s / low 80s on split 0 (paper Table 4 reports the mean
over seeds). Anything near 0.3 means features or the checkpoint are not being read.

## 4. Convert RelBench tasks

```bash
# small, full database
python scripts/data/convert_relbench.py --dataset rel-f1 --task driver-dnf
python scripts/data/convert_relbench.py --dataset rel-f1 --task driver-top3
# medium: junction tables become edges, keep every labeled entity, BFS out to the budget
python scripts/data/convert_relbench.py --dataset rel-trial --task study-outcome --contract-link-tables --max-nodes 300000
python scripts/data/convert_relbench.py --dataset rel-event --task user-repeat   --contract-link-tables --max-nodes 300000
# large: the labeled sets alone are huge; labels are capped at half the budget (--max-labeled-fraction)
python scripts/data/convert_relbench.py --dataset rel-stack --task user-engagement --contract-link-tables --max-nodes 300000
python scripts/data/convert_relbench.py --dataset rel-stack --task user-badge      --contract-link-tables --max-nodes 300000
# list the classification tasks of any dataset
python scripts/data/convert_relbench.py --dataset rel-amazon --list-tasks
```

Each conversion writes `raw_data/RelBench/<dataset>__<task>/{data.pt,meta.json}`;
`meta.json` records the relation table, feature columns, per-split label counts, and
how many labels were dropped by the test > val > train precedence rule. Read it
before trusting a number: on rel-f1 the test set is 42 drivers.

Conversion is CPU/RAM bound: the full database is loaded with pandas before the
BFS subsample, so rel-stack (5.4 M rows) and rel-trial (5.9 M rows) want ~20–30 GB RAM
and rel-event (44.8 M rows, 30 M of them in `user_friends`) ~60 GB, even with
`--max-nodes`. Hosted row counts: rel-f1 74k, rel-stack 5.4M, rel-trial 5.9M, rel-event
44.8M; rel-amazon / rel-hm / rel-avito are larger still.

## 5. Evaluate

```bash
# every converted task, seeds 0 1 2, one temporal split
bash slurm_scripts/eval/queue_eval_relbench.sh
# a subset, single seed
DATASETS="RELBENCH_F1_DRIVER_DNF RELBENCH_TRIAL_STUDY_OUTCOME" EVAL_SEEDS=0 \
  bash slurm_scripts/eval/queue_eval_relbench.sh
# ablations
IGNORE_EDGE_TYPES=1 bash slurm_scripts/eval/queue_eval_relbench.sh   # collapse relations
IGNORE_FEATURES=1   bash slurm_scripts/eval/queue_eval_relbench.sh   # structure only
# two GPUs: train-KV on rank 0, test batches sharded
SLURM_NUM_GPUS=2 bash slurm_scripts/eval/queue_eval_relbench.sh
```

`queue_eval_relbench.sh` is `queue_eval_nc.sh` with `DROP_CONSTANT_TRAIN_FEATURES=0
FINAL_INDUCTIVE_ZSCORE=0` (only the entity table is labeled, so train-node column
statistics are meaningless for the other tables) and `NC_SPLITS=first`. All the usual
knobs pass through: `WANDER_WALK_NUM`, `WANDER_WALK_LEN`, `SIZE_TIER_WALK_NUM_DIV=2`,
`EVAL_BATCH_SIZE_NODE`, `WANDER_TRAIN_KV_CHUNK_SIZE`, `CKPT`, `EVAL_EXTRA_ARGS`.

Metrics: `results/eval/relbench_<stamp>/<KEY>/metrics.json` has `accuracy` and
`roc_auc` (binary AUROC on class 1, the RelBench metric). Jobs are resumable: rerun
with the same `EVAL_RUN_DIR` and finished batches are skipped; `SKIP_EXISTING=1` skips
datasets that already have `metrics.json`.

Memory knobs if a job is OOM-killed (host) or CUDA OOM:

| Symptom | Knob |
|---|---|
| killed while building walks / train KV | `WANDER_TRAIN_KV_CHUNK_SIZE=5000`, `SIZE_TIER_WALK_NUM_DIV=2` |
| CUDA OOM in phase B | `EVAL_BATCH_SIZE_NODE=32`, `WANDER_TEST_SAMPLES=8` |
| graph itself too big | reconvert with a smaller `--max-nodes` |

## 6. Bring results home, stop the pod

```bash
# on the laptop
rsync -avz -e "ssh -p <port> -i ~/.ssh/<key>" \
  root@<pod-ip>:/workspace/wander-gfm/results/eval/ ~/wander-gfm/results/eval/
```

Then **Stop** the pod (keeps the network volume, stops GPU billing) or **Terminate**
it. `raw_data/RelBench/*.pt` caches are worth keeping on the volume: converting
rel-stack takes longer than evaluating it.

## RunPod-specific gotchas

* No `sbatch`: the queue scripts' `dataset_mem` / `CLUSTER_MEM` values are ignored.
  Memory is whatever the pod has.
* `common.sh` picks Python as: active conda env named `$CONDA_ENV` → `conda activate
  $CONDA_ENV` → `python` on PATH. On an image that ships conda but where you used the
  venv, run `CONDA_ENV=base` or deactivate conda, or the script exits at
  `conda activate wander`.
* `main.py` calls `wandb.login()` unless `WANDB_MODE=disabled`; `common.sh` only sets
  that when wandb is **not** importable. Export it yourself (the setup script adds it to
  `~/.bashrc`).
* Spot/community pods can be reclaimed. Keep `EVAL_RUN_DIR` on `/workspace` and rerun
  the same command; `eval_batch_resume` picks up where it stopped.
* `torch.compile` is on for eval (`WANDER_COMPILE_EVAL=1`); first batch takes a minute.
  Set `WANDER_COMPILE_EVAL=0` on images with a mismatched triton.
