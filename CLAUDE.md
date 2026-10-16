# Wander — random-walk graph foundation model

> Fresh workspace, separate from `~/rdblearn` and `~/LayerDAG`. Their CLAUDE.md files and
> the RAG-chatbot CLAUDE.md at `~/CLAUDE.md` are **not** relevant here.

## What this is
Clone of https://github.com/louistichelman/wander-gfm — code for *"To Learn is to
Wander: Learning Across Graphs and Tasks with Random Walks"* (arXiv:2610.06694).
**Wander is a graph foundation model whose shared interface is random walks**: one
pretrained checkpoint covers node classification, homogeneous link prediction, and
knowledge-graph link prediction, transferring across graphs, feature spaces, and
relational schemas. Walks are produced by a vendored C++ extension (`graph-walker/`,
based on kerighan/graph-walker 1.0.6).

Why the user cares (context from the rdblearn project): Wander claims transfer across
*relational schemas* — directly relevant to the DFS-traversal / RDB featurization work
in `~/rdblearn` (GFS traps, Griffin-vs-RDBLearn comparisons). A natural experiment:
run Wander on the gfs_family traversal datasets (random walks cover all join paths, so
it should be traversal-order invariant, like message passing).

## Environment (set up 2026-10-15, working)
`conda activate wander` — python 3.11, **torch 2.14.1+cu130** (CUDA works on the RTX
4090 Laptop), torch_geometric 2.8.1, graph-walker built from source (editable),
networkx/scipy/sklearn/pandas/xgboost/rdflib/ogb/wandb, nvidia-nccl-cu13.

**Install gotcha already solved**: `pip install -r requirements.txt` fails because
`graph-walker/setup.py` imports pybind11 at module level and there is no pyproject.toml,
so pip's isolated build env lacks it. Working order:
`pip install pybind11 numpy` → `pip install torch` →
`pip install --no-build-isolation -e ./graph-walker` → rest of requirements.
Smoke-tested: CUDA visible; `graph_walker.random_walks(nx.karate_club_graph(), n_walks=2,
walk_len=5)` returns (68, 5).

## Missing piece (user action)
The pretrained checkpoint is NOT in the repo. Download from
https://osf.io/2us6d/overview?view_only=a490e7fd33ef4ec2903cd2dbdbf5db15
and place at `checkpoints/pretrained_wander/pretrained_wander.pt`. Evaluation needs it;
pretraining from scratch is a 4-stage curriculum wanting 2 GPUs + 256G host RAM (this
laptop VM has ~15G — eval-scale work only here).

## Layout
- `main.py` — entry point; `config/` — YAML configs; `models/wander/` — the model
- `experiment/` — training/eval loops, mixed loaders, negative sampling
- `graph-walker/` — vendored C++ random-walk extension
- `slurm_scripts/` — `submit.sh` runs locally when sbatch is absent (this machine: local).
  Queue scripts (`queue_*.sh`) run in the current shell, one job per dataset — never sbatch them.
  Site knobs: SLURM_PARTITION/SLURM_GRES/CLUSTER_MEM/CONDA_ENV (default `wander`).
- `baselines/`, `third_party/` (pins in third_party/PINS), `scripts/setup_third_party.sh`
- dirs created: `logs/ raw_data/ checkpoints/ results/eval/` (datasets auto-download to raw_data/)

## Running
```bash
conda activate wander
cd ~/wander-gfm
export WANDB_MODE=disabled          # main.py uses wandb
# small check from the README (downloads CORA on first use; needs the checkpoint):
bash slurm_scripts/submit.sh slurm_scripts/eval/eval_one_dataset.sh CORA
# restrict any queue: DATASETS="CORA CITESEER" bash slurm_scripts/eval/queue_eval_nc.sh
# node classification: NC_SPLITS=first keeps split 0 only; seeds default 0:1:2
```
Paper-table → command mapping is in the repo README; dataset lists in
`slurm_scripts/paper_protocol.sh`. Beware memory: KG suite = 54 graphs; large NC
graphs request 128–256G (won't fit here — pick small datasets).

## Conventions / cautions
- `requirements-minilm.txt` only for Table 6 (MiniLM features); `requirements-priors.txt`
  only for GraphPFN/NodePFN prior ablations; baselines have separate envs (`baselines/setup_envs.sh`).
- This WSL2 VM: ~15G RAM, RTX 4090 Laptop (16G VRAM) — stay on small/medium datasets.
