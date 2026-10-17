#!/usr/bin/env python3
"""Convert RelBench entity-classification tasks into Wander NC datasets.

Usage (from the repo root, ``pip install relbench`` first):

    # one task, full database (rel-f1 is small); one label per entity node
    python scripts/data/convert_relbench.py --dataset rel-f1 --task driver-dnf

    # same task, one timestamped node per RelBench label row (recommended; key RELBENCH_F1_DRIVER_DNF_LN)
    python scripts/data/convert_relbench.py --dataset rel-f1 --task driver-dnf --label-mode label_nodes

    # ... with class-balanced train labels (key RELBENCH_F1_DRIVER_DNF_LN_BAL)
    python scripts/data/convert_relbench.py --dataset rel-f1 --task driver-dnf --label-mode label_nodes --balance-train-labels

    # large database: BFS-subsample to 200k nodes around the labeled entities
    python scripts/data/convert_relbench.py --dataset rel-trial --task study-outcome --max-nodes 200000

    # every known classification task (see data/datasets/relbench_nc.py)
    python scripts/data/convert_relbench.py --all --max-nodes 200000

    # list the classification tasks of a dataset
    python scripts/data/convert_relbench.py --dataset rel-stack --list-tasks

Output: raw_data/RelBench/<dataset>__<task>/{data.pt,meta.json}. Evaluate with

    DATASETS="RELBENCH_F1_DRIVER_DNF" bash slurm_scripts/eval/queue_eval_nc.sh
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from data.datasets.relbench_convert import convert_relbench_task  # noqa: E402
from data.datasets.relbench_nc import (  # noqa: E402
    RELBENCH_CLASSIFICATION_TASKS,
    cache_dir,
    registry_key,
)


def _list_tasks(dataset: str) -> int:
    import relbench
    from relbench.base import TaskType

    ds = relbench.load_dataset(dataset)
    print(f"{dataset}: val={ds.val_timestamp.date()} test={ds.test_timestamp.date()}")
    for name in ds.get_task_names():
        try:
            tk = ds.load_task(name)
        except Exception as exc:  # noqa: BLE001
            print(f"  {name:28s} (failed to load: {exc})")
            continue
        tt = TaskType(tk.task_type).value
        ok = tt in ("binary_classification", "multiclass_classification")
        ent = getattr(tk, "entity_table", "-")
        print(f"  {name:28s} {tt:26s} entity={ent:16s} {'-> ' + registry_key(dataset, name) + ' / _LN' if ok else ''}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", help="RelBench dataset, e.g. rel-f1 (or Hub spec / local path)")
    p.add_argument("--task", help="task name, e.g. driver-dnf")
    p.add_argument("--all", action="store_true", help="convert every task in RELBENCH_CLASSIFICATION_TASKS")
    p.add_argument("--list-tasks", action="store_true", help="print the dataset's tasks and exit")
    p.add_argument("--data_dir", default=str(REPO_ROOT / "raw_data"), help="Wander raw data dir")
    p.add_argument("--max-nodes", type=int, default=None, help="BFS-subsample to N nodes (seeded at labeled entities)")
    p.add_argument("--max-categories", type=int, default=32)
    p.add_argument("--text-drop-ratio", type=float, default=0.9)
    p.add_argument("--label-policy", choices=("latest", "earliest"), default="latest")
    p.add_argument("--split-precedence", choices=("test_first", "train_first"), default="test_first")
    p.add_argument("--max-labeled-fraction", type=float, default=0.5)
    p.add_argument("--feature-layout", choices=("shared", "block"), default="shared",
                   help="shared: per-table embeddings in the same 64 columns (default); block: raw columns side by side")
    p.add_argument("--feature-width", type=int, default=64)
    p.add_argument("--contract-link-tables", action="store_true",
                   help="junction tables (2 fkeys, no other columns) become edges instead of nodes")
    p.add_argument("--balance-train-labels", action="store_true",
                   help="subsample majority class(es) among TRAIN labels to the rarest class (key suffix _BAL)")
    p.add_argument("--balance-max-ratio", type=float, default=1.0,
                   help="with --balance-train-labels: max (class count)/(rarest class count)")
    p.add_argument("--max-train-labels", type=int, default=None,
                   help="cap TRAIN labels (random, after balancing); bounds Wander's train-KV cache (~3 MB RAM per train node)")
    p.add_argument("--label-mode", choices=("entity", "label_nodes"), default="entity",
                   help="entity: one label per entity node (registry key RELBENCH_*); "
                        "label_nodes: one timestamped node per RelBench label row (key RELBENCH_*_LN)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--force", action="store_true", help="rebuild even if data.pt exists")
    args = p.parse_args(argv)

    if args.list_tasks:
        if not args.dataset:
            p.error("--list-tasks needs --dataset")
        return _list_tasks(args.dataset)

    if args.all:
        jobs = list(RELBENCH_CLASSIFICATION_TASKS)
    else:
        if not (args.dataset and args.task):
            p.error("pass --dataset and --task, or --all")
        jobs = [(args.dataset, args.task)]

    failures = []
    for ds, tk in jobs:
        out = cache_dir(args.data_dir, ds, tk, args.label_mode, args.balance_train_labels)
        if (out / "data.pt").is_file() and not args.force:
            print(f"[skip] {ds}/{tk}: {out / 'data.pt'} exists (--force to rebuild)")
            continue
        print(f"=== {ds}/{tk} [{args.label_mode}{', balanced' if args.balance_train_labels else ''}] -> "
              f"{registry_key(ds, tk, args.label_mode, args.balance_train_labels)} ===")
        try:
            convert_relbench_task(
                ds, tk, out_dir=out, max_nodes=args.max_nodes,
                max_categories=args.max_categories, text_drop_ratio=args.text_drop_ratio,
                label_policy=args.label_policy, split_precedence=args.split_precedence,
                max_labeled_fraction=args.max_labeled_fraction, seed=args.seed,
                feature_layout=args.feature_layout, feature_width=args.feature_width,
                contract_link_tables=args.contract_link_tables, label_mode=args.label_mode,
                balance_train_labels=args.balance_train_labels, balance_max_ratio=args.balance_max_ratio,
                max_train_labels=args.max_train_labels,
            )
        except Exception as exc:  # noqa: BLE001
            if len(jobs) == 1:
                raise
            print(f"[fail] {ds}/{tk}: {exc}")
            failures.append((ds, tk, str(exc)))
    if failures:
        print("FAILED:", failures)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
