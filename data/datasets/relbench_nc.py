"""RelBench entity-classification tasks as Wander node-classification datasets.

A RelBench database is turned into ONE homogeneous-indexed, multi-relational graph:

* every row of every table is a node (``x`` carries the table's non-key columns,
  placed in a per-table column block; a categorical ``__table__`` column marks the
  node type);
* every foreign key is a relation type: row ``r`` of table ``t`` with
  ``t.fk = p`` links to row ``p`` of the parent table. Edges are stored in both
  directions with the same type (same convention as the other NC loaders; the
  walker still sees direction via the transposed CSR);
* nodes of the task's entity table carry one label each. RelBench labels are
  per ``(entity, timestamp)``; we keep the row at the LATEST timestamp inside each
  split and assign every entity to exactly one split with precedence
  test > val > train (see ``scripts/data/convert_relbench.py``).

Everything else (per-type quantile transforms, PCA-64, z-score, one-hot labels)
is the normal ``DataSet`` pipeline: the cached file exposes the same
``x_numerical_mask / x_fraction_mask / x_categorical_mask`` attributes as the
GraphLand GraphPFN raw path, so ``--graphland_different_transform`` applies.

Registry keys are ``RELBENCH_<DATASET>_<TASK>`` with ``rel-`` stripped and
dashes mapped to underscores, e.g. ``RELBENCH_F1_DRIVER_DNF``. The ``_LN`` suffix
(``RELBENCH_F1_DRIVER_DNF_LN``) selects the ``label_nodes`` conversion: every
RelBench label row is its own timestamped node linked to its entity, so nothing is
deduplicated (cache dir ``<dataset>__<task>__labelnodes``). A further ``_BAL``
suffix selects a graph whose TRAIN labels were class-balanced by subsampling
(``--balance-train-labels``; cache dir suffix ``__balanced``). Extra tasks can be
registered with ``RELBENCH_EXTRA_TASKS="rel-x:task-a,rel-y:task-b"``.

Converted graphs are cached at ``<data_dir>/RelBench/<dataset>__<task>/data.pt``.
Build them explicitly with ``python -m scripts.data.convert_relbench`` (this needs
``pip install relbench`` and network access); the loader converts on first use
with default options when the cache is missing.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch_geometric.data import Data

ROOT_FOLDER = "RelBench"

# RelBench v1 entity-classification tasks (all binary, AUROC). Row counts are the
# hosted databases (every row becomes a node unless ``--contract-link-tables``);
# subsample anything beyond a few hundred thousand rows with ``--max-nodes``.
RELBENCH_CLASSIFICATION_TASKS: List[Tuple[str, str]] = [
    ("rel-f1", "driver-dnf"),           # 74k rows (fits the laptop)
    ("rel-f1", "driver-top3"),          # 74k rows
    ("rel-trial", "study-outcome"),     # 5.9M rows, ~2.9M of them in junction tables
    ("rel-event", "user-repeat"),       # 44.8M rows (user_friends alone 30.4M) -> contract + max-nodes
    ("rel-event", "user-ignore"),       # 44.8M rows
    ("rel-stack", "user-engagement"),   # 5.4M rows
    ("rel-stack", "user-badge"),        # 5.4M rows
    ("rel-amazon", "user-churn"),       # >20M rows
    ("rel-amazon", "item-churn"),       # >20M rows
    ("rel-hm", "user-churn"),           # >30M rows
    ("rel-avito", "user-visits"),       # >20M rows
    ("rel-avito", "user-clicks"),       # >20M rows
]


LABEL_MODES = ("entity", "label_nodes")
# Registry-key suffix and cache-dir suffix per label mode. ``entity`` keeps the
# original names so existing caches stay valid.
_MODE_KEY_SUFFIX = {"entity": "", "label_nodes": "_LN"}
_MODE_DIR_SUFFIX = {"entity": "", "label_nodes": "__labelnodes"}


_BAL_KEY_SUFFIX = {False: "", True: "_BAL"}
_BAL_DIR_SUFFIX = {False: "", True: "__balanced"}
VARIANTS = [(m, b) for m in LABEL_MODES for b in (False, True)]


def _tag_suffix(tag: Optional[str]) -> str:
    """Free-form conversion tag (``--tag ctxval``) -> key suffix ``_CTXVAL`` / dir suffix ``__ctxval``."""
    if not tag:
        return ""
    clean = "".join(ch if ch.isalnum() else "_" for ch in tag).strip("_")
    return clean


def registry_key(dataset: str, task: str, label_mode: str = "entity", balanced: bool = False,
                 tag: Optional[str] = None) -> str:
    ds = dataset.lower()
    if ds.startswith("rel-"):
        ds = ds[len("rel-"):]
    key = (f"RELBENCH_{ds}_{task}".upper().replace("-", "_")
           + _MODE_KEY_SUFFIX[label_mode] + _BAL_KEY_SUFFIX[bool(balanced)])
    t = _tag_suffix(tag)
    return key + (f"_{t.upper()}" if t else "")


def cache_dir(data_dir: str, dataset: str, task: str, label_mode: str = "entity", balanced: bool = False,
              tag: Optional[str] = None) -> Path:
    t = _tag_suffix(tag)
    return (Path(data_dir) / ROOT_FOLDER
            / f"{dataset}__{task}{_MODE_DIR_SUFFIX[label_mode]}{_BAL_DIR_SUFFIX[bool(balanced)]}"
              f"{'__' + t.lower() if t else ''}")


def cache_path(data_dir: str, dataset: str, task: str, label_mode: str = "entity", balanced: bool = False,
               tag: Optional[str] = None) -> Path:
    return cache_dir(data_dir, dataset, task, label_mode, balanced, tag) / "data.pt"


def _tags_from_env() -> List[str]:
    """``RELBENCH_TAGS="ctxval,nofeatbig"`` registers tagged variants of every task."""
    raw = os.environ.get("RELBENCH_TAGS", "").strip()
    return [t.strip() for t in raw.replace(";", ",").split(",") if t.strip()]


def discover_tags(data_dir: str) -> List[str]:
    """Tags of tagged caches already present under ``<data_dir>/RelBench`` (no env needed)."""
    root = Path(data_dir) / ROOT_FOLDER
    tags: set = set()
    if not root.is_dir():
        return []
    known = {"labelnodes", "balanced"}
    for d in root.iterdir():
        if not (d / "data.pt").is_file():
            continue
        parts = d.name.split("__")[2:]  # dataset__task__<suffixes...>
        for ptag in parts:
            if ptag not in known:
                tags.add(ptag)
    return sorted(tags)


def load_relbench_nc(
    data_dir: str,
    dataset: str,
    task: str,
    *,
    label_mode: str = "entity",
    balanced: bool = False,
    tag: Optional[str] = None,
    force: bool = False,
    **convert_kwargs: Any,
) -> Data:
    """Load (or build) the converted RelBench task graph."""
    path = cache_path(data_dir, dataset, task, label_mode, balanced, tag)
    if path.is_file() and not force:
        data = torch.load(path, map_location="cpu", weights_only=False)
    else:
        if tag:
            raise FileNotFoundError(
                f"{path} missing: tagged variants must be built explicitly with "
                f"scripts/data/convert_relbench.py --tag {tag} (options are not derivable from the tag)"
            )
        # Lazy import: relbench is only needed to build the cache.
        from .relbench_convert import convert_relbench_task

        data = convert_relbench_task(
            dataset, task, out_dir=cache_dir(data_dir, dataset, task, label_mode, balanced),
            label_mode=label_mode, balance_train_labels=balanced, **convert_kwargs
        )
    if not hasattr(data, "num_nodes") or data.num_nodes is None:
        data.num_nodes = int(data.x.size(0))
    return data


def _mod(dataset: str, task: str, label_mode: str = "entity", balanced: bool = False,
         tag: Optional[str] = None) -> ModuleType:
    key = registry_key(dataset, task, label_mode, balanced, tag)
    m = ModuleType(key.lower())
    m.NAME = (f"{dataset}/{task}" + ("" if label_mode == "entity" else f" [{label_mode}]")
              + (" [balanced]" if balanced else "") + (f" [{tag}]" if tag else ""))
    m.RELBENCH_TAG = tag
    m.REGISTRY_KEY = key
    m.RELBENCH_DATASET = dataset
    m.RELBENCH_TASK = task
    m.RELBENCH_LABEL_MODE = label_mode
    m.RELBENCH_BALANCED = balanced
    m.HAS_NODE_FEATURES = True
    m.HAS_NODE_LABELS = True
    m.HAS_PREDEFINED_NODE_SPLIT = True   # temporal split from RelBench
    m.HAS_PREDEFINED_EDGE_SPLIT = False

    def load_base_dataset(
        data_dir: str,
        *,
        graphland_different_transform: bool = False,
        graphland_categorical_as_ordinals: bool = False,
    ) -> Data:
        # The two GraphLand flags only steer apply_feature_preprocessing; the
        # cached graph already carries the per-type column masks either way.
        return load_relbench_nc(data_dir, dataset, task, label_mode=label_mode, balanced=balanced, tag=tag)

    m.load_base_dataset = load_base_dataset
    return m


def _extra_tasks_from_env() -> List[Tuple[str, str]]:
    raw = os.environ.get("RELBENCH_EXTRA_TASKS", "").strip()
    out: List[Tuple[str, str]] = []
    if not raw:
        return out
    for item in raw.replace(";", ",").split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(
                f"RELBENCH_EXTRA_TASKS entry {item!r} must look like 'rel-f1:driver-dnf'"
            )
        ds, tk = item.split(":", 1)
        out.append((ds.strip(), tk.strip()))
    return out


def build_relbench_registry() -> Dict[str, ModuleType]:
    reg: Dict[str, ModuleType] = {}
    tags: List[Optional[str]] = [None] + _tags_from_env()
    for extra in discover_tags(os.environ.get("RAW_DATA_DIR", "raw_data")):
        if extra not in tags:
            tags.append(extra)
    for ds, tk in RELBENCH_CLASSIFICATION_TASKS + _extra_tasks_from_env():
        for mode, bal in VARIANTS:
            for tag in tags:
                m = _mod(ds, tk, mode, bal, tag)
                reg[m.REGISTRY_KEY] = m
    return reg


def register_relbench_datasets(registry: Dict[str, Any]) -> None:
    """Merge RelBench entity-classification tasks into DATASET_REGISTRY."""
    registry.update(build_relbench_registry())


def read_meta(data_dir: str, dataset: str, task: str) -> dict:
    p = cache_dir(data_dir, dataset, task) / "meta.json"
    with open(p) as f:
        return json.load(f)
