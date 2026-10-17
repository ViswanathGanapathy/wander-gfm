"""Convert a RelBench entity-classification task into a Wander NC graph.

See ``data/datasets/relbench_nc.py`` for the graph construction and the
``RELBENCH_*`` registry; ``scripts/data/convert_relbench.py`` is the CLI.

Requires ``pip install relbench`` (>= 3.0, Hugging Face hosted datasets).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix
from torch_geometric.data import Data

SPLITS = ("train", "val", "test")
SPLIT_PRECEDENCE = {
    "test_first": ("test", "val", "train"),
    "train_first": ("train", "val", "test"),
}
LABEL_POLICIES = ("latest", "earliest")
FEATURE_LAYOUTS = ("shared", "block")
LABEL_MODES = ("entity", "label_nodes")
LABEL_TABLE = "__labels__"   # pseudo-table holding one node per label row (label_nodes mode)

MISSING_CAT = 0  # categorical code reserved for "missing / not this table"


# --------------------------------------------------------------------------- #
# Feature extraction
# --------------------------------------------------------------------------- #


@dataclass
class TableFeatures:
    table: str
    num: np.ndarray                 # [n, a] float32, NaN allowed
    frac: np.ndarray                # [n, b] float32 in [0, 1], NaN allowed
    cat: np.ndarray                 # [n, c] int64 codes, 0 = missing
    num_cols: List[str] = field(default_factory=list)
    frac_cols: List[str] = field(default_factory=list)
    cat_cols: List[str] = field(default_factory=list)
    dropped: Dict[str, str] = field(default_factory=dict)  # col -> reason
    cat_vocab: Dict[str, List[str]] = field(default_factory=dict)


def _is_listlike_column(ser: pd.Series) -> bool:
    nn = ser.dropna()
    if nn.empty:
        return False
    v = nn.iloc[0]
    return isinstance(v, (list, tuple, np.ndarray, dict))


def _numeric_values(ser: pd.Series) -> np.ndarray:
    return pd.to_numeric(ser, errors="coerce").to_numpy(dtype=np.float64, na_value=np.nan)


def _bucket_categorical(
    ser: pd.Series, max_categories: int
) -> Tuple[np.ndarray, List[str]]:
    """Map a string-like column to codes 1..K (0 = missing); rare values -> 'other'."""
    s = ser.astype("object").where(ser.notna(), None)
    counts = s.value_counts(dropna=True)
    keep = list(counts.index[: max(1, max_categories - 1)])
    vocab = [str(v) for v in keep]
    mapping = {v: i + 1 for i, v in enumerate(keep)}
    other_code = len(keep) + 1
    codes = np.zeros(len(s), dtype=np.int64)
    present = s.notna().to_numpy()
    vals = s.to_numpy(dtype=object)
    for i in np.nonzero(present)[0]:
        codes[i] = mapping.get(vals[i], other_code)
    if (codes == other_code).any():
        vocab.append("__other__")
    return codes, vocab


def featurize_table(
    df: pd.DataFrame,
    table: str,
    *,
    skip_cols: Sequence[str],
    time_origin: Optional[pd.Timestamp],
    max_categories: int,
    text_drop_ratio: float,
) -> TableFeatures:
    """Split a table's non-key columns into numerical / fraction / categorical blocks.

    * datetimes -> days since ``time_origin`` (numerical)
    * booleans and binary {0,1} numerics -> categorical
    * numerics with all finite values in [0, 1] -> fraction
    * other numerics -> numerical (NaN kept; imputed later by the Wander pipeline)
    * strings: dropped when near-unique (ids / free text: nunique >= text_drop_ratio
      * non-null rows and > 20 rows); otherwise top-(max_categories-1) + 'other'
    * list / dict valued columns -> dropped
    * constant columns -> dropped
    """
    n = len(df)
    out = TableFeatures(
        table=table,
        num=np.empty((n, 0), dtype=np.float32),
        frac=np.empty((n, 0), dtype=np.float32),
        cat=np.empty((n, 0), dtype=np.int64),
    )
    num_blocks: List[np.ndarray] = []
    frac_blocks: List[np.ndarray] = []
    cat_blocks: List[np.ndarray] = []
    skip = set(skip_cols)

    for col in df.columns:
        if col in skip:
            continue
        ser = df[col]
        if _is_listlike_column(ser):
            out.dropped[col] = "list-valued"
            continue
        nn = int(ser.notna().sum())
        if nn == 0:
            out.dropped[col] = "all-missing"
            continue
        if ser.dropna().nunique() <= 1:
            out.dropped[col] = "constant"
            continue

        if pd.api.types.is_datetime64_any_dtype(ser):
            ts = pd.to_datetime(ser, errors="coerce")
            origin = time_origin if time_origin is not None else ts.min()
            days = (ts - origin).dt.total_seconds().to_numpy(dtype=np.float64) / 86400.0
            num_blocks.append(days.astype(np.float32)[:, None])
            out.num_cols.append(col)
            continue

        if pd.api.types.is_bool_dtype(ser):
            codes = np.zeros(n, dtype=np.int64)
            present = ser.notna().to_numpy()
            codes[present] = ser[present].astype(int).to_numpy() + 1
            cat_blocks.append(codes[:, None])
            out.cat_cols.append(col)
            out.cat_vocab[col] = ["False", "True"]
            continue

        if pd.api.types.is_numeric_dtype(ser):
            vals = _numeric_values(ser)
            finite = vals[np.isfinite(vals)]
            uniq = np.unique(finite)
            if uniq.size <= 2 and np.isin(uniq, [0.0, 1.0]).all():
                codes = np.zeros(n, dtype=np.int64)
                m = np.isfinite(vals)
                codes[m] = vals[m].astype(np.int64) + 1
                cat_blocks.append(codes[:, None])
                out.cat_cols.append(col)
                out.cat_vocab[col] = ["0", "1"]
            elif finite.size > 0 and finite.min() >= 0.0 and finite.max() <= 1.0:
                frac_blocks.append(vals.astype(np.float32)[:, None])
                out.frac_cols.append(col)
            else:
                num_blocks.append(vals.astype(np.float32)[:, None])
                out.num_cols.append(col)
            continue

        # string-like / categorical / object
        nunique = int(ser.dropna().nunique())
        if nn > 20 and nunique >= text_drop_ratio * nn:
            out.dropped[col] = f"near-unique text/id ({nunique}/{nn})"
            continue
        codes, vocab = _bucket_categorical(ser, max_categories)
        cat_blocks.append(codes[:, None])
        out.cat_cols.append(col)
        out.cat_vocab[col] = vocab

    if num_blocks:
        out.num = np.concatenate(num_blocks, axis=1)
    if frac_blocks:
        out.frac = np.concatenate(frac_blocks, axis=1)
    if cat_blocks:
        out.cat = np.concatenate(cat_blocks, axis=1)
    return out


# --------------------------------------------------------------------------- #
# Graph construction
# --------------------------------------------------------------------------- #


def _table_offsets(table_sizes: Dict[str, int]) -> Dict[str, int]:
    offsets: Dict[str, int] = {}
    cur = 0
    for name, n in table_sizes.items():
        offsets[name] = cur
        cur += int(n)
    return offsets


def is_pure_link_table(table) -> bool:
    """A junction table: exactly two foreign keys and no other columns besides
    its primary key / time column. Rows of such tables carry no information of
    their own; ``--contract-link-tables`` turns them into direct parent-parent edges."""
    fks = list(table.fkey_col_to_pkey_table)
    if len(fks) != 2:
        return False
    allowed = set(fks) | {c for c in (table.pkey_col, table.time_col) if c is not None}
    return set(table.df.columns) <= allowed


def build_fkey_edges(
    db,
    table_names: Sequence[str],
    offsets: Dict[str, int],
    contracted: Sequence[str] = (),
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[str, str, str]]]:
    """Directed (child row -> parent row) edges, one relation per foreign key.

    ``contracted`` tables are not nodes: each of their rows becomes one edge
    between the two parent rows (relation ``(table, "fk1->fk2", parent2)``).
    """
    relations: List[Tuple[str, str, str]] = []
    srcs: List[np.ndarray] = []
    dsts: List[np.ndarray] = []
    types: List[np.ndarray] = []
    for t in contracted:
        table = db.table_dict[t]
        (fk_a, par_a), (fk_b, par_b) = list(table.fkey_col_to_pkey_table.items())
        rel_id = len(relations)
        relations.append((t, f"{fk_a}->{fk_b}", par_b))
        a = _numeric_values(table.df[fk_a])
        b = _numeric_values(table.df[fk_b])
        m = np.isfinite(a) & np.isfinite(b)
        a_i, b_i = a[m].astype(np.int64), b[m].astype(np.int64)
        ok = (a_i >= 0) & (a_i < len(db.table_dict[par_a])) & (b_i >= 0) & (b_i < len(db.table_dict[par_b]))
        srcs.append(a_i[ok] + offsets[par_a])
        dsts.append(b_i[ok] + offsets[par_b])
        types.append(np.full(int(ok.sum()), rel_id, dtype=np.int64))
    for t in table_names:
        table = db.table_dict[t]
        for fk_col, parent in table.fkey_col_to_pkey_table.items():
            rel_id = len(relations)
            relations.append((t, fk_col, parent))
            fk = _numeric_values(table.df[fk_col])
            m = np.isfinite(fk)
            if not m.any():
                continue
            parent_n = len(db.table_dict[parent])
            fk_i = fk[m].astype(np.int64)
            ok = (fk_i >= 0) & (fk_i < parent_n)
            rows = np.nonzero(m)[0][ok]
            srcs.append(rows + offsets[t])
            dsts.append(fk_i[ok] + offsets[parent])
            types.append(np.full(rows.shape[0], rel_id, dtype=np.int64))
    if srcs:
        src = np.concatenate(srcs)
        dst = np.concatenate(dsts)
        et = np.concatenate(types)
    else:
        src = dst = et = np.zeros(0, dtype=np.int64)
    edge_index = np.stack([src, dst], axis=0)
    return edge_index, et, relations


def symmetrize(edge_index: np.ndarray, edge_type: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Add reverse edges with the same relation id (NC convention in this repo).

    Done here instead of relying on ``DataSet._ensure_undirected_homogeneous`` because
    PyG ``to_undirected(edge_index, edge_type)`` coalesces parallel edges by SUMMING
    their types — a child table with two foreign keys into the same parent (e.g. a
    post's owner and last editor) would get corrupted relation ids.
    """
    rev = edge_index[[1, 0]]
    ei = np.concatenate([edge_index, rev], axis=1)
    et = np.concatenate([edge_type, edge_type])
    return ei, et


# --------------------------------------------------------------------------- #
# Labels
# --------------------------------------------------------------------------- #


def _one_label_per_entity(df: pd.DataFrame, entity_col: str, time_col: str, policy: str) -> pd.DataFrame:
    if policy not in LABEL_POLICIES:
        raise ValueError(f"label_policy must be one of {LABEL_POLICIES}, got {policy!r}")
    keep = "last" if policy == "latest" else "first"
    if time_col in df.columns:
        df = df.sort_values([time_col, entity_col], kind="stable")
    return df.drop_duplicates(subset=[entity_col], keep=keep)


def collect_entity_labels(
    task,
    *,
    label_policy: str,
    split_precedence: str,
) -> Tuple[Dict[str, Dict[int, Any]], Dict[str, Any]]:
    """Return ``{split: {entity_id: raw_label}}`` with each entity in exactly one split."""
    if split_precedence not in SPLIT_PRECEDENCE:
        raise ValueError(
            f"split_precedence must be one of {tuple(SPLIT_PRECEDENCE)}, got {split_precedence!r}"
        )
    assigned: Dict[int, str] = {}
    per_split: Dict[str, Dict[int, Any]] = {s: {} for s in SPLITS}
    stats: Dict[str, Any] = {}
    for split in SPLIT_PRECEDENCE[split_precedence]:
        df = task.get_table(split, mask_input_cols=False).df
        n_rows = len(df)
        df = df[df[task.target_col].notna()]
        ents_all = int(df[task.entity_col].nunique())
        df = _one_label_per_entity(df, task.entity_col, task.time_col, label_policy)
        n_dropped_overlap = 0
        for ent, lab in zip(df[task.entity_col].to_numpy(), df[task.target_col].to_numpy()):
            ent = int(ent)
            if ent in assigned:
                n_dropped_overlap += 1
                continue
            assigned[ent] = split
            per_split[split][ent] = lab
        stats[split] = {
            "label_rows": int(n_rows),
            "unique_entities": ents_all,
            "kept_entities": len(per_split[split]),
            "dropped_overlap_with_higher_precedence_split": n_dropped_overlap,
        }
    return per_split, stats


def encode_labels(per_split: Dict[str, Dict[int, Any]], task_type: str) -> Tuple[Dict[str, Dict[int, int]], List[str]]:
    raw_values = sorted({v for d in per_split.values() for v in d.values()}, key=lambda v: (str(type(v)), v))
    if task_type == "binary_classification":
        classes = [0, 1]
        mapping = {v: int(round(float(v))) for v in raw_values}
        bad = [v for v, c in mapping.items() if c not in (0, 1)]
        if bad:
            raise ValueError(f"binary task has non-{{0,1}} labels: {bad[:5]}")
    else:
        classes = raw_values
        mapping = {v: i for i, v in enumerate(classes)}
    encoded = {s: {e: mapping[v] for e, v in d.items()} for s, d in per_split.items()}
    return encoded, [str(c) for c in classes]


def collect_label_rows(task) -> pd.DataFrame:
    """All ``(entity, timestamp, label, split)`` rows of the task (label_nodes mode)."""
    frames = []
    for split in SPLITS:
        df = task.get_table(split, mask_input_cols=False).df
        df = df[df[task.target_col].notna()]
        frames.append(pd.DataFrame({
            "entity": df[task.entity_col].to_numpy(dtype="int64"),
            "timestamp": pd.to_datetime(df[task.time_col]).to_numpy(),
            "label": df[task.target_col].to_numpy(),
            "split": split,
        }))
    return pd.concat(frames, ignore_index=True)


def encode_label_values(values: np.ndarray, task_type: str) -> Tuple[np.ndarray, List[str]]:
    """Map raw label values to ``0..K-1`` codes (binary tasks keep 0/1)."""
    if task_type == "binary_classification":
        codes = np.rint(values.astype(np.float64)).astype(np.int64)
        bad = np.setdiff1d(np.unique(codes), [0, 1])
        if bad.size:
            raise ValueError(f"binary task has non-{{0,1}} labels: {bad[:5].tolist()}")
        return codes, ["0", "1"]
    classes = sorted(set(values.tolist()), key=lambda v: (str(type(v)), v))
    mapping = {v: i for i, v in enumerate(classes)}
    return np.array([mapping[v] for v in values.tolist()], dtype=np.int64), [str(c) for c in classes]


def balanced_keep_mask(codes: np.ndarray, rng: np.random.RandomState, max_ratio: float = 1.0) -> np.ndarray:
    """Keep all rows of the rarest class and at most ``max_ratio`` x that many of
    every other class (uniformly sampled). Used to balance the in-context train labels."""
    keep = np.zeros(codes.shape[0], dtype=bool)
    if codes.size == 0:
        return keep
    classes, counts = np.unique(codes, return_counts=True)
    cap = int(np.ceil(counts.min() * max_ratio))
    for c, n in zip(classes, counts):
        idx = np.nonzero(codes == c)[0]
        if n > cap:
            idx = rng.choice(idx, size=cap, replace=False)
        keep[idx] = True
    return keep


# --------------------------------------------------------------------------- #
# Subsampling
# --------------------------------------------------------------------------- #


def multi_source_bfs(
    edge_index: np.ndarray, num_nodes: int, seeds: np.ndarray, target_nodes: int
) -> np.ndarray:
    """Visit up to ``target_nodes`` nodes by BFS from ``seeds`` (in seed order)."""
    cap = int(min(target_nodes, num_nodes))
    seeds = seeds[:cap]
    adj = csr_matrix(
        (np.ones(edge_index.shape[1], dtype=np.uint8), (edge_index[0], edge_index[1])),
        shape=(num_nodes, num_nodes),
    )
    indptr, indices = adj.indptr, adj.indices
    seen = np.zeros(num_nodes, dtype=np.bool_)
    order = np.empty(cap, dtype=np.int64)
    tail = 0
    for s in seeds:
        if not seen[s]:
            seen[s] = True
            order[tail] = s
            tail += 1
    head = 0
    while head < tail and tail < cap:
        u = int(order[head])
        head += 1
        nbrs = indices[indptr[u]: indptr[u + 1]]
        new = nbrs[~seen[nbrs]]
        if new.size == 0:
            continue
        new = np.unique(new)
        take = min(new.size, cap - tail)
        new = new[:take]
        seen[new] = True
        order[tail: tail + take] = new
        tail += take
    return order[:tail]



# --------------------------------------------------------------------------- #
# Feature layouts
# --------------------------------------------------------------------------- #


def _table_embedding(
    tf: "TableFeatures",
    *,
    width: int,
    max_categories: int,
    seed: int,
) -> np.ndarray:
    """Dense per-table embedding: quantile(num) + quantile-uniform(frac) + one-hot(cat),
    PCA to ``width`` when wider, then column z-score over the table's rows."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import OneHotEncoder

    from data.transforms.graphland_features import _quantile_transform_block

    n = tf.num.shape[0]
    parts: List[np.ndarray] = []
    if tf.num.shape[1]:
        parts.append(_quantile_transform_block(torch.from_numpy(tf.num), "normal", seed=seed).numpy())
    if tf.frac.shape[1]:
        parts.append(_quantile_transform_block(torch.from_numpy(tf.frac), "uniform", seed=seed).numpy())
    if tf.cat.shape[1]:
        enc = OneHotEncoder(drop="if_binary", sparse_output=False, handle_unknown="ignore", dtype=np.float32)
        parts.append(enc.fit_transform(tf.cat.astype(np.float32)))
    if not parts:
        return np.zeros((n, 0), dtype=np.float32)
    m = np.concatenate(parts, axis=1).astype(np.float32)
    m = np.nan_to_num(m, nan=0.0, posinf=0.0, neginf=0.0)
    k = min(width, m.shape[1], max(1, n - 1))
    if m.shape[1] > k:
        m = PCA(n_components=k, random_state=seed).fit_transform(m).astype(np.float32)
    mean = m.mean(axis=0, keepdims=True)
    std = m.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return ((m - mean) / std).astype(np.float32)


def assemble_shared_layout(
    feats: Dict[str, "TableFeatures"],
    table_names: Sequence[str],
    tbl_of_node: np.ndarray,
    *,
    width: int,
    max_categories: int,
    seed: int,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """Every table is embedded into the SAME ``width`` columns (zero padded).

    Why: the ``DataSet`` pipeline drops columns that are constant on train nodes
    and z-scores with train-node statistics. In a relational graph only the entity
    table is labeled, so any column owned by another table is constant on train
    nodes and would be dropped / blown up. With a shared layout column ``j`` varies
    on the entity table as long as that table has >= j+1 embedding dims; the table
    identity is carried by the typed foreign-key edges, not by the features.
    Disable ``--drop_constant_train_features`` / ``--final_inductive_zscore`` for
    these graphs anyway (``slurm_scripts/eval/queue_eval_relbench.sh`` does).
    """
    n = tbl_of_node.shape[0]
    x = np.zeros((n, width), dtype=np.float32)
    widths: Dict[str, int] = {}
    for ti, t in enumerate(table_names):
        rows = np.nonzero(tbl_of_node == ti)[0]
        if rows.size == 0:
            widths[t] = 0
            continue
        emb = _table_embedding(feats[t], width=width, max_categories=max_categories, seed=seed)
        widths[t] = int(emb.shape[1])
        if emb.shape[1]:
            x[rows, : emb.shape[1]] = emb
    return x, widths


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #


def convert_relbench_task(
    dataset: str,
    task: str,
    *,
    out_dir: Optional[Path] = None,
    max_nodes: Optional[int] = None,
    max_categories: int = 32,
    text_drop_ratio: float = 0.9,
    label_policy: str = "latest",
    split_precedence: str = "test_first",
    max_labeled_fraction: float = 0.5,
    feature_layout: str = "shared",
    feature_width: int = 64,
    contract_link_tables: bool = False,
    label_mode: str = "entity",
    balance_train_labels: bool = False,
    balance_max_ratio: float = 1.0,
    seed: int = 0,
    write: bool = True,
    verbose: bool = True,
) -> Data:
    """Build the Wander NC ``Data`` object for one RelBench entity-classification task.

    Args:
        dataset: RelBench dataset name (``rel-f1``) or Hub spec / local path.
        task: task name (``driver-dnf``).
        out_dir: where to write ``data.pt`` + ``meta.json`` (when ``write``).
        max_nodes: BFS-subsample to this many nodes, seeded at labeled entities
            (test, then val, then train). ``None`` keeps the full database.
        max_categories: top-(k-1) + 'other' bucketing for string columns.
        text_drop_ratio: string columns with nunique >= ratio * rows are dropped.
        label_policy: which of an entity's rows inside a split gives its label.
        split_precedence: ``test_first`` (default) keeps every test entity and
            removes those entities from val/train; ``train_first`` is the opposite.
        max_labeled_fraction: when subsampling, at most this fraction of the node
            budget is spent on labeled seeds (rest of the labels are dropped, train first).
        feature_layout: ``shared`` (default) embeds every table into the same
            ``feature_width`` columns (per-table quantile/one-hot -> PCA -> z-score;
            no column-type masks, so the pipeline only re-z-scores). ``block`` keeps
            the raw per-table columns side by side with GraphLand-style type masks
            (``--graphland_different_transform`` path); only sensible with
            ``--no_drop_constant_train_features --no_final_inductive_zscore``.
        feature_width: shared-layout width (<= --pca_target_dim so PCA is skipped).
        contract_link_tables: junction tables (two foreign keys, no other columns)
            become direct edges between their parents instead of nodes. Halves the
            walk length between e.g. two users of rel-event's 30M-row user_friends
            table and removes those rows from the node budget.
        label_mode: ``entity`` puts one label on each entity-table node (see
            ``label_policy`` / ``split_precedence``). ``label_nodes`` keeps EVERY
            RelBench label row as its own node, linked to its entity by a dedicated
            relation and carrying the label timestamp as its only feature; train /
            val / test masks live on those nodes. Nothing is deduplicated, so an
            entity with ten monthly labels contributes ten nodes.
        balance_train_labels: subsample the majority class(es) among TRAIN labels
            so no class has more than ``balance_max_ratio`` x the rarest class. Val
            and test are untouched. Wander uses train labels as in-context examples,
            so a skewed prior (rel-f1 driver-dnf train is 88% positive, test 70%)
            shifts its scores; balancing removes that prior.
        seed: RNG seed for seed-order shuffling inside each split and balancing.
    """
    if label_mode not in LABEL_MODES:
        raise ValueError(f"label_mode must be one of {LABEL_MODES}, got {label_mode!r}")
    if feature_layout not in FEATURE_LAYOUTS:
        raise ValueError(f"feature_layout must be one of {FEATURE_LAYOUTS}, got {feature_layout!r}")
    import relbench  # lazy: only needed when building the cache
    from relbench.base import TaskType

    t0 = time.time()
    log = print if verbose else (lambda *a, **k: None)

    ds = relbench.load_dataset(dataset)
    tk = ds.load_task(task)
    task_type = TaskType(tk.task_type).value
    if task_type not in ("binary_classification", "multiclass_classification"):
        raise ValueError(
            f"{dataset}/{task} is {task_type}; only binary / multiclass entity "
            "classification can be converted to a Wander NC dataset"
        )
    db = tk.get_db()  # rows after test_timestamp hidden; task's leak columns removed
    all_tables = sorted(db.table_dict)
    contracted: List[str] = []
    if contract_link_tables:
        contracted = [
            t for t in all_tables
            if t != tk.entity_table and is_pure_link_table(db.table_dict[t])
        ]
    table_names = [t for t in all_tables if t not in contracted]
    table_sizes = {t: len(db.table_dict[t]) for t in table_names}
    label_rows: Optional[pd.DataFrame] = None
    if label_mode == "label_nodes":
        label_rows = collect_label_rows(tk)
        n_ent = len(db.table_dict[tk.entity_table])
        label_rows = label_rows[(label_rows["entity"] >= 0) & (label_rows["entity"] < n_ent)]
        label_rows = label_rows.reset_index(drop=True)
        n_train_before_balance = int((label_rows["split"] == "train").sum())
        if balance_train_labels:
            codes_all, _ = encode_label_values(label_rows["label"].to_numpy(), task_type)
            is_train = (label_rows["split"] == "train").to_numpy()
            keep_train = balanced_keep_mask(codes_all[is_train], np.random.RandomState(seed), balance_max_ratio)
            keep = ~is_train
            keep[np.nonzero(is_train)[0][keep_train]] = True
            label_rows = label_rows[keep].reset_index(drop=True)
        table_names.append(LABEL_TABLE)
        table_sizes[LABEL_TABLE] = len(label_rows)
    offsets = _table_offsets(table_sizes)
    num_nodes = sum(table_sizes.values())
    log(f"[relbench] {dataset}/{task}: {task_type}, entity table {tk.entity_table}, "
        f"{len(all_tables)} tables, {num_nodes} row-nodes"
        + (f" (contracted link tables: {contracted})" if contracted else "")
        + (f" (incl. {len(label_rows)} label nodes)" if label_rows is not None else "")
        + f" ({time.time()-t0:.1f}s)")

    # ---- edges
    ei_dir, et_dir, relations = build_fkey_edges(
        db, [t for t in table_names if t != LABEL_TABLE], offsets, contracted
    )
    if label_rows is not None:
        rel_id = len(relations)
        relations.append((LABEL_TABLE, "entity", tk.entity_table))
        src = offsets[LABEL_TABLE] + np.arange(len(label_rows), dtype=np.int64)
        dst = offsets[tk.entity_table] + label_rows["entity"].to_numpy(dtype=np.int64)
        ei_dir = np.concatenate([ei_dir, np.stack([src, dst])], axis=1)
        et_dir = np.concatenate([et_dir, np.full(len(label_rows), rel_id, dtype=np.int64)])
    edge_index, edge_type = symmetrize(ei_dir, et_dir)
    log(f"[relbench] {ei_dir.shape[1]} directed edges, {len(relations)} relation types")

    # ---- labels
    y = np.full(num_nodes, np.nan, dtype=np.float32)
    split_masks = {s: np.zeros(num_nodes, dtype=bool) for s in SPLITS}
    if label_rows is None:
        per_split_raw, label_stats = collect_entity_labels(
            tk, label_policy=label_policy, split_precedence=split_precedence
        )
        per_split, class_names = encode_labels(per_split_raw, task_type)
        n_train_before_balance = len(per_split["train"])
        if balance_train_labels and per_split["train"]:
            ents = np.array(list(per_split["train"]), dtype=np.int64)
            codes_tr = np.array([per_split["train"][e] for e in ents], dtype=np.int64)
            keep = balanced_keep_mask(codes_tr, np.random.RandomState(seed), balance_max_ratio)
            per_split["train"] = {int(e): int(c) for e, c in zip(ents[keep], codes_tr[keep])}
        ent_off = offsets[tk.entity_table]
        for s in SPLITS:
            for ent, lab in per_split[s].items():
                node = ent_off + ent
                if node >= ent_off + table_sizes[tk.entity_table]:
                    continue  # dangling entity (already filtered by relbench, be safe)
                y[node] = lab
                split_masks[s][node] = True
    else:
        codes, class_names = encode_label_values(label_rows["label"].to_numpy(), task_type)
        nodes = offsets[LABEL_TABLE] + np.arange(len(label_rows), dtype=np.int64)
        y[nodes] = codes
        for s in SPLITS:
            split_masks[s][nodes[(label_rows["split"] == s).to_numpy()]] = True
        label_stats = {
            s: {
                "label_rows": int((label_rows["split"] == s).sum()),
                "unique_entities": int(label_rows.loc[label_rows["split"] == s, "entity"].nunique()),
            }
            for s in SPLITS
        }
    if balance_train_labels:
        tr_codes = y[split_masks["train"]].astype(np.int64)
        label_stats["train_balancing"] = {
            "before": int(n_train_before_balance),
            "after": int(split_masks["train"].sum()),
            "max_ratio": float(balance_max_ratio),
            "class_counts_after": {str(c): int(n) for c, n in zip(*np.unique(tr_codes, return_counts=True))},
        }
    log("[relbench] labels per split: " + ", ".join(
        f"{s}={int(split_masks[s].sum())}" for s in SPLITS) + f"; classes={class_names}"
        + (f"; train balanced {n_train_before_balance} -> {int(split_masks['train'].sum())}" if balance_train_labels else ""))

    # ---- subsample (before featurizing: features are the memory hog)
    keep: Optional[np.ndarray] = None
    if max_nodes is not None and num_nodes > max_nodes:
        rng = np.random.RandomState(seed)
        seeds: List[np.ndarray] = []
        for s in SPLIT_PRECEDENCE[split_precedence]:
            idx = np.nonzero(split_masks[s])[0]
            rng.shuffle(idx)
            seeds.append(idx)
        seeds_arr = np.concatenate(seeds)
        budget = int(max_nodes * max_labeled_fraction)
        if seeds_arr.size > budget:
            log(f"[relbench] WARNING: {seeds_arr.size} labeled nodes exceed the labeled budget "
                f"{budget}; dropping the rest (lowest precedence split first)")
            dropped = seeds_arr[budget:]
            y[dropped] = np.nan
            for s in SPLITS:
                split_masks[s][dropped] = False
            seeds_arr = seeds_arr[:budget]
        keep = np.sort(multi_source_bfs(edge_index, num_nodes, seeds_arr, max_nodes))
        remap = np.full(num_nodes, -1, dtype=np.int64)
        remap[keep] = np.arange(keep.size)
        m = (remap[edge_index[0]] >= 0) & (remap[edge_index[1]] >= 0)
        edge_index = remap[edge_index[:, m]]
        edge_type = edge_type[m]
        y = y[keep]
        split_masks = {s: v[keep] for s, v in split_masks.items()}
        log(f"[relbench] BFS subsample: kept {keep.size}/{num_nodes} nodes, "
            f"{edge_index.shape[1]} edges; labels " + ", ".join(
                f"{s}={int(split_masks[s].sum())}" for s in SPLITS))
    kept_nodes = keep if keep is not None else np.arange(num_nodes)
    n_kept = kept_nodes.size

    # ---- node table id + per-table kept rows
    node_table = np.empty(n_kept, dtype=np.int64)
    kept_rows: Dict[str, np.ndarray] = {}
    bounds = np.array([offsets[t] for t in table_names] + [num_nodes])
    tbl_of_node = np.searchsorted(bounds, kept_nodes, side="right") - 1
    node_table[:] = tbl_of_node
    for ti, t in enumerate(table_names):
        sel = kept_nodes[tbl_of_node == ti] - offsets[t]
        kept_rows[t] = sel

    # ---- features
    time_origin = None
    try:
        time_origin = pd.Timestamp(db.min_timestamp)
    except Exception:
        pass
    feats: Dict[str, TableFeatures] = {}
    for t in table_names:
        rows = kept_rows[t]
        if t == LABEL_TABLE:
            assert label_rows is not None
            df = label_rows.iloc[rows][["timestamp"]]
            skip: List[str] = []
        else:
            table = db.table_dict[t]
            df = table.df.iloc[rows] if rows.size != len(table.df) else table.df
            skip = [c for c in [table.pkey_col, *table.fkey_col_to_pkey_table] if c is not None]
        feats[t] = featurize_table(
            df, t, skip_cols=skip, time_origin=time_origin,
            max_categories=max_categories, text_drop_ratio=text_drop_ratio,
        )
        tf = feats[t]
        log(f"[relbench]   {t:>24s}: rows={len(df):>8d} num={len(tf.num_cols)} "
            f"frac={len(tf.frac_cols)} cat={len(tf.cat_cols)} dropped={list(tf.dropped)}")

    num_cols = [(t, c) for t in table_names for c in feats[t].num_cols]
    frac_cols = [(t, c) for t in table_names for c in feats[t].frac_cols]
    cat_cols = [("__table__", "__table__")] + [(t, c) for t in table_names for c in feats[t].cat_cols]
    n_num, n_frac, n_cat = len(num_cols), len(frac_cols), len(cat_cols)
    table_widths: Dict[str, int] = {}
    if feature_layout == "shared":
        x, table_widths = assemble_shared_layout(
            feats, table_names, tbl_of_node,
            width=feature_width, max_categories=max_categories, seed=seed,
        )
        F = x.shape[1]
        x_num_mask = x_frac_mask = x_cat_mask = None
        log("[relbench] shared layout: per-table embedding widths " + str(table_widths))
    else:
        F = n_num + n_frac + n_cat
        x = np.zeros((n_kept, F), dtype=np.float32)
        x[:, : n_num + n_frac] = np.nan
        x[:, n_num + n_frac] = node_table + 1  # table indicator, codes 1..T
        col_num = col_frac = 0
        col_cat = 1
        for ti, t in enumerate(table_names):
            tf = feats[t]
            node_sel = np.nonzero(tbl_of_node == ti)[0]
            if tf.num.shape[1]:
                x[node_sel, col_num: col_num + tf.num.shape[1]] = tf.num
            if tf.frac.shape[1]:
                x[node_sel, n_num + col_frac: n_num + col_frac + tf.frac.shape[1]] = tf.frac
            if tf.cat.shape[1]:
                x[node_sel, n_num + n_frac + col_cat: n_num + n_frac + col_cat + tf.cat.shape[1]] = tf.cat
            col_num += tf.num.shape[1]
            col_frac += tf.frac.shape[1]
            col_cat += tf.cat.shape[1]
        x_num_mask = torch.zeros(F, dtype=torch.bool)
        x_num_mask[:n_num] = True
        x_frac_mask = torch.zeros(F, dtype=torch.bool)
        x_frac_mask[n_num: n_num + n_frac] = True
        x_cat_mask = torch.zeros(F, dtype=torch.bool)
        x_cat_mask[n_num + n_frac:] = True

    data = Data(
        x=torch.from_numpy(x),
        y=torch.from_numpy(y),
        edge_index=torch.from_numpy(edge_index.astype(np.int64)),
        edge_type=torch.from_numpy(edge_type.astype(np.int64)),
        train_mask=torch.from_numpy(split_masks["train"]),
        val_mask=torch.from_numpy(split_masks["val"]),
        test_mask=torch.from_numpy(split_masks["test"]),
    )
    if x_num_mask is not None:
        data.x_numerical_mask = x_num_mask
        data.x_fraction_mask = x_frac_mask
        data.x_categorical_mask = x_cat_mask
    if feature_layout == "shared":
        # Already quantile-transformed, PCA'd and z-scored per table; tell
        # apply_feature_preprocessing not to re-normalize across tables (zero
        # padding of narrow tables would otherwise dominate the column statistics).
        data.features_preprocessed = True
    data.num_nodes = int(n_kept)
    data.num_relations = int(len(relations))
    data.node_table = torch.from_numpy(node_table)
    data.orig_node_id = torch.from_numpy(kept_nodes.astype(np.int64))

    meta = {
        "dataset": dataset,
        "task": task,
        "task_type": task_type,
        "entity_table": tk.entity_table,
        "entity_col": tk.entity_col,
        "target_col": tk.target_col,
        "val_timestamp": str(ds.val_timestamp),
        "test_timestamp": str(ds.test_timestamp),
        "tables": table_names,
        "contracted_link_tables": contracted,
        "table_sizes_full": table_sizes,
        "table_offsets_full": offsets,
        "table_sizes_kept": {t: int(kept_rows[t].size) for t in table_names},
        "relations": [{"id": i, "table": t, "fkey": c, "parent": p} for i, (t, c, p) in enumerate(relations)],
        "num_nodes": int(n_kept),
        "num_nodes_full": int(num_nodes),
        "num_edges_directed": int(edge_index.shape[1] // 2),
        "classes": class_names,
        "label_mode": label_mode,
        "labels": {s: int(split_masks[s].sum()) for s in SPLITS},
        "label_stats": label_stats,
        "feature_layout": feature_layout,
        "feature_width": int(F),
        "table_embedding_widths": table_widths,
        "feature_columns": {
            "numerical": [f"{t}.{c}" for t, c in num_cols],
            "fraction": [f"{t}.{c}" for t, c in frac_cols],
            "categorical": [f"{t}.{c}" for t, c in cat_cols],
        },
        "dropped_columns": {t: feats[t].dropped for t in table_names if feats[t].dropped},
        "options": {
            "max_nodes": max_nodes,
            "max_categories": max_categories,
            "text_drop_ratio": text_drop_ratio,
            "label_policy": label_policy,
            "split_precedence": split_precedence,
            "max_labeled_fraction": max_labeled_fraction,
            "feature_layout": feature_layout,
            "feature_width": feature_width,
            "contract_link_tables": contract_link_tables,
            "label_mode": label_mode,
            "balance_train_labels": balance_train_labels,
            "balance_max_ratio": balance_max_ratio,
            "seed": seed,
        },
        "relbench_version": getattr(relbench, "__version__", "unknown"),
    }
    data.relbench_meta = json.dumps(meta)

    if write:
        if out_dir is None:
            raise ValueError("out_dir is required when write=True")
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(data, out_dir / "data.pt")
        with open(out_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2, default=str)
        log(f"[relbench] wrote {out_dir / 'data.pt'} ({time.time()-t0:.1f}s)")
    log(f"[relbench] done: {n_kept} nodes, {edge_index.shape[1]} edges, x[{n_kept}, {F}] "
        f"({feature_layout}; raw cols num={n_num} frac={n_frac} cat={n_cat}), {len(relations)} relations")
    return data
