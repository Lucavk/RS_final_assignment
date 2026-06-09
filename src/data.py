"""
Data loading, ID mapping, and DataBundle construction.

DataBundle is the central object passed to every model's fit() method.
ID maps are always built from the FULL dataset so matrix dimensions are
consistent across all training splits.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


@dataclass
class DataBundle:
    """All derived data needed by models and evaluation code."""

    train_matrix: csr_matrix            # binary [n_users × n_items]
    user_to_idx: Dict[object, int]
    idx_to_user: Dict[int, object]
    item_to_idx: Dict[object, int]
    idx_to_item: Dict[int, object]
    user_sequences: Dict[int, List[int]]  # user_idx -> [item_idx, ...] by time
    user_seen_idxs: Dict[int, Set[int]]   # user_idx -> set(item_idx)
    n_users: int
    n_items: int
    # Submission users (from sample_submission.csv)
    submission_user_ids: List = field(default_factory=list)
    submission_user_idxs: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int32))


# ── Loading ─────────────────────────────────────────────────────────────────

def _load_interactions(path: Path) -> pd.DataFrame:
    """Load one interaction file, rename to standard columns, coerce types."""
    df = pd.read_csv(path)

    # Auto-detect column names (case-insensitive)
    col_map = {c.lower(): c for c in df.columns}

    user_col = next(c for k, c in col_map.items()
                    if k in ("user_id", "userid", "user", "uid", "reviewerid"))
    item_col = next(c for k, c in col_map.items()
                    if k in ("item_id", "itemid", "item", "iid", "asin", "product_id"))
    ts_col   = next(c for k, c in col_map.items()
                    if k in ("timestamp", "time", "unixreviewtime", "date", "datetime"))

    df = df[[user_col, item_col, ts_col]].copy()
    df.columns = ["user_id", "item_id", "timestamp"]

    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["user_id", "item_id", "timestamp"])
    df["timestamp"] = df["timestamp"].astype(np.int64)

    return df


def load_train_only(config) -> pd.DataFrame:
    """Load only train.csv — used for honest local validation (no time-leak)."""
    df = _load_interactions(config.TRAIN_PATH)
    df = (
        df.sort_values(["user_id", "item_id", "timestamp"])
          .drop_duplicates(subset=["user_id", "item_id"], keep="last")
          .sort_values(["user_id", "timestamp"])
          .reset_index(drop=True)
    )
    print(f"Train-only dataset: {len(df):,} interactions | "
          f"{df['user_id'].nunique():,} users | {df['item_id'].nunique():,} items")
    return df


def load_all_data(config) -> pd.DataFrame:
    """
    Load train.csv + test.csv, combine, deduplicate, and sort.

    Used for the FINAL SUBMISSION only — not for local validation.
    test.csv contains post-cutoff observed interactions (input, not labels).
    Deduplication: for the same (user, item) pair, keep the most recent event.
    """
    train_df = _load_interactions(config.TRAIN_PATH)
    test_df  = _load_interactions(config.TEST_PATH)

    df = pd.concat([train_df, test_df], ignore_index=True)

    # Keep latest occurrence per (user, item)
    df = (
        df.sort_values(["user_id", "item_id", "timestamp"])
          .drop_duplicates(subset=["user_id", "item_id"], keep="last")
    )

    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    print(f"Combined dataset: {len(df):,} interactions | "
          f"{df['user_id'].nunique():,} users | {df['item_id'].nunique():,} items")

    return df


def load_submission_user_ids(config) -> list:
    """Read the 2,255 user_ids from sample_submission.csv."""
    sub = pd.read_csv(config.SAMPLE_SUBMISSION_PATH)
    return sub["user_id"].tolist()


# ── ID mapping ───────────────────────────────────────────────────────────────

def build_id_maps(df: pd.DataFrame):
    """
    Build contiguous int mappings for all users and items in df.

    Build maps from the FULL combined dataset once; reuse them for all splits
    so matrix dimensions are always consistent.
    """
    users = sorted(df["user_id"].unique())
    items = sorted(df["item_id"].unique())

    user_to_idx = {u: i for i, u in enumerate(users)}
    idx_to_user = {i: u for u, i in user_to_idx.items()}
    item_to_idx = {it: i for i, it in enumerate(items)}
    idx_to_item = {i: it for it, i in item_to_idx.items()}

    return user_to_idx, idx_to_user, item_to_idx, idx_to_item


# ── Bundle construction ──────────────────────────────────────────────────────

def build_bundle(
    df: pd.DataFrame,
    user_to_idx: dict,
    idx_to_user: dict,
    item_to_idx: dict,
    idx_to_item: dict,
    submission_user_ids: list,
) -> DataBundle:
    """
    Build a DataBundle from a (possibly split) DataFrame.

    Always uses the pre-built ID maps so dimensions stay consistent.
    Users/items in the maps but absent from df will have zero-rows in the matrix.
    """
    n_users = len(user_to_idx)
    n_items = len(item_to_idx)

    # ── Sparse user-item matrix (binary implicit) ──
    u_idxs = df["user_id"].map(user_to_idx).to_numpy(dtype=np.int32)
    i_idxs = df["item_id"].map(item_to_idx).to_numpy(dtype=np.int32)
    vals   = np.ones(len(df), dtype=np.float32)

    train_matrix = csr_matrix((vals, (u_idxs, i_idxs)), shape=(n_users, n_items))

    # ── Per-user sequences and seen-item sets ──
    user_sequences: Dict[int, List[int]] = {}
    user_seen_idxs: Dict[int, Set[int]]  = {}

    for uid, group in df.groupby("user_id", sort=False):
        u_idx = user_to_idx[uid]
        sorted_items = (
            group.sort_values("timestamp")["item_id"]
                 .map(item_to_idx)
                 .tolist()
        )
        user_sequences[u_idx] = sorted_items
        user_seen_idxs[u_idx] = set(sorted_items)

    # ── Submission user index array ──
    sub_idxs = np.array(
        [user_to_idx[u] for u in submission_user_ids if u in user_to_idx],
        dtype=np.int32,
    )

    return DataBundle(
        train_matrix=train_matrix,
        user_to_idx=user_to_idx,
        idx_to_user=idx_to_user,
        item_to_idx=item_to_idx,
        idx_to_item=idx_to_item,
        user_sequences=user_sequences,
        user_seen_idxs=user_seen_idxs,
        n_users=n_users,
        n_items=n_items,
        submission_user_ids=submission_user_ids,
        submission_user_idxs=sub_idxs,
    )


# ── Persistence helpers ──────────────────────────────────────────────────────

def save_bundle(bundle: DataBundle, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(bundle, f, protocol=4)


def load_bundle(path: Path) -> DataBundle:
    with open(path, "rb") as f:
        return pickle.load(f)
