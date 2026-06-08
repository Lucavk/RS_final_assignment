"""
Validation splits for local evaluation.

Fold A — "submission LOO" (faithful): hold out the last interaction of each
          submission user (>=2 interactions). Mirrors the Kaggle setup exactly.

Fold B — "global LOO" (low-variance): hold out the last interaction of every
          user with >=2 interactions. ~10x more targets → lower variance for
          hyperparameter search.

Both folds return:
  - train_df  : interactions with targets removed (use this to build DataBundle)
  - val_targets: {user_id: target_item_id}  (original string IDs)
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd


def _temporal_loo(
    df: pd.DataFrame,
    user_ids_to_split: List,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Remove the chronologically last interaction for each user in user_ids_to_split.
    Returns (train_df, {user_id: held_out_item_id}).
    Users with fewer than 2 interactions are skipped (no target can be held out).
    """
    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    # Indices of the last interaction per user (within the subset to split)
    subset = df[df["user_id"].isin(set(user_ids_to_split))]
    last_idx = (
        subset.groupby("user_id", sort=False)
              .apply(lambda g: g.index[-1])
    )

    val_targets: Dict = {}
    drop_indices = []

    for user_id, idx in last_idx.items():
        user_df = subset[subset["user_id"] == user_id]
        if len(user_df) < 2:
            continue  # can't hold out — only 1 interaction
        val_targets[user_id] = df.at[idx, "item_id"]
        drop_indices.append(idx)

    train_df = df.drop(index=drop_indices).reset_index(drop=True)
    return train_df, val_targets


def fold_a(df_full: pd.DataFrame, submission_user_ids: list) -> Tuple[pd.DataFrame, Dict]:
    """Fold A: LOO restricted to the 2,255 submission users."""
    return _temporal_loo(df_full, submission_user_ids)


def fold_b(df_full: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """Fold B: LOO for every user in the dataset."""
    all_user_ids = df_full["user_id"].unique().tolist()
    return _temporal_loo(df_full, all_user_ids)


def val_targets_to_arrays(val_targets: Dict, user_to_idx: dict, item_to_idx: dict):
    """
    Convert {user_id: item_id} targets to aligned numpy arrays.

    Returns
    -------
    eval_user_idxs : np.ndarray [n_eval]  — user indices for the score matrix rows
    target_item_idxs : list[int] [n_eval] — target item index per user
    Both arrays share the same ordering.
    Unknown IDs (items/users not in maps) are silently dropped.
    """
    import numpy as np

    pairs = [
        (user_to_idx[u], item_to_idx[it])
        for u, it in val_targets.items()
        if u in user_to_idx and it in item_to_idx
    ]
    if not pairs:
        return np.array([], dtype=np.int32), []

    eval_user_idxs   = np.array([p[0] for p in pairs], dtype=np.int32)
    target_item_idxs = [p[1] for p in pairs]
    return eval_user_idxs, target_item_idxs
