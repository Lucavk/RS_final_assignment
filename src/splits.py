from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd


def _temporal_loo(
    df: pd.DataFrame,
    user_ids_to_split: List,
) -> Tuple[pd.DataFrame, Dict]:
    # Hold out each selected user's last interaction
    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    # Find each selected user's last interaction
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
            continue
        val_targets[user_id] = df.at[idx, "item_id"]
        drop_indices.append(idx)

    train_df = df.drop(index=drop_indices).reset_index(drop=True)
    return train_df, val_targets


def fold_a(df_full: pd.DataFrame, submission_user_ids: list) -> Tuple[pd.DataFrame, Dict]:
    # Fold A is restricted to submission users
    return _temporal_loo(df_full, submission_user_ids)


def fold_b(df_full: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    # Fold B uses every user in the dataset
    all_user_ids = df_full["user_id"].unique().tolist()
    return _temporal_loo(df_full, all_user_ids)


def val_targets_to_arrays(val_targets: Dict, user_to_idx: dict, item_to_idx: dict):
    # Convert original ids to aligned matrix indices
    import numpy as np

    pairs = [
        (user_to_idx[u], item_to_idx[it])
        for u, it in val_targets.items()
        if u in user_to_idx and it in item_to_idx
    ]
    if not pairs:
        return np.array([], dtype=np.int32), []

    eval_user_idxs = np.array([p[0] for p in pairs], dtype=np.int32)
    target_item_idxs = [p[1] for p in pairs]
    return eval_user_idxs, target_item_idxs
