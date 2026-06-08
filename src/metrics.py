"""
Vectorised ranking metrics over a score matrix.

score_matrix : float32 [n_eval_users × n_items]
    Higher score = more recommended. Seen items are masked to -inf before ranking.

All public functions return scalar floats.
"""

from __future__ import annotations

from typing import Dict, List, Set

import numpy as np


def _mask_seen(scores: np.ndarray, user_seen_idxs: List[Set[int]]) -> np.ndarray:
    """Set scores of already-seen items to -inf (in-place copy)."""
    scores = scores.copy()
    for row, seen in enumerate(user_seen_idxs):
        if seen:
            scores[row, list(seen)] = -np.inf
    return scores


def _top_k_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """Return [n_users × k] array of top-k item indices (unordered within top-k)."""
    if scores.shape[1] <= k:
        return np.argsort(-scores, axis=1)
    # argpartition is O(n) per row vs O(n log n) for full sort
    return np.argpartition(-scores, k, axis=1)[:, :k]


def recall_at_k(
    score_matrix: np.ndarray,
    target_item_idxs: List[int],
    user_seen_idxs: List[Set[int]],
    k: int = 10,
) -> float:
    """Recall@k — fraction of users where the single target is in the top-k."""
    scores  = _mask_seen(score_matrix, user_seen_idxs)
    top_k   = _top_k_indices(scores, k)           # [U × k]
    hits    = sum(
        target_item_idxs[i] in top_k[i]
        for i in range(len(target_item_idxs))
    )
    return hits / len(target_item_idxs) if target_item_idxs else 0.0


def ndcg_at_k(
    score_matrix: np.ndarray,
    target_item_idxs: List[int],
    user_seen_idxs: List[Set[int]],
    k: int = 10,
) -> float:
    """
    NDCG@k for single-positive ranking.
    For a single relevant item, NDCG@k = 1/log2(rank+1) if rank <= k, else 0.
    """
    scores  = _mask_seen(score_matrix, user_seen_idxs)
    # Full sort needed to get the exact rank of the target
    ranked  = np.argsort(-scores, axis=1)          # [U × n_items]
    total   = 0.0
    for i, target in enumerate(target_item_idxs):
        rank_arr = np.where(ranked[i] == target)[0]
        if rank_arr.size == 0:
            continue
        rank = rank_arr[0]  # 0-based
        if rank < k:
            total += 1.0 / np.log2(rank + 2)      # +2 because rank is 0-based
    return total / len(target_item_idxs) if target_item_idxs else 0.0


def compute_metrics(
    score_matrix: np.ndarray,
    target_item_idxs: List[int],
    user_seen_idxs: List[Set[int]],
    k: int = 10,
) -> Dict[str, float]:
    """Compute Recall@k and NDCG@k together (shares the masking step)."""
    scores = _mask_seen(score_matrix, user_seen_idxs)
    top_k  = _top_k_indices(scores, k)

    hits = 0
    ndcg = 0.0
    ranked = np.argsort(-scores, axis=1)

    for i, target in enumerate(target_item_idxs):
        if target in top_k[i]:
            hits += 1
        rank_arr = np.where(ranked[i] == target)[0]
        if rank_arr.size and rank_arr[0] < k:
            ndcg += 1.0 / np.log2(rank_arr[0] + 2)

    n = len(target_item_idxs) or 1
    return {
        f"recall@{k}": hits / n,
        f"ndcg@{k}":   ndcg / n,
    }


# ── Legacy per-user loop (kept for backward compatibility) ──────────────────

class RankingMetrics:
    """Old single-user evaluation loop. Prefer the vectorised functions above."""

    @staticmethod
    def recall_at_k(model, val_targets, user_histories, user_seen_items, k=10):
        hits = total = 0
        for user_id, target_item in val_targets.items():
            history   = user_histories.get(user_id, [])
            seen      = user_seen_items.get(user_id, set())
            recs      = model.recommend(user_id, history, seen, k)
            if target_item in recs[:k]:
                hits += 1
            total += 1
        return hits / total if total > 0 else 0.0
