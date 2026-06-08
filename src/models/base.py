"""
Abstract base class for all recommender models.

Every model must implement:
  fit(bundle)           — train on a DataBundle
  score_users(idxs)     — return float32 [len(idxs) × n_items] score matrix

The score matrix must be higher for more preferred items.
Seen-item masking is handled by the caller (metrics.py / ensemble).

recommend() provides a convenient single-user API for backward compatibility
and for submit.py's fallback logic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Set

import numpy as np

from src.data import DataBundle


class Recommender(ABC):

    @abstractmethod
    def fit(self, bundle: DataBundle) -> "Recommender":
        ...

    @abstractmethod
    def score_users(self, user_idxs: np.ndarray) -> np.ndarray:
        """Return float32 [len(user_idxs) × n_items]. Higher = more preferred."""
        ...

    def recommend(
        self,
        user_id,
        user_history: Optional[List] = None,
        seen_items: Optional[Set] = None,
        k: int = 10,
    ) -> List:
        """Single-user top-k recommendation using original string IDs."""
        if seen_items is None:
            seen_items = set()

        user_to_idx = getattr(self, "user_to_idx", {})
        item_to_idx = getattr(self, "item_to_idx", {})
        idx_to_item = getattr(self, "idx_to_item", {})
        n_items     = getattr(self, "n_items", 0)

        if user_id not in user_to_idx or n_items == 0:
            return self._fallback_recommend(seen_items, k)

        user_idx = user_to_idx[user_id]
        scores   = self.score_users(np.array([user_idx], dtype=np.int32))[0]  # [n_items]

        # Mask seen items
        seen_idxs = [item_to_idx[i] for i in seen_items if i in item_to_idx]
        if seen_idxs:
            scores[seen_idxs] = -np.inf

        top_k_idxs = np.argpartition(scores, -k)[-k:]
        top_k_idxs = top_k_idxs[np.argsort(scores[top_k_idxs])[::-1]]

        return [idx_to_item[idx] for idx in top_k_idxs if idx in idx_to_item]

    def _fallback_recommend(self, seen_items: Set, k: int) -> List:
        """Fallback used when user is unknown. Override in subclasses."""
        return []

    # Convenience: expose ID maps set during fit() so recommend() works
    def _store_id_maps(self, bundle: DataBundle) -> None:
        self.user_to_idx = bundle.user_to_idx
        self.idx_to_user = bundle.idx_to_user
        self.item_to_idx = bundle.item_to_idx
        self.idx_to_item = bundle.idx_to_item
        self.n_users     = bundle.n_users
        self.n_items     = bundle.n_items
