from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix

from src.data import DataBundle
from src.models.base import Recommender


class ItemKNNRecommender(Recommender):

    def __init__(self, topk: int = 200, shrinkage: float = 100.0):
        self.topk = topk
        self.shrinkage = shrinkage
        self._sim_matrix = None
        self._train_matrix = None

    def fit(self, bundle: DataBundle) -> "ItemKNNRecommender":
        self._store_id_maps(bundle)
        R = bundle.train_matrix.astype(np.float32)
        self._train_matrix = R
        n_items = bundle.n_items

        print(
            f"ItemKNN: computing item co-occurrence  ({n_items} × {n_items})…")
        # Count how often item pairs appear together.
        G = (R.T @ R).toarray().astype(np.float64)

        pop = G.diagonal().copy()

        print(f"ItemKNN: computing shrinkage-cosine similarity…")
        # Build cosine similarity from item counts.
        denom_cos = np.sqrt(np.outer(pop, pop))
        denom_cos[denom_cos == 0] = 1.0  # avoid div-by-zero for zero-pop items

        # Shrink weak item pairs so rare pairs do not dominate.
        support = G / (G + self.shrinkage)

        sim = support * (G / denom_cos)

        # Do not recommend an item because of itself.
        np.fill_diagonal(sim, 0.0)

        # Keep only the strongest neighbours for each item.
        if self.topk < n_items - 1:
            print(f"ItemKNN: pruning to top-{self.topk} neighbours per item…")

            part = np.argpartition(-sim, self.topk, axis=1)
            to_zero = part[:, self.topk:]
            rows = np.arange(n_items)[:, None]
            sim[rows, to_zero] = 0.0

        self._sim_matrix = csr_matrix(sim.astype(np.float32))
        nnz = self._sim_matrix.nnz
        print(f"ItemKNN: fit complete  (topk={self.topk}, shrinkage={self.shrinkage}, "
              f"nnz={nnz:,})")
        return self

    def score_users(self, user_idxs: np.ndarray) -> np.ndarray:
        # Score every item for each requested user.
        R_sub = self._train_matrix[user_idxs]
        scores = R_sub.dot(self._sim_matrix)
        return np.asarray(scores.todense(), dtype=np.float32)
