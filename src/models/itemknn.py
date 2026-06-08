"""
Item-item cosine KNN with support-based shrinkage.

Similarity formula (support-weighted cosine):
    sim(i, j) = cooc(i,j) / (cooc(i,j) + shrinkage) * cosine(i, j)

where:
    cooc(i,j) = |users who interacted with both i and j|
    cosine(i,j) = cooc(i,j) / sqrt(pop_i * pop_j)

Combining:
    sim(i, j) = cooc(i,j)² / ((cooc(i,j) + shrinkage) * sqrt(pop_i * pop_j))

This penalises item pairs with few co-occurrences (reduces noisy high-similarity
scores between rare items) while retaining standard cosine for popular items.

Score(u) = R[u] @ Sim   (sparse matrix multiply)
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix

from src.data import DataBundle
from src.models.base import Recommender


class ItemKNNRecommender(Recommender):
    """
    Parameters
    ----------
    topk : int
        Number of nearest neighbours to keep per item.  Higher = more coverage,
        slower scoring, more memory.  Typical range: 50–500.
    shrinkage : float
        Denominator shrinkage.  0 = no shrinkage (raw cosine).
        Higher values penalise rare co-occurrences more.  Typical: 0–500.
    """

    def __init__(self, topk: int = 200, shrinkage: float = 100.0):
        self.topk      = topk
        self.shrinkage = shrinkage
        self._sim_matrix  = None   # CSR [n_items × n_items]
        self._train_matrix = None  # CSR [n_users × n_items]

    def fit(self, bundle: DataBundle) -> "ItemKNNRecommender":
        self._store_id_maps(bundle)
        R = bundle.train_matrix.astype(np.float32)
        self._train_matrix = R
        n_items = bundle.n_items

        print(f"ItemKNN: computing item co-occurrence  ({n_items} × {n_items})…")
        # G[i,j] = cooc(i,j); diagonal = item popularity
        G = (R.T @ R).toarray().astype(np.float64)  # dense [n_items × n_items]

        pop = G.diagonal().copy()                    # [n_items]

        print(f"ItemKNN: computing shrinkage-cosine similarity…")
        # Cosine denominator: sqrt(pop_i * pop_j)
        denom_cos = np.sqrt(np.outer(pop, pop))
        denom_cos[denom_cos == 0] = 1.0             # avoid div-by-zero for zero-pop items

        # Support factor: cooc / (cooc + shrinkage)
        support = G / (G + self.shrinkage)

        sim = support * (G / denom_cos)             # element-wise

        # No self-similarity
        np.fill_diagonal(sim, 0.0)

        # Keep only top-k neighbours per item (zero out the rest)
        if self.topk < n_items - 1:
            print(f"ItemKNN: pruning to top-{self.topk} neighbours per item…")
            # Indices of elements we want to ZERO (below top-k)
            # argpartition gives the indices of the topk largest values at [:topk]
            part = np.argpartition(-sim, self.topk, axis=1)  # [n_items × n_items]
            to_zero = part[:, self.topk:]                     # indices to zero
            rows = np.arange(n_items)[:, None]
            sim[rows, to_zero] = 0.0

        self._sim_matrix = csr_matrix(sim.astype(np.float32))
        nnz = self._sim_matrix.nnz
        print(f"ItemKNN: fit complete  (topk={self.topk}, shrinkage={self.shrinkage}, "
              f"nnz={nnz:,})")
        return self

    def score_users(self, user_idxs: np.ndarray) -> np.ndarray:
        """Score = R[user_idxs] @ Sim  →  float32 [U × n_items]."""
        R_sub  = self._train_matrix[user_idxs]      # CSR [U × n_items]
        scores = R_sub.dot(self._sim_matrix)         # CSR [U × n_items]
        return np.asarray(scores.todense(), dtype=np.float32)
