from __future__ import annotations

import numpy as np

from src.data import DataBundle
from src.models.base import Recommender


class EASERecommender(Recommender):

    def __init__(self, lam: float = 500.0):
        self.lam = lam
        self._B: np.ndarray | None = None  # [n_items × n_items]
        self._train_matrix = None  # CSR (needed for score_users)

    def fit(self, bundle: DataBundle) -> "EASERecommender":
        self._store_id_maps(bundle)
        self._train_matrix = bundle.train_matrix

        R = bundle.train_matrix.astype(np.float64)

        print(
            f"EASE: computing G = RᵀR  ({bundle.n_items} × {bundle.n_items})…")
        G = (R.T @ R).toarray()  # [n_items × n_items] dense

        print("EASE: solving (G + λI)⁻¹…")
        G[np.diag_indices_from(G)] += self.lam  # in-place: G + λI
        P = np.linalg.inv(G)

        # Scale each column using the diagonal.
        diag_P = np.diag(P)
        B = -P / diag_P[np.newaxis, :]  # broadcast division

        # Do not recommend an item because of itself.
        np.fill_diagonal(B, 0.0)

        self._B = B.astype(np.float32)
        print(f"EASE: fit complete  (λ={self.lam})")
        return self

    def score_users(self, user_idxs: np.ndarray) -> np.ndarray:
        # Score every item for each requested user.
        R_sub = self._train_matrix[user_idxs]  # CSR [U × n_items]
        scores = R_sub.dot(self._B)  # dense [U × n_items]
        return scores.astype(np.float32)
