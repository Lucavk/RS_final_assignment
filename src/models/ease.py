"""
EASE^R — Embarrassingly Shallow Autoencoders for Sparse Data.
Steck, WWW 2019.

Closed-form solution:
    G  = Rᵀ R                        (item co-occurrence)
    P  = (G + λI)⁻¹
    B  = P / (−diag(P))              (normalise columns)
    B[diag] = 0                       (no self-loops)
    Score(u) = R[u] @ B

B is a dense [n_items × n_items] float32 matrix (~676 MB for 13k items).
This fits comfortably in both the M1 Max (32 GB) and the university node.
"""

from __future__ import annotations

import numpy as np

from src.data import DataBundle
from src.models.base import Recommender


class EASERecommender(Recommender):
    """
    Parameters
    ----------
    lam : float
        L2 regularisation.  Higher = more regularised (smoother).
        Typical range: 50–2000.  Tune with Optuna.
    """

    def __init__(self, lam: float = 500.0):
        self.lam = lam
        self._B: np.ndarray | None = None          # [n_items × n_items]
        self._train_matrix = None                   # CSR (needed for score_users)

    def fit(self, bundle: DataBundle) -> "EASERecommender":
        self._store_id_maps(bundle)
        self._train_matrix = bundle.train_matrix

        R = bundle.train_matrix.astype(np.float64)

        print(f"EASE: computing G = RᵀR  ({bundle.n_items} × {bundle.n_items})…")
        G = (R.T @ R).toarray()                     # [n_items × n_items] dense

        print("EASE: solving (G + λI)⁻¹…")
        G[np.diag_indices_from(G)] += self.lam      # in-place: G + λI
        P = np.linalg.inv(G)

        # Normalise each column: B_ij = -P_ij / P_jj
        diag_P = np.diag(P)
        B = -P / diag_P[np.newaxis, :]              # broadcast division

        # Remove self-loops
        np.fill_diagonal(B, 0.0)

        self._B = B.astype(np.float32)
        print(f"EASE: fit complete  (λ={self.lam})")
        return self

    def score_users(self, user_idxs: np.ndarray) -> np.ndarray:
        """Score = R[user_idxs] @ B  →  float32 [U × n_items]."""
        R_sub = self._train_matrix[user_idxs]       # CSR [U × n_items]
        scores = R_sub.dot(self._B)                 # dense [U × n_items]
        return scores.astype(np.float32)
