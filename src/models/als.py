"""
Matrix factorisation via Alternating Least Squares (ALS).

Uses the 'implicit' library which provides fast CPU/GPU ALS for implicit data.

Confidence model: C = 1 + alpha * R  (Hu et al., 2008)
Score(u) = user_factors[u] @ item_factors.T
"""

from __future__ import annotations

import os

import numpy as np
from scipy.sparse import csr_matrix

from src.data import DataBundle
from src.models.base import Recommender

# Suppress OpenBLAS threading warning
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


class ALSRecommender(Recommender):
    """
    Parameters
    ----------
    factors : int
        Embedding dimensionality.  Higher = more expressive, slower.
    regularization : float
        L2 regularisation for ALS updates.
    iterations : int
        Number of ALS alternation steps.
    alpha : float
        Confidence scaling: C = 1 + alpha * R.
    random_state : int
        For reproducibility.
    """

    def __init__(
        self,
        factors: int = 128,
        regularization: float = 0.05,
        iterations: int = 50,
        alpha: float = 20.0,
        random_state: int = 42,
    ):
        self.factors        = factors
        self.regularization = regularization
        self.iterations     = iterations
        self.alpha          = alpha
        self.random_state   = random_state

        self._user_factors: np.ndarray | None = None  # [n_users × factors]
        self._item_factors: np.ndarray | None = None  # [n_items × factors]
        self._train_matrix = None  # kept for the legacy recommend() path

    def fit(self, bundle: DataBundle) -> "ALSRecommender":
        try:
            from implicit.als import AlternatingLeastSquares
        except ImportError as e:
            raise ImportError("Install 'implicit':  pip install implicit") from e

        self._store_id_maps(bundle)
        self._train_matrix = bundle.train_matrix

        confidence = (bundle.train_matrix * self.alpha).tocsr()

        print(f"ALS: training  (factors={self.factors}, reg={self.regularization}, "
              f"alpha={self.alpha}, iters={self.iterations})…")

        import torch
        use_gpu = torch.cuda.is_available()

        model = AlternatingLeastSquares(
            factors=self.factors,
            regularization=self.regularization,
            iterations=self.iterations,
            random_state=self.random_state,
            use_gpu=False,
        )
        if use_gpu:
            print(f"  (using GPU: {torch.cuda.get_device_name(0)})")
        model.fit(confidence, show_progress=True)

        self._user_factors = model.user_factors  # float32
        self._item_factors = model.item_factors  # float32
        print("ALS: fit complete")
        return self

    def score_users(self, user_idxs: np.ndarray) -> np.ndarray:
        """Score = U[user_idxs] @ Vᵀ  →  float32 [U × n_items]."""
        return self._user_factors[user_idxs] @ self._item_factors.T
