"""
Bayesian Personalised Ranking matrix factorisation (Rendle et al., 2009).

Uses the 'implicit' library.  Unlike ALS (which fits a pointwise confidence
objective), BPR optimises a *pairwise* ranking loss — it learns to score
observed items above unobserved ones.  This different inductive bias makes
BPR a useful, decorrelated member of the ensemble.

Score(u) = user_factors[u] @ item_factors.T   (bias term folded in by implicit)
"""

from __future__ import annotations

import os

import numpy as np

from src.data import DataBundle
from src.models.base import Recommender

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


class BPRRecommender(Recommender):
    """
    Parameters
    ----------
    factors : int
        Latent dimensionality.
    learning_rate : float
        SGD step size.
    regularization : float
        L2 regularisation on factors.
    iterations : int
        Number of training epochs.
    """

    def __init__(
        self,
        factors: int = 128,
        learning_rate: float = 0.01,
        regularization: float = 0.01,
        iterations: int = 100,
        random_state: int = 42,
    ):
        self.factors        = factors
        self.learning_rate  = learning_rate
        self.regularization = regularization
        self.iterations     = iterations
        self.random_state   = random_state

        self._user_factors: np.ndarray | None = None
        self._item_factors: np.ndarray | None = None

    def fit(self, bundle: DataBundle) -> "BPRRecommender":
        try:
            from implicit.bpr import BayesianPersonalizedRanking
        except ImportError as e:
            raise ImportError("Install 'implicit':  pip install implicit") from e

        self._store_id_maps(bundle)

        import torch
        use_gpu = torch.cuda.is_available()

        print(f"BPR: training  (factors={self.factors}, lr={self.learning_rate}, "
              f"reg={self.regularization}, iters={self.iterations})…")

        model = BayesianPersonalizedRanking(
            factors=self.factors,
            learning_rate=self.learning_rate,
            regularization=self.regularization,
            iterations=self.iterations,
            random_state=self.random_state,
            use_gpu=use_gpu,
        )
        model.fit(bundle.train_matrix, show_progress=True)

        self._user_factors = model.user_factors  # [n_users × (factors+1)]
        self._item_factors = model.item_factors  # [n_items × (factors+1)]
        print("BPR: fit complete")
        return self

    def score_users(self, user_idxs: np.ndarray) -> np.ndarray:
        """Score = U[user_idxs] @ Vᵀ  →  float32 [U × n_items]."""
        scores = self._user_factors[user_idxs] @ self._item_factors.T
        return np.asarray(scores, dtype=np.float32)
