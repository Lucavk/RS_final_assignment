from __future__ import annotations

import os

import numpy as np
from scipy.sparse import csr_matrix

from src.data import DataBundle
from src.models.base import Recommender

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


class ALSRecommender(Recommender):

    def __init__(
        self,
        factors: int = 128,
        regularization: float = 0.05,
        iterations: int = 50,
        alpha: float = 20.0,
        random_state: int = 42,
    ):
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations
        self.alpha = alpha
        self.random_state = random_state

        self._user_factors: np.ndarray | None = None
        self._item_factors: np.ndarray | None = None
        self._train_matrix = None  # Used by the shared recommend methods

    def fit(self, bundle: DataBundle) -> "ALSRecommender":
        try:
            from implicit.als import AlternatingLeastSquares
        except ImportError as e:
            raise ImportError(
                "Install 'implicit':  pip install implicit") from e

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
        # Score every item for each requested user.
        return self._user_factors[user_idxs] @ self._item_factors.T
