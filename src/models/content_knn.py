"""
Content-based item-item KNN.

Similarity comes from item METADATA (TF-IDF cosine), not from co-occurrence.
This gives the ensemble a genuinely orthogonal signal and helps users with
short histories or rare items where collaborative signal is weak.

Score(u) = R[u] @ Sim     (same scoring mechanism as collaborative ItemKNN)
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize

from src.config import Config
from src.data import DataBundle
from src.features.content import build_content_tfidf
from src.models.base import Recommender


class ContentKNNRecommender(Recommender):
    """
    Parameters
    ----------
    topk : int
        Neighbours kept per item.
    max_features, min_df : TF-IDF vocabulary controls (see features/content.py).
    """

    def __init__(self, topk: int = 200, max_features: int = 30000, min_df: int = 2):
        self.topk         = topk
        self.max_features = max_features
        self.min_df       = min_df
        self._sim_matrix   = None
        self._train_matrix = None

    def fit(self, bundle: DataBundle) -> "ContentKNNRecommender":
        self._store_id_maps(bundle)
        self._train_matrix = bundle.train_matrix.astype(np.float32)
        n_items = bundle.n_items

        # ── Item content vectors ──
        tfidf = build_content_tfidf(
            Config, bundle.item_to_idx,
            max_features=self.max_features, min_df=self.min_df,
        )
        tfidf = normalize(tfidf, norm="l2", axis=1)  # cosine = dot of normalised rows

        print(f"Content: computing cosine similarity ({n_items} × {n_items})…")
        sim = (tfidf @ tfidf.T).toarray().astype(np.float32)  # dense
        np.fill_diagonal(sim, 0.0)

        # ── Keep top-k neighbours per item ──
        if self.topk < n_items - 1:
            print(f"Content: pruning to top-{self.topk} neighbours per item…")
            part   = np.argpartition(-sim, self.topk, axis=1)
            to_zero = part[:, self.topk:]
            rows   = np.arange(n_items)[:, None]
            sim[rows, to_zero] = 0.0

        self._sim_matrix = csr_matrix(sim)
        print(f"Content: fit complete (topk={self.topk}, nnz={self._sim_matrix.nnz:,})")
        return self

    def score_users(self, user_idxs: np.ndarray) -> np.ndarray:
        """Score = R[user_idxs] @ Sim  →  float32 [U × n_items]."""
        R_sub  = self._train_matrix[user_idxs]
        scores = R_sub.dot(self._sim_matrix)
        return np.asarray(scores.todense(), dtype=np.float32)
