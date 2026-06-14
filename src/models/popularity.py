from __future__ import annotations

import numpy as np

from src.data import DataBundle
from src.models.base import Recommender


class PopularityRecommender(Recommender):

    def __init__(self, halflife_days: float | None = 365):
        self.halflife_days = halflife_days
        self._item_scores: np.ndarray | None = None

    def fit(self, bundle: DataBundle) -> "PopularityRecommender":
        self._store_id_maps(bundle)

        # Without timestamps, just count interactions per item.
        # Instead, use interaction counts from the matrix
        counts = np.asarray(bundle.train_matrix.sum(axis=0)
                            ).ravel().astype(np.float32)
        self._item_scores = counts
        return self

    def fit_with_decay(self, bundle: DataBundle, df) -> "PopularityRecommender":
        # Use timestamps so newer interactions count more.
        self._store_id_maps(bundle)

        if self.halflife_days is None:
            return self.fit(bundle)

        # Convert half-life to the same time unit as the data.
        halflife_ms = self.halflife_days * 86_400_000.0
        decay_lambda = np.log(2) / halflife_ms

        max_ts = df["timestamp"].max()
        df = df.copy()
        df["weight"] = np.exp(-decay_lambda * (max_ts - df["timestamp"]))

        scores = np.zeros(bundle.n_items, dtype=np.float32)
        for row in df.itertuples(index=False):
            i_idx = bundle.item_to_idx.get(row.item_id)
            if i_idx is not None:
                scores[i_idx] += float(row.weight)

        self._item_scores = scores
        return self

    def score_users(self, user_idxs: np.ndarray) -> np.ndarray:
        # Score every item for each requested user.
        return np.tile(self._item_scores, (len(user_idxs), 1))

    def _fallback_recommend(self, seen_items, k):
        # Use popular unseen items when a model has no user data.
        if self._item_scores is None:
            return []
        scores = self._item_scores.copy()
        seen_idxs = [self.item_to_idx[i]
                     for i in seen_items if i in self.item_to_idx]
        if seen_idxs:
            scores[seen_idxs] = -np.inf
        top_k = np.argpartition(scores, -k)[-k:]
        top_k = top_k[np.argsort(scores[top_k])[::-1]]
        return [self.idx_to_item[i] for i in top_k if i in self.idx_to_item]
