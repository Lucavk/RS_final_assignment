from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.sparse import csr_matrix

from src.data import DataBundle
from src.models.base import Recommender


class RecencyTransitionRecommender(Recommender):

    def __init__(
        self,
        window: int = 3,
        decay: float = 1.0,
        recency_decay: float = 0.5,
        shrinkage: float = 10.0,
        pop_discount: float = 0.0,
        max_recent: int = 20,
    ):
        self.window = window
        self.decay = decay
        self.recency_decay = recency_decay
        self.shrinkage = shrinkage
        self.pop_discount = pop_discount
        self.max_recent = max_recent

        self._T = None
        self._user_sequences = None
        self._pop_factor = None

    def fit(self, bundle: DataBundle) -> "RecencyTransitionRecommender":
        self._store_id_maps(bundle)
        self._user_sequences = bundle.user_sequences
        n_items = bundle.n_items

        print(f"Recency: counting directional transitions "
              f"(window={self.window}, decay={self.decay})…")

        trans = defaultdict(float)
        row_sum = defaultdict(float)

        for seq in self._user_sequences.values():
            L = len(seq)
            for i in range(L):
                a = seq[i]
                for j in range(i + 1, min(i + 1 + self.window, L)):
                    b = seq[j]
                    if a == b:
                        continue
                    w = 1.0 / ((j - i) ** self.decay)
                    trans[(a, b)] += w
                    row_sum[a] += w

        # Build the item-to-next-item transition matrix.
        rows = np.empty(len(trans), dtype=np.int32)
        cols = np.empty(len(trans), dtype=np.int32)
        vals = np.empty(len(trans), dtype=np.float32)
        for k, ((a, b), w) in enumerate(trans.items()):
            rows[k] = a
            cols[k] = b
            vals[k] = w / (row_sum[a] + self.shrinkage)

        self._T = csr_matrix((vals, (rows, cols)), shape=(n_items, n_items))

        # Optionally reduce scores for very popular items.
        pop = np.asarray(bundle.train_matrix.sum(
            axis=0)).ravel().astype(np.float32)
        if self.pop_discount > 0:
            self._pop_factor = 1.0 / np.power(pop + 1.0, self.pop_discount)
        else:
            self._pop_factor = None

        print(f"Recency: fit complete (transitions={self._T.nnz:,})")
        return self

    def _recent_unique(self, seq):
        # Get recent items without repeats.
        seen, recent = set(), []
        for it in reversed(seq):
            if it in seen:
                continue
            seen.add(it)
            recent.append(it)
            if len(recent) >= self.max_recent:
                break
        return recent

    def score_users(self, user_idxs: np.ndarray) -> np.ndarray:
        # Score every item for each requested user.
        rows, cols, vals = [], [], []
        for row_i, u in enumerate(user_idxs):
            recent = self._recent_unique(self._user_sequences.get(int(u), []))
            for r, a in enumerate(recent):
                rows.append(row_i)
                cols.append(a)
                vals.append(1.0 / ((1 + r) ** self.recency_decay))

        U_recent = csr_matrix(
            (vals, (rows, cols)),
            shape=(len(user_idxs), self.n_items),
            dtype=np.float32,
        )
        scores = np.asarray(U_recent.dot(self._T).todense(), dtype=np.float32)

        if self._pop_factor is not None:
            scores *= self._pop_factor[np.newaxis, :]

        return scores
