from __future__ import annotations

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from src.config import RANDOM_SEED
from src.data import DataBundle
from src.models.base import Recommender


class _LightGCNNet(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=128, num_layers=2):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.num_layers = num_layers
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)

    def propagate(self, norm_adj):
        # Pass embeddings through the graph layers.
        all_emb = torch.cat([self.user_embedding.weight,
                            self.item_embedding.weight], dim=0)
        embs = [all_emb]
        cur = all_emb
        for _ in range(self.num_layers):
            cur = torch.sparse.mm(norm_adj, cur)
            embs.append(cur)
        final = torch.stack(embs, dim=0).mean(dim=0)
        return torch.split(final, [self.num_users, self.num_items], dim=0)

    def bpr_scores(self, users, pos, neg, norm_adj):
        ue, ie = self.propagate(norm_adj)
        u = ue[users]
        pos_s = (u * ie[pos]).sum(-1)
        neg_s = (u * ie[neg]).sum(-1)
        return pos_s, neg_s


class LightGCNRecommender(Recommender):

    def __init__(
        self,
        embedding_dim: int = 128,
        num_layers: int = 2,
        epochs: int = 30,
        batch_size: int = 2048,
        lr: float = 1e-3,
        weight_decay: float = 1e-6,
        score_batch_size: int = 4096,
        random_state: int = RANDOM_SEED,
    ):
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.score_batch_size = score_batch_size
        self.random_state = random_state

        self._model = None
        self._norm_adj = None
        self._device = None
        self._user_emb = None
        self._item_emb = None

    # Build the graph matrix from user-item interactions.
    def _build_norm_adj(self, R, n_users, n_items):
        coo = R.tocoo()
        u = coo.row
        item_node = n_users + coo.col  # item nodes offset after users
        rows = np.concatenate([u, item_node])  # symmetric edges
        cols = np.concatenate([item_node, u])

        n_nodes = n_users + n_items
        idx = torch.tensor(np.vstack([rows, cols]), dtype=torch.long)
        val = torch.ones(len(rows), dtype=torch.float32)
        adj = torch.sparse_coo_tensor(idx, val, (n_nodes, n_nodes)).coalesce()

        deg = torch.sparse.sum(adj, dim=1).to_dense()
        d_inv_sqrt = torch.pow(deg + 1e-8, -0.5)
        r, c = adj.indices()
        norm_val = d_inv_sqrt[r] * adj.values() * d_inv_sqrt[c]
        return torch.sparse_coo_tensor(adj.indices(), norm_val, adj.shape).coalesce()

    def fit(self, bundle: DataBundle) -> "LightGCNRecommender":
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        self._store_id_maps(bundle)
        n_users, n_items = bundle.n_users, bundle.n_items
        R = bundle.train_matrix.tocsr()

        self._device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        print(f"LightGCN: training on {self._device}  "
              f"(dim={self.embedding_dim}, layers={self.num_layers}, epochs={self.epochs})…")

        self._norm_adj = self._build_norm_adj(
            R, n_users, n_items).to(self._device)

        # Store known user-item pairs for training.
        coo = R.tocoo()
        pos_users = coo.row.astype(np.int64)
        pos_items = coo.col.astype(np.int64)
        user_pos = [set(R.indices[R.indptr[u]:R.indptr[u + 1]])
                    for u in range(n_users)]

        self._model = _LightGCNNet(
            n_users, n_items, self.embedding_dim, self.num_layers).to(self._device)
        opt = torch.optim.Adam(self._model.parameters(),
                               lr=self.lr, weight_decay=self.weight_decay)

        n = len(pos_users)
        rng = np.random.default_rng(self.random_state)

        self._model.train()
        for epoch in range(1, self.epochs + 1):
            perm = rng.permutation(n)
            total = 0.0
            for start in range(0, n, self.batch_size):
                b = perm[start:start + self.batch_size]
                u = pos_users[b]
                p = pos_items[b]
                # Pick negative items the user has not seen.
                neg = rng.integers(0, n_items, size=len(b))
                for i in range(len(b)):
                    while neg[i] in user_pos[u[i]]:
                        neg[i] = rng.integers(0, n_items)

                u_t = torch.from_numpy(u).to(self._device)
                p_t = torch.from_numpy(p).to(self._device)
                n_t = torch.from_numpy(neg).to(self._device)

                opt.zero_grad()
                pos_s, neg_s = self._model.bpr_scores(
                    u_t, p_t, n_t, self._norm_adj)
                loss = -F.logsigmoid(pos_s - neg_s).mean()
                loss.backward()
                opt.step()
                total += loss.item() * len(b)

            if epoch % 10 == 0 or epoch == 1:
                print(
                    f"  epoch {epoch:02d}/{self.epochs}  BPR loss={total / n:.6f}")

        # Cache embeddings so scoring is faster.
        self._model.eval()
        with torch.no_grad():
            ue, ie = self._model.propagate(self._norm_adj)
            self._user_emb = ue.detach()
            self._item_emb = ie.detach()
        print("LightGCN: fit complete")
        return self

    def score_users(self, user_idxs: np.ndarray) -> np.ndarray:
        # Score every item for each requested user.
        out = np.empty((len(user_idxs), self.n_items), dtype=np.float32)
        with torch.no_grad():
            for start in range(0, len(user_idxs), self.score_batch_size):
                idx = user_idxs[start:start + self.score_batch_size]
                u = self._user_emb[torch.from_numpy(
                    np.asarray(idx)).to(self._device)]
                s = u @ self._item_emb.T
                out[start:start + len(idx)] = s.cpu().numpy()
        return out
