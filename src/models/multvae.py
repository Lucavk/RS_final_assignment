from __future__ import annotations

import numpy as np

from src.config import RANDOM_SEED
from src.data import DataBundle
from src.models.base import Recommender


def _get_torch():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    return torch, nn, F


class MultVAERecommender(Recommender):

    def __init__(
        self,
        hidden: int = 600,
        latent: int = 200,
        dropout: float = 0.5,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        batch_size: int = 500,
        epochs: int = 150,
        beta: float = 0.2,
        anneal_epochs: int = 50,
        random_state: int = RANDOM_SEED,
    ):
        self.hidden = hidden
        self.latent = latent
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.beta = beta
        self.anneal_epochs = anneal_epochs
        self.random_state = random_state

        self._model = None
        self._device = None
        self._train_matrix = None

    # Build the neural network.
    def _build_net(self, n_items):
        torch, nn, F = _get_torch()

        class _Net(nn.Module):
            def __init__(self, n_items, hidden, latent, dropout):
                super().__init__()
                self.latent = latent
                self.drop = nn.Dropout(dropout)
                self.enc1 = nn.Linear(n_items, hidden)
                self.enc2 = nn.Linear(hidden, latent * 2)   # μ and logσ²
                self.dec1 = nn.Linear(latent, hidden)
                self.dec2 = nn.Linear(hidden, n_items)
                for layer in (self.enc1, self.enc2, self.dec1, self.dec2):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)

            def encode(self, x):
                x = F.normalize(x, p=2, dim=1)   # L2-normalise input
                x = self.drop(x)
                h = torch.tanh(self.enc1(x))
                h = self.enc2(h)
                return h[:, :self.latent], h[:, self.latent:]

            def forward(self, x):
                mu, logvar = self.encode(x)
                if self.training:
                    std = torch.exp(0.5 * logvar)
                    z = mu + std * torch.randn_like(std)
                else:
                    z = mu  # deterministic at inference
                h = torch.tanh(self.dec1(z))
                return self.dec2(h), mu, logvar

        return _Net(n_items, self.hidden, self.latent, self.dropout)

    # Train the model.
    def fit(self, bundle: DataBundle) -> "MultVAERecommender":
        torch, nn, F = _get_torch()
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        self._store_id_maps(bundle)
        self._train_matrix = bundle.train_matrix.tocsr()
        n_users, n_items = self._train_matrix.shape

        self._device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        print(f"Mult-VAE: training on {self._device}  "
              f"(hidden={self.hidden}, latent={self.latent}, epochs={self.epochs})…")

        self._model = self._build_net(n_items).to(self._device)
        opt = torch.optim.Adam(self._model.parameters(),
                               lr=self.lr, weight_decay=self.weight_decay)

        # Only train on users with history.
        active_users = np.where(np.asarray(
            self._train_matrix.sum(axis=1)).ravel() > 0)[0]

        step = 0
        total_anneal_steps = max(1, self.anneal_epochs *
                                 (len(active_users) // self.batch_size + 1))

        self._model.train()
        for epoch in range(self.epochs):
            perm = np.random.permutation(active_users)
            epoch_loss = 0.0
            for start in range(0, len(perm), self.batch_size):
                batch_idx = perm[start:start + self.batch_size]
                x = torch.from_numpy(
                    self._train_matrix[batch_idx].toarray().astype(np.float32)
                ).to(self._device)

                logits, mu, logvar = self._model(x)
                log_softmax = F.log_softmax(logits, dim=1)
                # Main reconstruction loss.
                nll = -(log_softmax * x).sum(dim=1).mean()
                kld = -0.5 * (1 + logvar - mu.pow(2) -
                              logvar.exp()).sum(dim=1).mean()

                beta = min(self.beta, self.beta * step / total_anneal_steps)
                loss = nll + beta * kld

                opt.zero_grad()
                loss.backward()
                opt.step()
                epoch_loss += loss.item()
                step += 1

            if (epoch + 1) % 25 == 0 or epoch == 0:
                print(f"  epoch {epoch+1:3d}/{self.epochs}  loss={epoch_loss:.2f}  "
                      f"beta={beta:.3f}")

        self._model.eval()
        print("Mult-VAE: fit complete")
        return self

    # Score users.
    def score_users(self, user_idxs: np.ndarray) -> np.ndarray:
        # Score every item for each requested user.
        torch, _, _ = _get_torch()
        out = np.empty((len(user_idxs), self.n_items), dtype=np.float32)

        self._model.eval()
        with torch.no_grad():
            for start in range(0, len(user_idxs), self.batch_size):
                idx = user_idxs[start:start + self.batch_size]
                x = torch.from_numpy(
                    self._train_matrix[idx].toarray().astype(np.float32)
                ).to(self._device)
                logits, _, _ = self._model(x)
                out[start:start + len(idx)] = logits.cpu().numpy()

        return out
