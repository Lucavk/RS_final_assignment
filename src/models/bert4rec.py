from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from src.config import RANDOM_SEED
from src.data import DataBundle
from src.models.base import Recommender


class _SeqDataset(Dataset):
    def __init__(self, sequences, targets, max_len, mask_idx, pad_idx):
        self.sequences = sequences
        self.targets = targets
        self.max_len = max_len
        self.mask_idx = mask_idx
        self.pad_idx = pad_idx

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx][-(self.max_len - 1):] + [self.mask_idx]
        pad = self.max_len - len(seq)
        if pad > 0:
            seq = [self.pad_idx] * pad + seq
        return torch.tensor(seq, dtype=torch.long), torch.tensor(self.targets[idx], dtype=torch.long)


class _Bert4RecNet(nn.Module):
    def __init__(self, num_items, max_len=20, dim=128, heads=4, layers=2, dropout=0.2, pad_idx=0):
        super().__init__()
        self.pad_idx = pad_idx
        self.item_embedding = nn.Embedding(
            num_items + 2, dim, padding_idx=pad_idx)
        self.position_embedding = nn.Embedding(max_len, dim)
        enc = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=dim * 4,
            dropout=dropout, activation="gelu", batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc, num_layers=layers)
        self.output_layer = nn.Linear(dim, num_items + 2)
        nn.init.normal_(self.item_embedding.weight, std=0.02)
        nn.init.normal_(self.position_embedding.weight, std=0.02)
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, seqs):
        bsz, slen = seqs.shape
        pos = torch.arange(slen, device=seqs.device).unsqueeze(
            0).expand(bsz, slen)
        x = self.item_embedding(seqs) + self.position_embedding(pos)
        pad_mask = seqs.eq(self.pad_idx)
        enc = self.encoder(x, src_key_padding_mask=pad_mask)
        # logits at the [MASK] position
        return self.output_layer(enc[:, -1, :])


class Bert4RecRecommender(Recommender):

    def __init__(
        self,
        max_seq_len: int = 20,
        embedding_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.2,
        epochs: int = 20,
        batch_size: int = 512,
        lr: float = 1e-3,
        weight_decay: float = 1e-6,
        score_batch_size: int = 1024,
        random_state: int = RANDOM_SEED,
    ):
        self.max_seq_len = max_seq_len
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.score_batch_size = score_batch_size
        self.random_state = random_state

        self._model = None
        self._device = None
        self._pad_idx = 0
        self._mask_idx = None
        self._sequences = None   # User histories from the data bundle.

    def fit(self, bundle: DataBundle) -> "Bert4RecRecommender":
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        self._store_id_maps(bundle)
        n_items = bundle.n_items
        self._mask_idx = n_items + 1
        self._sequences = bundle.user_sequences

        self._device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        print(f"Bert4Rec: training on {self._device}  "
              f"(dim={self.embedding_dim}, layers={self.num_layers}, epochs={self.epochs})…")

        # Build examples where the model predicts the next item.
        sequences, targets = [], []
        for u_idx, hist in self._sequences.items():
            internal = [g + 1 for g in hist]
            if len(internal) < 2:
                continue
            for t in range(1, len(internal)):
                sequences.append(internal[:t])
                targets.append(internal[t])
        print(f"  {len(sequences):,} training sequences")

        loader = DataLoader(
            _SeqDataset(sequences, targets, self.max_seq_len,
                        self._mask_idx, self._pad_idx),
            batch_size=self.batch_size, shuffle=True, num_workers=0,
        )

        self._model = _Bert4RecNet(
            n_items, self.max_seq_len, self.embedding_dim,
            self.num_heads, self.num_layers, self.dropout, self._pad_idx,
        ).to(self._device)
        opt = torch.optim.AdamW(self._model.parameters(
        ), lr=self.lr, weight_decay=self.weight_decay)
        criterion = nn.CrossEntropyLoss(ignore_index=self._pad_idx)

        self._model.train()
        for epoch in range(1, self.epochs + 1):
            total, n = 0.0, 0
            for seqs, tgts in loader:
                seqs, tgts = seqs.to(self._device), tgts.to(self._device)
                opt.zero_grad()
                loss = criterion(self._model(seqs), tgts)
                loss.backward()
                opt.step()
                total += loss.item() * tgts.size(0)
                n += tgts.size(0)
            if epoch % 5 == 0 or epoch == 1:
                print(
                    f"  epoch {epoch:02d}/{self.epochs}  CE loss={total / n:.6f}")

        self._model.eval()
        print("Bert4Rec: fit complete")
        return self

    def _build_input(self, u_idx):
        # Build one masked sequence for a user.
        hist = self._sequences.get(u_idx, [])
        internal = [
            g + 1 for g in hist][-(self.max_seq_len - 1):] + [self._mask_idx]
        pad = self.max_seq_len - len(internal)
        if pad > 0:
            internal = [self._pad_idx] * pad + internal
        return internal

    def score_users(self, user_idxs: np.ndarray) -> np.ndarray:
        # Score all items from each user history.
        out = np.zeros((len(user_idxs), self.n_items), dtype=np.float32)
        with torch.no_grad():
            for start in range(0, len(user_idxs), self.score_batch_size):
                batch_ids = user_idxs[start:start + self.score_batch_size]
                seqs = torch.tensor([self._build_input(int(u)) for u in batch_ids],
                                    dtype=torch.long, device=self._device)
                logits = self._model(seqs)
                # Drop PAD and MASK, keeping only real item scores.
                real = logits[:, 1:self.n_items + 1]
                out[start:start + len(batch_ids)] = real.cpu().numpy()
        return out
