# -*- coding: utf-8 -*-
"""
code/model/sasrec.py

SASRec baseline aligned to YOUR current CSV mapping + KuaiSim s:

- We use item ids in [1..n_items], PAD=0  (NO extra "+1 shift" inside the model)
- Sequences are expected to be LEFT-padded (so the last position is the most recent item).
  In env-eval, you can re-pack the history into this format before feeding SASRec.

Why this file:
- Your previous sasrec.py had "shift_env_ids_to_model(+1)".
  But your training pipeline already maps items to 1..n_items. Double-shifting made id==n_items -> n_items+1,
  causing CUDA "index out of bounds" in embedding/gather.

Also: use BOOL causal mask to avoid the "mismatched mask type" warning.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SASRecConfig:
    n_items: int                 # valid ids: 1..n_items
    max_len: int = 50
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.2


class SASRec(nn.Module):
    def __init__(self, cfg: SASRecConfig):
        super().__init__()
        self.cfg = cfg
        self.n_items = int(cfg.n_items)
        self.max_len = int(cfg.max_len)
        self.d_model = int(cfg.d_model)

        self.item_emb = nn.Embedding(self.n_items + 1, self.d_model, padding_idx=0)
        self.pos_emb = nn.Embedding(self.max_len, self.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=self.d_model * 4,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.n_layers)
        self.ln_final = nn.LayerNorm(self.d_model)

        self.register_buffer("_causal_mask", torch.empty(0), persistent=False)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.item_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_emb.weight, mean=0.0, std=0.02)

    def _get_causal_mask(self, L: int, device: torch.device) -> torch.Tensor:
        """
        BOOL causal mask for MultiheadAttention inside TransformerEncoderLayer.
        Shape: (L, L), True means masked.
        """
        if self._causal_mask.numel() == 0 or self._causal_mask.size(0) != L or self._causal_mask.device != device:
            self._causal_mask = torch.triu(torch.ones(L, L, dtype=torch.bool, device=device), diagonal=1)
        return self._causal_mask

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """
        seq: (B,L) with ids in [0..n_items], PAD=0. LEFT-padded.
        returns: (B,L,D)
        """
        B, L = seq.shape
        device = seq.device
        pos = torch.arange(L, device=device).unsqueeze(0).expand(B, L)

        x = self.item_emb(seq) + self.pos_emb(pos)
        x = self.dropout(x)

        key_padding_mask = (seq == 0)        # (B,L) bool
        attn_mask = self._get_causal_mask(L, device)  # (L,L) bool

        x = self.encoder(x, mask=attn_mask, src_key_padding_mask=key_padding_mask)
        x = self.ln_final(x)
        return x

    def encode(self, seq: torch.Tensor) -> torch.Tensor:
        """
        returns: user embedding (B,D) at last position (L-1).
        """
        h = self.forward(seq)
        return h[:, -1, :]

    def predict_next_logits(self, seq: torch.Tensor) -> torch.Tensor:
        """
        returns logits: (B, n_items+1), logits[:,0] masked.
        """
        u = self.encode(seq)
        logits = u @ self.item_emb.weight.t()  # (B, n_items+1)
        logits[:, 0] = -1e9
        return logits

    def score_candidates(self, user_emb: torch.Tensor, cand_iids: Union[torch.Tensor, list]) -> torch.Tensor:
        """
        user_emb: (B,D)
        cand_iids: (C,) or (B,C) with ids in [1..n_items] (0 may appear but will be masked)
        returns: (B,C)
        """
        if not torch.is_tensor(cand_iids):
            cand_iids = torch.tensor(cand_iids, dtype=torch.long, device=user_emb.device)
        cand = cand_iids.to(user_emb.device).long()
        if cand.dim() == 1:
            cand = cand.unsqueeze(0)  # (1,C)

        emb = self.item_emb(cand.clamp(min=0, max=self.n_items))  # (B?,C,D)
        if emb.size(0) == 1 and user_emb.size(0) != 1:
            emb = emb.expand(user_emb.size(0), -1, -1)
        scores = (emb * user_emb.unsqueeze(1)).sum(dim=-1)
        if (cand == 0).any():
            scores = scores.masked_fill(cand == 0, -1e9)
        return scores

    def compute_loss_ce(self, seq: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        seq: (B,L) ids in [0..n_items]
        target: (B,) ids in [1..n_items]
        """
        logits = self.predict_next_logits(seq)
        return F.cross_entropy(logits, target, ignore_index=0)
