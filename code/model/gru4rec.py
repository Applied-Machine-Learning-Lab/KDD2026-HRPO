# -*- coding: utf-8 -*-
"""
code/model/gru4rec.py

GRU4Rec (official-style) baseline for next-item recommendation.

Key design choices (aligned with the GRU4Rec papers):
- Session-parallel mini-batch training is implemented in the TRAIN script (not here).
- Supports original pairwise losses: TOP1, BPR, and their "-max" variants (TOP1-max, BPR-max),
  plus Cross-Entropy over full softmax as a convenient baseline.
- Item ids are 1..n_items; PAD=0 (embedding uses padding_idx=0).

This module is compatible with your existing sasrec_utils.py evaluation helpers:
- predict_next_logits(seq) -> (B, n_items+1) with logits[:,0] masked
- encode(seq) -> (B, D)
- score_candidates(user_emb, cand_iids) -> (B, C)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Union, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence


@dataclass
class GRU4RecConfig:
    n_items: int                 # valid ids: 1..n_items
    max_len: int = 50
    d_model: int = 128
    n_layers: int = 1
    dropout: float = 0.2
    tie_weights: bool = True
    final_act: str = "linear"    # "linear" | "elu" | "tanh" | "relu"


def _get_final_act(name: str):
    name = (name or "linear").lower()
    if name == "linear":
        return lambda x: x
    if name == "elu":
        return F.elu
    if name == "tanh":
        return torch.tanh
    if name == "relu":
        return F.relu
    raise ValueError(f"Unknown final_act={name}. Use linear|elu|tanh|relu")


class GRU4Rec(nn.Module):
    def __init__(self, cfg: GRU4RecConfig):
        super().__init__()
        self.cfg = cfg
        self.n_items = int(cfg.n_items)
        self.max_len = int(cfg.max_len)
        self.d_model = int(cfg.d_model)
        self.n_layers = int(cfg.n_layers)

        self.item_emb = nn.Embedding(self.n_items + 1, self.d_model, padding_idx=0)
        self.inp_dropout = nn.Dropout(cfg.dropout)

        self.gru = nn.GRU(
            input_size=self.d_model,
            hidden_size=self.d_model,
            num_layers=self.n_layers,
            dropout=(cfg.dropout if self.n_layers > 1 else 0.0),
            batch_first=True,
        )
        self.out_dropout = nn.Dropout(cfg.dropout)

        self.tie_weights = bool(cfg.tie_weights)
        if not self.tie_weights:
            self.out = nn.Linear(self.d_model, self.n_items + 1, bias=False)

        self.final_act_name = (cfg.final_act or "linear").lower()
        self.final_act = _get_final_act(self.final_act_name)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.uniform_(self.item_emb.weight, a=-0.1, b=0.1)
        if not self.tie_weights:
            nn.init.uniform_(self.out.weight, a=-0.1, b=0.1)

        for name, p in self.gru.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(p)
            elif "bias" in name:
                nn.init.zeros_(p)

    @staticmethod
    def _left_to_right_padded(seq: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Convert LEFT-padded sequences to RIGHT-padded sequences.

        Input:
          seq: (B,L) left padded, last position is most recent.
          lengths: (B,) number of non-zero tokens.
        Output:
          right: (B,L) right padded, first position is earliest.
        """
        B, L = seq.shape
        right = torch.zeros_like(seq)
        for i in range(B):
            l = int(lengths[i].item())
            if l <= 0:
                continue
            right[i, :l] = seq[i, L - l: L]
        return right

    def encode(self, seq: torch.Tensor) -> torch.Tensor:
        """
        seq: (B,L) ids in [0..n_items], LEFT-padded
        return: (B,D) last hidden state for each sequence
        """
        device = seq.device
        lengths = (seq != 0).sum(dim=1).to(torch.long)  # (B,)
        lengths_clamped = lengths.clamp(min=1)

        right = self._left_to_right_padded(seq, lengths_clamped)
        x = self.item_emb(right)  # (B,L,D)
        x = self.inp_dropout(x)

        packed = pack_padded_sequence(
            x, lengths_clamped.detach().cpu(),
            batch_first=True, enforce_sorted=False
        )
        out_packed, h_n = self.gru(packed)  # h_n: (n_layers,B,D)
        h = h_n[-1]  # (B,D)
        h = self.out_dropout(h)

        if (lengths == 0).any():
            h = h.masked_fill((lengths == 0).unsqueeze(1), 0.0)
        return h

    def _all_item_logits(self, user_emb: torch.Tensor) -> torch.Tensor:
        """
        user_emb: (B,D)
        returns: (B, n_items+1) raw scores (final_act applied), PAD=0 later masked.
        """
        if self.tie_weights:
            logits = user_emb @ self.item_emb.weight.t()
        else:
            logits = self.out(user_emb)
        logits = self.final_act(logits)
        logits[:, 0] = -1e9
        return logits

    def predict_next_logits(self, seq: torch.Tensor) -> torch.Tensor:
        """
        seq: (B,L) LEFT-padded
        returns: (B, n_items+1), logits[:,0] masked
        """
        u = self.encode(seq)
        return self._all_item_logits(u)

    def score_candidates(self, user_emb: torch.Tensor, cand_iids: Union[torch.Tensor, list]) -> torch.Tensor:
        """
        user_emb: (B,D)
        cand_iids: (C,) or (B,C) with ids in [0..n_items]
        returns: (B,C) scores (final_act applied), masked where cand==0
        """
        if not torch.is_tensor(cand_iids):
            cand_iids = torch.tensor(cand_iids, dtype=torch.long, device=user_emb.device)
        cand = cand_iids.to(user_emb.device).long()
        if cand.dim() == 1:
            cand = cand.unsqueeze(0)  # (1,C)

        emb = self.item_emb(cand.clamp(min=0, max=self.n_items))  # (B?,C,D)
        if emb.size(0) == 1 and user_emb.size(0) != 1:
            emb = emb.expand(user_emb.size(0), -1, -1)
        scores = (emb * user_emb.unsqueeze(1)).sum(dim=-1)  # (B,C)
        scores = self.final_act(scores)
        if (cand == 0).any():
            scores = scores.masked_fill(cand == 0, -1e9)
        return scores

    def compute_loss_ce(self, seq: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Cross-entropy over full item softmax (pointwise).
        seq: (B,L)
        target: (B,) ids in [1..n_items]
        """
        logits = self.predict_next_logits(seq)
        return F.cross_entropy(logits, target, ignore_index=0)

    def forward_step(self, inp_iids: torch.Tensor, h_prev: torch.Tensor,
                     reset_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        One GRU step.

        inp_iids: (B,) current item id
        h_prev: (n_layers,B,D) previous hidden state
        reset_mask: (B,) bool; if True, reset the corresponding hidden to 0 BEFORE this step.

        Returns:
          user_emb: (B,D) output of last layer at this step (after dropout)
          h_next: (n_layers,B,D)
        """
        if reset_mask is not None and reset_mask.any():
            h_prev = h_prev.clone()
            h_prev[:, reset_mask] = 0.0

        x = self.item_emb(inp_iids.clamp(min=0, max=self.n_items))  # (B,D)
        x = self.inp_dropout(x).unsqueeze(1)  # (B,1,D)
        out, h_next = self.gru(x, h_prev)
        user_emb = self.out_dropout(out[:, -1, :])  # (B,D)
        return user_emb, h_next
