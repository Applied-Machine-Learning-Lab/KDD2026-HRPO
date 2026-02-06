# -*- coding: utf-8 -*-
"""
sasrec_utils.py

Datasets + evaluation utilities for SASRec offline training on log_session CSV sequences.
These are model-agnostic as long as the model exposes:
  - compute_loss_ce(seq, target)  OR predict_next_logits(seq)
  - predict_next_logits(seq): (B, n_items+1) with logits[:,0] masked

Conventions:
- item ids are 1..n_items; PAD=0
- sequences are LEFT padded; last position is most recent
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def left_pad(seq: List[int], max_len: int, pad_id: int = 0) -> np.ndarray:
    if len(seq) >= max_len:
        arr = np.array(seq[-max_len:], dtype=np.int64)
    else:
        arr = np.full((max_len,), pad_id, dtype=np.int64)
        if len(seq) > 0:
            arr[-len(seq):] = np.array(seq, dtype=np.int64)
    return arr


class SASRecTrainDataset(torch.utils.data.Dataset):
    """
    For each user seq [i1..iT], create pairs:
      input = [i1..i(t-1)]  target = i(t)
    for t=2..T  (1-based)
    """
    def __init__(self, user_seqs: List[List[int]], max_len: int):
        super().__init__()
        self.user_seqs = user_seqs
        self.max_len = int(max_len)
        self.pairs: List[Tuple[int,int]] = []
        for u, s in enumerate(user_seqs):
            if len(s) < 2:
                continue
            for t in range(1, len(s)):
                self.pairs.append((u, t))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        u, t = self.pairs[idx]
        s = self.user_seqs[u]
        prefix = s[:t]
        target = s[t]
        x = left_pad(prefix, self.max_len, pad_id=0)
        return torch.from_numpy(x).long(), torch.tensor(target, dtype=torch.long)


class SASRecValDataset(torch.utils.data.Dataset):
    """
    Each user contributes one validation instance:
      input = [i1..i(T-1)]  target = iT
    """
    def __init__(self, user_seqs: List[List[int]], max_len: int):
        super().__init__()
        self.max_len = int(max_len)
        self.users: List[int] = []
        for u, s in enumerate(user_seqs):
            if len(s) >= 2:
                self.users.append(u)
        self.user_seqs = user_seqs

    def __len__(self) -> int:
        return len(self.users)

    def __getitem__(self, idx: int):
        u = self.users[idx]
        s = self.user_seqs[u]
        x = left_pad(s[:-1], self.max_len, pad_id=0)
        y = int(s[-1])
        return torch.from_numpy(x).long(), torch.tensor(y, dtype=torch.long)


@torch.no_grad()
def _ndcg_at_k(rank: int, k: int) -> float:
    if rank < 1 or rank > k:
        return 0.0
    return 1.0 / np.log2(rank + 1.0)


@torch.no_grad()
def evaluate_full_ranking(model, dataloader, device: torch.device, ks=(1, 5, 10, 20)) -> Dict[str, float]:
    """
    Full ranking over all items (1..n_items). More expensive but deterministic.
    """
    model.eval()
    hits = {k: 0 for k in ks}
    ndcg = {k: 0.0 for k in ks}
    n = 0

    for seq, target in dataloader:
        seq = seq.to(device)
        target = target.to(device)
        logits = model.predict_next_logits(seq)  # (B, n_items+1)
        topk_max = max(ks)
        top_idx = torch.topk(logits, k=topk_max, dim=1).indices  # (B, topk_max)
        for i in range(seq.size(0)):
            n += 1
            t = int(target[i].item())
            row = top_idx[i].tolist()
            rank = row.index(t) + 1 if t in row else 10**9
            for k in ks:
                if rank <= k:
                    hits[k] += 1
                    ndcg[k] += _ndcg_at_k(rank, k)

    out = {}
    for k in ks:
        out[f"hit@{k}"] = hits[k] / max(n, 1)
        out[f"ndcg@{k}"] = ndcg[k] / max(n, 1)
    return out


@torch.no_grad()
def evaluate_sampled_ranking(model, dataloader, device: torch.device, n_items: int, n_neg: int = 100,
                            seed: int = 2025, ks=(1, 5, 10, 20)) -> Dict[str, float]:
    """
    Sampled ranking: for each user, rank target among {1 positive + n_neg negatives}.
    Deterministic with seed.
    """
    rng = np.random.default_rng(int(seed))
    model.eval()

    hits = {k: 0 for k in ks}
    ndcg = {k: 0.0 for k in ks}
    n = 0

    for seq, target in dataloader:
        seq = seq.to(device)
        target = target.to(device)
        B = seq.size(0)

        cand = np.zeros((B, 1 + n_neg), dtype=np.int64)
        cand[:, 0] = target.detach().cpu().numpy()
        for i in range(B):
            t = int(cand[i, 0])
            neg = rng.integers(1, n_items + 1, size=n_neg * 2, endpoint=False)
            neg = neg[neg != t]
            if neg.size < n_neg:
                neg2 = rng.integers(1, n_items + 1, size=n_neg, endpoint=False)
                neg2 = neg2[neg2 != t]
                neg = np.concatenate([neg, neg2], axis=0)
            cand[i, 1:] = neg[:n_neg]

        cand_t = torch.from_numpy(cand).to(device).long()  # (B,C)
        user_emb = model.encode(seq)  # (B,D)
        scores = model.score_candidates(user_emb, cand_t)  # (B,C)
        pos = scores[:, 0:1]
        better = (scores > pos).sum(dim=1)  # (B,)
        rank = better + 1  # 1-based

        for i in range(B):
            n += 1
            r = int(rank[i].item())
            for k in ks:
                if r <= k:
                    hits[k] += 1
                    ndcg[k] += _ndcg_at_k(r, k)

    out = {}
    for k in ks:
        out[f"hit@{k}"] = hits[k] / max(n, 1)
        out[f"ndcg@{k}"] = ndcg[k] / max(n, 1)
    return out
