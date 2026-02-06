# -*- coding: utf-8 -*-
"""
train_gru4rec_baseline.py

GRU4Rec baseline ():
- same GLOBAL video_id -> item_id mapping trick as your SASRec baseline
- supports *session-parallel mini-batch* training (the canonical GRU4Rec training loop)
- supports GRU4Rec losses: TOP1 / BPR / TOP1-max / BPR-max, plus cross-entropy (XE)

It is compatible with your current evaluation utilities in sasrec_utils.py.
"""
from __future__ import annotations

import os
import json
import time
import argparse
from dataclasses import asdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model.gru4rec import GRU4Rec, GRU4RecConfig
from sasrec_utils import (
    SASRecTrainDataset, SASRecValDataset,
    evaluate_full_ranking, evaluate_sampled_ranking,
    set_seed
)


def build_global_mapping_and_click_sequences(csv_path: str) -> Tuple[List[List[int]], Dict[int, int], Dict[int, int], int]:
    """
    Same as train_sasrec_baseline_v3:
    1) Build GLOBAL mapping on ALL rows: video_id -> item_id in 1..n_items_total
    2) Build per-user CLICK sequences (sorted by time)
    """
    usecols = ["user_id", "video_id", "time_ms", "is_click"]
    dtypes = {"user_id": "int32", "video_id": "int64", "time_ms": "int64", "is_click": "int8"}
    df = pd.read_csv(csv_path, usecols=usecols, dtype=dtypes)

    all_vid = df["video_id"].to_numpy()
    uniq_vid = np.unique(all_vid)  # sorted
    vid2iid = {int(v): int(i + 1) for i, v in enumerate(uniq_vid.tolist())}
    n_items_total = int(len(uniq_vid))

    df = df[df["is_click"] == 1].copy()
    df.sort_values(["user_id", "time_ms"], inplace=True, kind="mergesort")

    vids_click = df["video_id"].to_numpy()
    iid_click = (np.searchsorted(uniq_vid, vids_click).astype(np.int32) + 1)

    uids = df["user_id"].to_numpy()
    unique_uids, start_idx, counts = np.unique(uids, return_index=True, return_counts=True)
    user_seqs: List[List[int]] = []
    for s, c in zip(start_idx.tolist(), counts.tolist()):
        user_seqs.append(iid_click[s:s + c].astype(int).tolist())

    uid2uix = {int(u): int(i) for i, u in enumerate(unique_uids.tolist())}
    return user_seqs, vid2iid, uid2uix, n_items_total


def save_json(obj: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


@torch.no_grad()
def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def _sample_negatives(rng: np.random.Generator, B: int, n_items: int, target: torch.Tensor, n_neg: int) -> torch.Tensor:
    """
    Sample n_neg negatives per row, avoid the positive target when possible.
    Returns neg_ids: (B, n_neg) torch.LongTensor on CPU.
    """
    tgt = target.detach().cpu().numpy().astype(np.int64)
    neg = rng.integers(1, n_items + 1, size=(B, n_neg * 2), endpoint=False)
    out = np.empty((B, n_neg), dtype=np.int64)
    for i in range(B):
        cand = neg[i]
        cand = cand[cand != tgt[i]]
        if cand.size < n_neg:
            extra = rng.integers(1, n_items + 1, size=(n_neg,), endpoint=False)
            extra = extra[extra != tgt[i]]
            cand = np.concatenate([cand, extra], axis=0)
        out[i] = cand[:n_neg]
    return torch.from_numpy(out).long()


def loss_gru4rec(user_emb: torch.Tensor,
                 target: torch.Tensor,
                 model: GRU4Rec,
                 loss_type: str,
                 n_items: int,
                 neg_mode: str,
                 n_neg: int,
                 bprmax_reg: float,
                 rng: Optional[np.random.Generator] = None) -> torch.Tensor:
    """
    Compute GRU4Rec-style loss given user_emb at current step and target items.

    neg_mode:
      - "inbatch": negatives are other targets within minibatch (B-1 each)
      - "sampled": negatives are random sampled (n_neg each)
    """
    loss_type = loss_type.lower()
    neg_mode = neg_mode.lower()

    eps = 1e-24
    B = user_emb.size(0)
    device = user_emb.device

    if loss_type == "ce":
        logits = model._all_item_logits(user_emb)  # (B, n_items+1)
        return torch.nn.functional.cross_entropy(logits, target, ignore_index=0)

    if neg_mode == "sampled":
        assert rng is not None, "rng required for sampled negatives"
        assert n_neg > 0
        neg_ids = _sample_negatives(rng, B, n_items, target, n_neg).to(device, non_blocking=True)  # (B,n_neg)
        pos = model.score_candidates(user_emb, target.view(-1, 1)).squeeze(1)  # (B,)
        neg = model.score_candidates(user_emb, neg_ids)  # (B,n_neg)

        if loss_type == "top1":
            term = torch.sigmoid(neg - pos.unsqueeze(1)) + torch.sigmoid(neg ** 2)
            return term.mean()

        if loss_type == "top1-max":
            w = torch.softmax(neg, dim=1)
            term = torch.sigmoid(neg - pos.unsqueeze(1)) + torch.sigmoid(neg ** 2)
            return (w * term).sum(dim=1).mean()

        if loss_type == "bpr":
            return (-torch.log(torch.sigmoid(pos.unsqueeze(1) - neg) + eps)).mean()

        if loss_type == "bpr-max":
            w = torch.softmax(neg, dim=1)
            prob = (w * torch.sigmoid(pos.unsqueeze(1) - neg)).sum(dim=1) + eps
            loss = (-torch.log(prob))
            if bprmax_reg > 0:
                loss = loss + float(bprmax_reg) * (w * (neg ** 2)).sum(dim=1)
            return loss.mean()

        raise ValueError(f"Unknown loss_type={loss_type}")

    tgt_emb = model.item_emb(target.clamp(min=0, max=n_items))  # (B,D)
    scores = (user_emb @ tgt_emb.t())  # (B,B)
    scores = model.final_act(scores)

    diag = torch.eye(B, dtype=torch.bool, device=device)
    pos = scores.diag().view(B, 1)          # (B,1)
    neg_scores = scores.masked_fill(diag, -1e9)

    if loss_type == "top1":
        term = torch.sigmoid(neg_scores - pos) + torch.sigmoid(neg_scores ** 2)
        return term.mean()

    if loss_type == "top1-max":
        w = torch.softmax(neg_scores, dim=1)
        term = torch.sigmoid(neg_scores - pos) + torch.sigmoid(neg_scores ** 2)
        return (w * term).sum(dim=1).mean()

    if loss_type == "bpr":
        return (-torch.log(torch.sigmoid(pos - neg_scores) + eps)).mean()

    if loss_type == "bpr-max":
        w = torch.softmax(neg_scores, dim=1)  # weights over negatives
        prob = (w * torch.sigmoid(pos - neg_scores)).sum(dim=1) + eps
        loss = -torch.log(prob)
        if bprmax_reg > 0:
            loss = loss + float(bprmax_reg) * (w * (neg_scores ** 2)).sum(dim=1)
        return loss.mean()

    raise ValueError(f"Unknown loss_type={loss_type}")


class SessionParallelIterator:
    """
    Canonical GRU4Rec training stream:
      For each active session, emit (inp=item[t], tgt=item[t+1]) at each step.
      When a session ends, replace it with the next session and mark reset_mask for hidden reset.

    Notes:
    - No shuffling (matches the official PyTorch implementation behavior).
    - Each epoch iterates over ALL adjacent pairs exactly once.
    """
    def __init__(self, sessions: List[List[int]], batch_size: int):
        self.sessions = [s for s in sessions if len(s) >= 2]
        self.batch_size = int(batch_size)
        self.n_sessions = len(self.sessions)
        if self.n_sessions == 0:
            raise ValueError("No sessions with length>=2")

        self._next_session = 0
        B = min(self.batch_size, self.n_sessions)
        self.B = B

        self.sidx = np.full((B,), -1, dtype=np.int64)
        self.pos = np.zeros((B,), dtype=np.int64)
        self.finished = np.zeros((B,), dtype=np.bool_)

        for b in range(B):
            self.sidx[b] = self._next_session
            self._next_session += 1

    def __iter__(self):
        return self

    def __next__(self):
        reset = np.zeros((self.B,), dtype=np.bool_)
        for b in range(self.B):
            if self.sidx[b] == -1:
                continue
            if self.finished[b]:
                if self._next_session < self.n_sessions:
                    self.sidx[b] = self._next_session
                    self._next_session += 1
                    self.pos[b] = 0
                    self.finished[b] = False
                    reset[b] = True
                else:
                    self.sidx[b] = -1  # deactivate

        active = self.sidx != -1
        if not active.any():
            raise StopIteration

        inp = np.zeros((self.B,), dtype=np.int64)
        tgt = np.zeros((self.B,), dtype=np.int64)

        for b in range(self.B):
            if self.sidx[b] == -1:
                continue
            s = self.sessions[int(self.sidx[b])]
            p = int(self.pos[b])
            inp[b] = s[p]
            tgt[b] = s[p + 1]
            self.pos[b] = p + 1
            if (p + 1) >= (len(s) - 1):
                self.finished[b] = True

        return (
            torch.from_numpy(inp).long(),
            torch.from_numpy(tgt).long(),
            torch.from_numpy(reset).bool(),
            torch.from_numpy(active).bool()
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--max_len", type=int, default=50)
    parser.add_argument("--cache_dir", type=str, default="")
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--tie_weights", action="store_true", default=True)
    parser.add_argument("--no_tie_weights", action="store_false", dest="tie_weights")
    parser.add_argument("--final_act", type=str, default="linear", choices=["linear", "elu", "tanh", "relu"])
    parser.add_argument("--train_mode", type=str, default="session", choices=["session", "pairs"],
                        help="session=canonical session-parallel training; pairs=prefix->next pairs with DataLoader")
    parser.add_argument("--loss", type=str, default="bpr-max",
                        choices=["bpr-max", "top1-max", "bpr", "top1", "ce"])
    parser.add_argument("--neg_mode", type=str, default="inbatch", choices=["inbatch", "sampled"])
    parser.add_argument("--n_neg", type=int, default=0, help="Only used when neg_mode=sampled")
    parser.add_argument("--bprmax_reg", type=float, default=0.0, help="Score regularization weight for BPR-max")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--optim", type=str, default="adagrad", choices=["adagrad", "adamw"],
                        help="Optimizer. Official GRU4Rec often uses Adagrad.")
    parser.add_argument("--lr", type=float, default=0.05,
                        help="Learning rate (typical GRU4Rec starts higher with Adagrad; tune for your data).")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=0.0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--eval_neg", type=int, default=0)
    parser.add_argument("--eval_seed", type=int, default=2025)

    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--save_metric_k", type=int, default=10)
    parser.add_argument("--early_stop_patience", type=int, default=0)
    parser.add_argument("--early_stop_min_delta", type=float, default=0.0)

    args = parser.parse_args()
    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    cache_npz = ""
    cache_meta = ""
    if args.cache_dir:
        os.makedirs(args.cache_dir, exist_ok=True)
        cache_npz = os.path.join(args.cache_dir, "gru4rec_cache.npz")
        cache_meta = os.path.join(args.cache_dir, "gru4rec_meta.json")

    if cache_npz and os.path.exists(cache_npz) and os.path.exists(cache_meta):
        print(f"[Data] Loading cached sequences: {cache_npz}")
        pack = np.load(cache_npz, allow_pickle=True)
        user_seqs = pack["user_seqs"].tolist()
        with open(cache_meta, "r", encoding="utf-8") as f:
            meta = json.load(f)
        n_items_total = int(meta["n_items_total"])
    else:
        print(f"[Data] Building GLOBAL mapping + click sequences from CSV: {args.csv_path}")
        user_seqs, _, _, n_items_total = build_global_mapping_and_click_sequences(args.csv_path)
        clicks_total = sum(len(s) for s in user_seqs)
        print(f"[Data] n_users={len(user_seqs)} n_items_total={n_items_total} clicks_total={clicks_total}")

        if cache_npz:
            np.savez_compressed(cache_npz, user_seqs=np.array(user_seqs, dtype=object))
            save_json({"n_items_total": n_items_total}, cache_meta)
            print(f"[Data] Saved cache to {cache_npz} and {cache_meta}")

    n_users = len(user_seqs)
    clicks_total = sum(len(s) for s in user_seqs)
    print(f"[Data] Summary: n_users={n_users} n_items_total={n_items_total} clicks_total={clicks_total}")

    mx = max((max(s) for s in user_seqs if len(s) > 0), default=0)
    mn = min((min(s) for s in user_seqs if len(s) > 0), default=0)
    print(f"[Data] Sanity: seq_id_min={mn} seq_id_max={mx}")
    if mx > n_items_total or mn < 1:
        raise RuntimeError(f"[Data] ID out of range: min={mn} max={mx} n_items_total={n_items_total}")

    val_ds = SASRecValDataset(user_seqs=user_seqs, max_len=args.max_len)
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
    
    train_user_seqs = [s[:-1] for s in user_seqs if len(s) >= 3]  # need >=2 after truncation
    train_loader = None
    if args.train_mode == "pairs":
        train_ds = SASRecTrainDataset(user_seqs=train_user_seqs, max_len=args.max_len)
        print(f"[Data] Train samples={len(train_ds)} (last-click holdout)  Val users={len(val_ds)}")
        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True, drop_last=True
        )
    else:
        sessions = [s for s in train_user_seqs if len(s) >= 2]
        n_pairs = sum(len(s) - 1 for s in sessions)
        print(f"[Data] Session-parallel(train): sessions={len(sessions)} pairs(total steps)={n_pairs}  Val users={len(val_ds)}")
    

    cfg = GRU4RecConfig(
        n_items=n_items_total,
        max_len=args.max_len,
        d_model=args.d_model,
        n_layers=args.n_layers,
        dropout=args.dropout,
        tie_weights=args.tie_weights,
        final_act=args.final_act,
    )
    device = torch.device(args.device if torch.cuda.is_available() or "cuda" not in args.device else "cpu")
    model = GRU4Rec(cfg).to(device)

    n_params = count_parameters(model)
    print(f"[Model] {cfg}")
    print(f"[Model] Total parameters: {n_params:,}")

    if args.optim == "adagrad":
        optim = torch.optim.Adagrad(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_metric = -1.0
    best_path = os.path.join(args.save_dir, "best.pt")
    last_path = os.path.join(args.save_dir, "last.pt")

    save_json({
        "cfg": asdict(cfg),
        "args": vars(args),
        "n_params": n_params,
        "n_users": n_users,
        "n_items_total": n_items_total,
    }, os.path.join(args.save_dir, "run_meta.json"))

    rng = np.random.default_rng(int(args.seed))

    for ep in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()

        loss_sum = 0.0
        n_step = 0

        if args.train_mode == "pairs":
            assert train_loader is not None
            for seq, target in train_loader:
                s_max = int(seq.max().item())
                s_min = int(seq.min().item())
                t_max = int(target.max().item())
                t_min = int(target.min().item())
                if s_min < 0 or s_max > n_items_total or t_min < 1 or t_max > n_items_total:
                    print(f"[OOB] seq_min={s_min} seq_max={s_max}  tgt_min={t_min} tgt_max={t_max}  n_items_total={n_items_total}")
                    raise RuntimeError("Found out-of-bound ids. Fix mapping / delete cache and rebuild.")

                seq = seq.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                user_emb = model.encode(seq)

                loss = loss_gru4rec(
                    user_emb=user_emb,
                    target=target,
                    model=model,
                    loss_type=args.loss,
                    n_items=n_items_total,
                    neg_mode=args.neg_mode,
                    n_neg=int(args.n_neg),
                    bprmax_reg=float(args.bprmax_reg),
                    rng=rng
                )

                optim.zero_grad(set_to_none=True)
                loss.backward()
                if args.grad_clip and args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
                optim.step()

                loss_sum += float(loss.item())
                n_step += 1

        else:
            sessions = [s for s in train_user_seqs if len(s) >= 2]
            it = SessionParallelIterator(sessions, batch_size=args.batch_size)
            B = it.B
            h = torch.zeros((cfg.n_layers, B, cfg.d_model), device=device, dtype=torch.float32)

            for inp, tgt, reset_mask, active_mask in it:
                inp = inp.to(device, non_blocking=True)
                tgt = tgt.to(device, non_blocking=True)
                reset_mask = reset_mask.to(device, non_blocking=True)
                active_mask = active_mask.to(device, non_blocking=True)

                user_emb, h = model.forward_step(inp, h, reset_mask=reset_mask)

                if not active_mask.all():
                    user_emb = user_emb[active_mask]
                    tgt = tgt[active_mask]

                loss = loss_gru4rec(
                    user_emb=user_emb,
                    target=tgt,
                    model=model,
                    loss_type=args.loss,
                    n_items=n_items_total,
                    neg_mode=args.neg_mode,
                    n_neg=int(args.n_neg),
                    bprmax_reg=float(args.bprmax_reg),
                    rng=rng
                )

                optim.zero_grad(set_to_none=True)
                loss.backward()
                if args.grad_clip and args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
                optim.step()

                h = h.detach()

                loss_sum += float(loss.item())
                n_step += 1

        train_loss = loss_sum / max(n_step, 1)
        dt = time.time() - t0

        if args.eval_neg and args.eval_neg > 0:
            val_metrics = evaluate_sampled_ranking(
                model, val_loader, device=device, n_items=n_items_total,
                n_neg=int(args.eval_neg), seed=int(args.eval_seed)
            )
        else:
            val_metrics = evaluate_full_ranking(model, val_loader, device=device)

        flat = " ".join([f"hit@{k}={val_metrics[f'hit@{k}']:.4f} ndcg@{k}={val_metrics[f'ndcg@{k}']:.4f}" for k in (1,5,10,20)])
        print(f"[Epoch {ep}] Train loss: {train_loss:.4f}  dt={dt:.1f}s")
        print(f"[Epoch {ep}] Val: {flat}")

        torch.save({"model": model.state_dict(), "cfg": asdict(cfg), "epoch": ep}, last_path)

        k = int(args.save_metric_k)
        key = f"hit@{k}"
        cur = float(val_metrics.get(key, -1.0))

        improved = False
        if cur == cur:  # not NaN
            if cur > best_metric + args.early_stop_min_delta:
                improved = True

        if improved:
            best_metric = cur
            torch.save({"model": model.state_dict(), "cfg": asdict(cfg), "epoch": ep}, best_path)
            print(f"[Save] best.pt updated by {key}={best_metric:.6f}")

        if args.early_stop_patience and args.early_stop_patience > 0:
            if not hasattr(main, "_bad_epochs"):
                main._bad_epochs = 0  # type: ignore[attr-defined]
            if improved:
                main._bad_epochs = 0  # type: ignore[attr-defined]
            else:
                main._bad_epochs += 1  # type: ignore[attr-defined]

            if main._bad_epochs >= args.early_stop_patience:  # type: ignore[attr-defined]
                print(f"[EarlyStop] {key} did not improve for {args.early_stop_patience} consecutive epochs. Stop at epoch {ep}.")
                break

    print("[Done]")


if __name__ == "__main__":
    main()
