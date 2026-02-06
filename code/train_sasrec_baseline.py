# -*- coding: utf-8 -*-
"""
train_sasrec_baseline_v3.py

Fix for CUDA "index out of bounds" in embedding/gather:

- Build a GLOBAL video_id -> item_id mapping from ALL rows in the CSV (not only clicked rows),
  so n_items matches the full catalog used by KuaiSim candidate pool (typically 5659 for Pure).
- Then build per-user CLICK sequences using that global mapping.

IDs:
- item_id in [1..n_items_total]; PAD=0
"""
from __future__ import annotations

import os
import json
import time
import argparse
from dataclasses import asdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model.sasrec import SASRec, SASRecConfig
from sasrec_utils import (
    SASRecTrainDataset, SASRecValDataset,
    evaluate_full_ranking, evaluate_sampled_ranking,
    set_seed
)


def build_global_mapping_and_click_sequences(csv_path: str) -> Tuple[List[List[int]], Dict[int, int], Dict[int, int], int]:
    """
    1) Read minimal columns.
    2) Build GLOBAL mapping on ALL rows:
         video_id -> item_id in 1..n_items_total
       (this prevents OOB when raw video_id is large / not contiguous)
    3) Filter is_click==1 and build per-user sequences (sorted by time).

    Returns:
      user_seqs (click-only, mapped ids)
      vid2iid   (global mapping)
      uid2uix
      n_items_total
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--max_len", type=int, default=50)
    parser.add_argument("--cache_dir", type=str, default="")
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--eval_neg", type=int, default=0)
    parser.add_argument("--eval_seed", type=int, default=2025)

    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--save_metric_k", type=int, default=10)
    parser.add_argument("--early_stop_patience", type=int, default=0,
                        help="If >0, stop training when monitored Val metric does not improve for this many consecutive epochs")
    parser.add_argument("--early_stop_min_delta", type=float, default=0.0,
                        help="Minimum improvement in monitored metric to reset patience")

    args = parser.parse_args()
    set_seed(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)

    cache_npz = ""
    cache_meta = ""
    if args.cache_dir:
        os.makedirs(args.cache_dir, exist_ok=True)
        cache_npz = os.path.join(args.cache_dir, "sasrec_cache.npz")
        cache_meta = os.path.join(args.cache_dir, "sasrec_meta.json")

    if cache_npz and os.path.exists(cache_npz) and os.path.exists(cache_meta):
        print(f"[Data] Loading cached sequences: {cache_npz}")
        pack = np.load(cache_npz, allow_pickle=True)
        user_seqs = pack["user_seqs"].tolist()
        with open(cache_meta, "r", encoding="utf-8") as f:
            meta = json.load(f)
        vid2iid = {int(k): int(v) for k, v in meta["vid2iid"].items()}
        uid2uix = {int(k): int(v) for k, v in meta["uid2uix"].items()}
        n_items_total = int(meta["n_items_total"])
    else:
        print(f"[Data] Building GLOBAL mapping + click sequences from CSV: {args.csv_path}")
        user_seqs, vid2iid, uid2uix, n_items_total = build_global_mapping_and_click_sequences(args.csv_path)

        clicks_total = sum(len(s) for s in user_seqs)
        print(f"[Data] n_users={len(user_seqs)} n_items_total={n_items_total} clicks_total={clicks_total}")

        if cache_npz:
            np.savez_compressed(cache_npz, user_seqs=np.array(user_seqs, dtype=object))
            save_json({"vid2iid": vid2iid, "uid2uix": uid2uix, "n_items_total": n_items_total}, cache_meta)
            print(f"[Data] Saved cache to {cache_npz} and {cache_meta}")

    n_users = len(user_seqs)
    clicks_total = sum(len(s) for s in user_seqs)
    print(f"[Data] Summary: n_users={n_users} n_items_total={n_items_total} clicks_total={clicks_total}")

    mx = max((max(s) for s in user_seqs if len(s) > 0), default=0)
    mn = min((min(s) for s in user_seqs if len(s) > 0), default=0)
    print(f"[Data] Sanity: seq_id_min={mn} seq_id_max={mx}")
    if mx > n_items_total or mn < 1:
        raise RuntimeError(f"[Data] ID out of range: min={mn} max={mx} n_items_total={n_items_total}")

    train_ds = SASRecTrainDataset(user_seqs=user_seqs, max_len=args.max_len)
    val_ds = SASRecValDataset(user_seqs=user_seqs, max_len=args.max_len)

    print(f"[Data] Train samples={len(train_ds)} Val users={len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

    cfg = SASRecConfig(
        n_items=n_items_total,
        max_len=args.max_len,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
    )
    device = torch.device(args.device if torch.cuda.is_available() or "cuda" not in args.device else "cpu")
    model = SASRec(cfg).to(device)

    n_params = count_parameters(model)
    print(f"[Model] {cfg}")
    print(f"[Model] Total parameters: {n_params:,}")

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

    for ep in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        loss_sum = 0.0
        n_step = 0

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

            loss = model.compute_loss_ce(seq, target)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optim.step()

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
