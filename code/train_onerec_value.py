# -*- coding: utf-8 -*-
"""
Effect-first trainer for OneRecWithValue (Encoder-Decoder + Value head).

This is a clean retrain script (no backward-compat branches).

Dataset:
  - KRPureValueDataset should yield a batch:
      target_sid:   [B, L] long
      user_feat:    [B, F] float
      hist_sid:     [B, H, L] long   (PAD is 0; PAD is at the END)
      hist_len:     [B] long         (number of valid history items)
      reward_label: [B, L] float     (per-token short-term reward label)
      ltv_label:    [B] float or [B,1] float

Validation:
  - Constrained Beam Search generates K SID sequences and checks exact SID match.
  - Reports Recall@K / NDCG@K over the generated beam list.
"""
from __future__ import annotations

import os
import argparse
import random
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from model.onerec_value import OneRecWithValue
from dataset_krpure_value import KRPureValueDataset


def set_seed(seed: int = 2025):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--log_paths",
        type=str,
        default="",
        help=" log csv ",
    )
    parser.add_argument("--sid_mapping_path", type=str, required=True)
    parser.add_argument("--user_feat_path", type=str, required=True)
    parser.add_argument("--label_col", type=str, default="is_click")

    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--min_hist_len", type=int, default=1)
    parser.add_argument("--max_hist_len", type=int, default=50)

    parser.add_argument("--model_size", type=str, default="mini", choices=["mini", "medium", "large"])
    parser.add_argument("--sid_depth", type=int, default=4)
    parser.add_argument("--num_classes", type=int, default=32)

    parser.add_argument("--num_layers", type=int, default=3, help="decoder layers")
    parser.add_argument("--hist_num_layers", type=int, default=2, help="encoder layers")
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--use_user_token", action="store_true", help="Add [USER] token into encoder input")
    parser.add_argument("--use_decoder_ctx", action="store_true", help="Add pooled ctx into decoder input")
    parser.add_argument("--value_layers", type=int, default=2)

    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=2025)

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--grad_accum_steps", type=int, default=1)

    parser.add_argument("--w_rev", type=float, default=0.5)
    parser.add_argument("--w_ltv", type=float, default=0.5)
    parser.add_argument("--w_cls", type=float, default=1.0)
    parser.add_argument("--label_smoothing", type=float, default=0.0)

    parser.add_argument(
        "--value_token_weights",
        type=str,
        default="0,0,0,1",
        help="L  token ,  0,0,0,1 (); ",
    )

    parser.add_argument("--train_value", action="store_true", help="Whether to train value heads")
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--save_every", type=int, default=1)

    parser.add_argument("--beam_width", type=int, default=50)
    parser.add_argument("--topk_list", type=str, default="1,5,10")
    parser.add_argument("--early_stop_patience", type=int, default=5)

    return parser.parse_args()


def model_dim_from_size(model_size: str) -> Tuple[int, int]:
    if model_size == "mini":
        return 128, 4
    if model_size == "medium":
        return 256, 8
    if model_size == "large":
        return 512, 8
    raise ValueError(f"Unknown model_size={model_size}")


def build_trie_from_sid_mapping(
    mapping_path: str,
    sid_cols: List[str],
    sid_depth: int,
    num_classes: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build trie for constrained beam search.
    Returns:
      trie_mask: [num_nodes, V] float (-inf for invalid, 0 for valid)
      trie_next: [num_nodes, V] long (next node id)
    """
    df = pd.read_csv(mapping_path)

    id_col = None
    for c in ["video_id", "item_id", "iid", "videoId"]:
        if c in df.columns:
            id_col = c
            break
    if id_col is None:
        df = df[sid_cols].drop_duplicates()
    else:
        df = df[[id_col] + sid_cols].drop_duplicates(id_col)

    paths = torch.tensor(df[sid_cols].values, dtype=torch.long, device=device)
    if paths.size(1) != sid_depth:
        raise ValueError(f"sid_cols={sid_cols} must have length sid_depth={sid_depth}")

    trie_graph: Dict[int, Dict[int, int]] = {0: {}}
    node_count = 1
    for p in paths:
        node = 0
        for tok in p.tolist():
            tok = int(tok)
            if tok < 0 or tok >= num_classes:
                node = None
                break
            if tok not in trie_graph[node]:
                trie_graph[node][tok] = node_count
                trie_graph[node_count] = {}
                node_count += 1
            node = trie_graph[node][tok]
        if node is None:
            continue

    trie_mask = torch.full((node_count, num_classes), -float("inf"), device=device)
    trie_next = torch.zeros((node_count, num_classes), dtype=torch.long, device=device)
    for u, trans in trie_graph.items():
        for tok, v in trans.items():
            trie_mask[u, tok] = 0.0
            trie_next[u, tok] = v
    print(f"[Trie] Built from {paths.shape[0]} paths, nodes={node_count}, vocab={num_classes}")
    return trie_mask, trie_next


@torch.no_grad()
def constrained_beam_search_sid(
    model: OneRecWithValue,
    enc_out: torch.Tensor,
    enc_mask: torch.Tensor,
    sid_depth: int,
    num_classes: int,
    trie_mask: torch.Tensor,
    trie_next: torch.Tensor,
    beam_width: int,
    ctx: Optional[torch.Tensor] = None,  # [B,D] or None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Constrained beam search that uses model.decoder (cross-attn to enc_out).

    Returns:
      sequences: [B, W, L] (without BOS)
      log_probs: [B, W]
    """
    device = enc_out.device
    B = enc_out.size(0)
    W = beam_width
    D = model.hid_dim

    sequences = torch.zeros((B, W, 1), dtype=torch.long, device=device)
    logp = torch.full((B, W), -float("inf"), device=device)
    logp[:, 0] = 0.0
    nodes = torch.zeros((B, W), dtype=torch.long, device=device)  # trie node per beam

    if ctx is not None:
        ctx_flat = ctx.unsqueeze(1).expand(B, W, -1).reshape(B * W, -1)
    else:
        ctx_flat = None

    enc_flat = enc_out.unsqueeze(1).expand(B, W, -1, -1).reshape(B * W, enc_out.size(1), D)
    msk_flat = enc_mask.unsqueeze(1).expand(B, W, -1).reshape(B * W, enc_mask.size(1))

    for t in range(sid_depth):
        prev = sequences[:, :, 1:]  # [B,W,t]
        BW = B * W
        T = t + 1
        x = torch.zeros((BW, T, D), device=device)
        x[:, 0, :] = model.bos
        if t > 0:
            prev_flat = prev.reshape(BW, t)
            x[:, 1:, :] = model.sid_embedding(prev_flat)

        if hasattr(model, "dec_pos_emb"):
            pos = torch.arange(T, device=device)
            x = x + model.dec_pos_emb(pos)[None, :, :]
        else:
            pos = torch.arange(T, device=device)
            x = x + model.pos_embedding(pos)[None, :, :]

        if ctx_flat is not None:
            x = x + ctx_flat.unsqueeze(1)

        tgt_mask = model._build_causal_mask(T, device=device)
        dec = model.decoder(tgt=x, tgt_mask=tgt_mask, memory=enc_flat, memory_key_padding_mask=msk_flat)
        logits = model.out_proj(dec[:, -1, :])  # [BW,V]
        logits = logits.view(B, W, num_classes)

        cur_nodes = nodes  # [B,W]
        mask = trie_mask[cur_nodes]  # [B,W,V]
        logits = logits + mask

        log_probs_step = torch.log_softmax(logits, dim=-1)  # [B,W,V]
        total = logp.unsqueeze(-1) + log_probs_step         # [B,W,V]

        total_flat = total.view(B, W * num_classes)
        topv, topi = torch.topk(total_flat, k=W, dim=-1)  # [B,W]

        next_beam = topi // num_classes
        next_tok = topi % num_classes

        new_seq = torch.gather(sequences, 1, next_beam.unsqueeze(-1).expand(-1, -1, sequences.size(-1)))
        new_seq = torch.cat([new_seq, next_tok.unsqueeze(-1)], dim=-1)  # append token
        sequences = new_seq
        logp = topv

        prev_nodes = torch.gather(nodes, 1, next_beam)
        nodes = trie_next[prev_nodes, next_tok]

    return sequences[:, :, 1:], logp  # drop placeholder


@torch.no_grad()
def evaluate_beam(
    model: OneRecWithValue,
    data_loader: DataLoader,
    device: torch.device,
    sid_depth: int,
    topk_list: List[int],
    trie_mask: torch.Tensor,
    trie_next: torch.Tensor,
    beam_width: int,
) -> Dict[str, float]:
    model.eval()
    beam_width = max(beam_width, max(topk_list))
    num_classes = model.sid_embedding.num_embeddings

    hits = {k: 0.0 for k in topk_list}
    ndcgs = {k: 0.0 for k in topk_list}
    total = 0

    for batch in data_loader:
        target_sid, user_feat, hist_sid, hist_len, reward_label, ltv_label = batch

        target_sid = target_sid.to(device).long()
        user_feat = user_feat.to(device).float()
        hist_sid = hist_sid.to(device).long()
        hist_len = hist_len.to(device).long()

        B = target_sid.size(0)
        total += B

        user_vec = model.user_proj(model._pad_or_trim_user_feat(user_feat, model.user_proj.in_features))
        enc_out, enc_mask = model.encode_history_seq(hist_sid, hist_len, user_ctx=user_vec)

        ctx = None
        if getattr(model, "use_decoder_ctx", False):
            ctx = model.build_ctx(enc_out, enc_mask, user_vec)

        gen, _ = constrained_beam_search_sid(
            model=model,
            enc_out=enc_out,
            enc_mask=enc_mask,
            sid_depth=sid_depth,
            num_classes=num_classes,
            trie_mask=trie_mask,
            trie_next=trie_next,
            beam_width=beam_width,
            ctx=ctx,
        )  # [B,W,L]

        W = gen.size(1)
        eq = (gen == target_sid.unsqueeze(1).expand(-1, W, -1)).all(dim=-1)  # [B,W]

        hit_any = eq.any(dim=1)
        first_idx = torch.where(
            hit_any,
            eq.float().argmax(dim=1),
            torch.full((B,), -1, device=device, dtype=torch.long),
        )

        for k in topk_list:
            in_topk = (first_idx >= 0) & (first_idx < k)
            hits[k] += in_topk.float().sum().item()
            nd = torch.zeros((B,), device=device)
            r = first_idx[in_topk].float()
            nd[in_topk] = 1.0 / torch.log2(r + 2.0)
            ndcgs[k] += nd.sum().item()

    metrics = {}
    for k in topk_list:
        metrics[f"recall@{k}"] = hits[k] / max(1, total)
        metrics[f"ndcg@{k}"] = ndcgs[k] / max(1, total)
    return metrics


def main():
    args = parse_args()
    os.makedirs(args.model_dir, exist_ok=True)
    set_seed(args.seed)

    device = torch.device(args.device)

    log_paths = [p.strip() for p in args.log_paths.split(",") if p.strip()]
    if len(log_paths) == 0:
        raise ValueError("--log_paths is empty. Provide at least one csv path.")

    full_dataset = KRPureValueDataset(
        log_paths=log_paths,
        sid_mapping_path=args.sid_mapping_path,
        user_feat_path=args.user_feat_path,
        label_col=args.label_col,
        gamma=args.gamma,
        min_hist_len=args.min_hist_len,
        max_hist_len=args.max_hist_len,
    )

    num_samples = len(full_dataset)
    sample_uids = full_dataset.sample_user_ids.numpy()  # [N]

    uid_last_idx: Dict[int, int] = {}
    for idx, uid in enumerate(sample_uids.tolist()):
        uid_last_idx[int(uid)] = idx

    val_indices = sorted(uid_last_idx.values())
    val_set = set(val_indices)
    train_indices = [i for i in range(num_samples) if i not in val_set]

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    print(f"[Data] N={num_samples} train={len(train_dataset)} val={len(val_dataset)}")

    hid_dim, default_head = model_dim_from_size(args.model_size)
    nhead = args.nhead if args.nhead > 0 else default_head

    model = OneRecWithValue(
        num_decoder_block=args.num_layers,
        hid_dim=hid_dim,
        nhead=nhead,
        sid_depth=args.sid_depth,
        num_classes=args.num_classes,
        user_feat_dim=full_dataset.user_feat_dim,
        max_hist_len=args.max_hist_len,
        hist_num_layers=args.hist_num_layers,
        dropout_ratio=args.dropout,
        use_user_token=args.use_user_token,
        use_decoder_ctx=args.use_decoder_ctx,
        value_layers=args.value_layers,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    df_map = pd.read_csv(args.sid_mapping_path, nrows=5)
    sid_cols = [c for c in df_map.columns if c.startswith("sid_")]
    sid_cols = sid_cols[:args.sid_depth]
    trie_mask, trie_next = build_trie_from_sid_mapping(
        args.sid_mapping_path, sid_cols=sid_cols, sid_depth=args.sid_depth, num_classes=args.num_classes, device=device
    )

    token_weights = None
    if args.value_token_weights.strip():
        w = [float(x) for x in args.value_token_weights.split(",")]
        if len(w) != args.sid_depth:
            raise ValueError(f"--value_token_weights must have L={args.sid_depth} numbers, got {w}")
        token_weights = torch.tensor(w, dtype=torch.float32, device=device)

    topk_list = [int(x) for x in args.topk_list.split(",") if x.strip()]
    best_recall5 = -1.0
    bad_epochs = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        sum_loss = 0.0
        sum_cls = 0.0
        sum_rev = 0.0
        sum_ltv = 0.0
        n_batches = 0

        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_loader):
            target_sid, user_feat, hist_sid, hist_len, reward_label, ltv_label = batch
            target_sid = target_sid.to(device).long()
            user_feat = user_feat.to(device).float()
            hist_sid = hist_sid.to(device).long()
            hist_len = hist_len.to(device).long()
            reward_label = reward_label.to(device).float()
            ltv_label = ltv_label.to(device).float()

            w_rev = args.w_rev if args.train_value else 0.0
            w_ltv = args.w_ltv if args.train_value else 0.0

            loss, logs, _ = model.compute_loss_with_value(
                target_sid=target_sid,
                user_feat=user_feat,
                hist_sid=hist_sid,
                hist_len=hist_len,
                immediate_rewards=reward_label,
                ltv=ltv_label,
                token_weights=token_weights,
                w_cls=args.w_cls,
                w_rev=w_rev,
                w_ltv=w_ltv,
                label_smoothing=args.label_smoothing,
                value_only=False,
            )

            (loss / args.grad_accum_steps).backward()

            if (step + 1) % args.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            sum_loss += logs["loss"]
            sum_cls += logs["cls"]
            sum_rev += logs["rev"]
            sum_ltv += logs["ltv"]
            n_batches += 1

            if (step + 1) % 50 == 0:
                print(f"[Epoch {epoch}] step {step+1}/{len(train_loader)} loss={sum_loss/n_batches:.4f} cls={sum_cls/n_batches:.4f} rev={sum_rev/n_batches:.4f} ltv={sum_ltv/n_batches:.4f}")

        metrics = evaluate_beam(
            model=model,
            data_loader=val_loader,
            device=device,
            sid_depth=args.sid_depth,
            topk_list=topk_list,
            trie_mask=trie_mask,
            trie_next=trie_next,
            beam_width=args.beam_width,
        )
        metric_str = " ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
        print(f"[Val][Epoch {epoch}] {metric_str}")

        if epoch % args.save_every == 0:
            ckpt_path = os.path.join(args.model_dir, f"epoch_{epoch}.pt")
            torch.save({"model": model.state_dict(), "args": vars(args)}, ckpt_path)
            print(f"[Save] {ckpt_path}")

        recall5 = metrics.get("recall@5", metrics.get("recall@1", 0.0))
        if recall5 > best_recall5:
            best_recall5 = recall5
            bad_epochs = 0
            best_path = os.path.join(args.model_dir, "best.pt")
            torch.save({"model": model.state_dict(), "args": vars(args)}, best_path)
            print(f"[Best] Updated best.pt (recall@5={best_recall5:.4f})")
        else:
            bad_epochs += 1
            if bad_epochs >= args.early_stop_patience:
                print(f"[EarlyStop] no improvement for {bad_epochs} epochs.")
                break

    last_path = os.path.join(args.model_dir, "last.pt")
    torch.save({"model": model.state_dict(), "args": vars(args)}, last_path)
    print(f"[Save] {last_path}")


if __name__ == "__main__":
    main()
