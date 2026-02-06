# -*- coding: utf-8 -*-

import argparse
import os
import time
from copy import deepcopy
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple

import re

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

try:
    from dataset_krpure_value_gt import KRPureValueDataset as _DS
    print("[Data] Using dataset_krpure_value_gt.KRPureValueDataset")
except ImportError:
    from dataset_krpure_value import KRPureValueDataset as _DS
    print("[Data] Using dataset_krpure_value.KRPureValueDataset")
KRPureValueDataset = _DS

from model.onerec_value import OneRecWithValue


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def model_dim_from_size(model_size: str) -> Tuple[int, int]:
    model_size = model_size.lower()
    if model_size in ["mini", "s", "small"]:
        return 128, 4
    if model_size in ["medium", "m"]:
        return 256, 6
    if model_size in ["large", "l"]:
        return 512, 8
    try:
        d = int(model_size)
        return d, max(4, d // 64)
    except Exception as e:
        raise ValueError(f"Unknown --model_size={model_size}") from e


@contextmanager
def temporary_mode(model: torch.nn.Module, train: bool):
    was_train = model.training
    model.train(train)
    try:
        yield
    finally:
        model.train(was_train)


def safe_load_state_dict(model: torch.nn.Module, state_dict: dict, *, strict: bool = False, verbose: bool = True):
    model_sd = model.state_dict()

    filtered = {}
    mismatched = []
    unexpected = []
    for k, v in state_dict.items():
        if k not in model_sd:
            unexpected.append(k)
            continue
        if tuple(v.shape) != tuple(model_sd[k].shape):
            mismatched.append((k, tuple(v.shape), tuple(model_sd[k].shape)))
            continue
        filtered[k] = v

    missing = [k for k in model_sd.keys() if k not in filtered]

    if strict:
        if unexpected or mismatched or missing:
            raise RuntimeError(
                "[SafeLoad][STRICT] state_dict not compatible:\n"
                f"  missing={len(missing)} unexpected={len(unexpected)} mismatched={len(mismatched)}"
            )
        model.load_state_dict(state_dict, strict=True)
    else:
        model.load_state_dict(filtered, strict=False)

    if verbose:
        print(f"[SafeLoad] loaded={len(filtered)} missing={len(missing)} unexpected={len(unexpected)} mismatched={len(mismatched)}")
        if unexpected:
            print("[SafeLoad] unexpected examples (up to 20):")
            for k in unexpected[:20]:
                print(f"  - {k}")
        if mismatched:
            print("[SafeLoad] shape-mismatch examples (up to 20):")
            for k, s_ckpt, s_now in mismatched[:20]:
                print(f"  - {k}: ckpt={s_ckpt} now={s_now}")


def build_trie_from_sid_mapping(
    sid_mapping_path: str,
    sid_depth: int,
    num_classes: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    import pandas as pd
    df = pd.read_csv(sid_mapping_path)
    sid_cols = [f"sid_{i+1}" for i in range(sid_depth)]
    for c in sid_cols:
        if c not in df.columns:
            raise ValueError(f"sid_mapping csv missing column {c}. Columns={list(df.columns)[:20]}...")
    paths = df[sid_cols].values.astype(np.int64)

    trie_graph: Dict[int, Dict[int, int]] = {0: {}}
    node_count = 1
    for p in paths:
        node = 0
        ok = True
        for tok in p.tolist():
            tok = int(tok)
            if tok < 0 or tok >= num_classes:
                ok = False
                break
            if tok not in trie_graph[node]:
                trie_graph[node][tok] = node_count
                trie_graph[node_count] = {}
                node_count += 1
            node = trie_graph[node][tok]
        if not ok:
            continue

    trie_mask = torch.full((node_count, num_classes), -float("inf"), device=device)
    trie_next = torch.zeros((node_count, num_classes), dtype=torch.long, device=device)
    for node in range(node_count):
        nxt = trie_graph.get(node, {})
        for tok, nn in nxt.items():
            trie_mask[node, tok] = 0.0
            trie_next[node, tok] = int(nn)

    print(f"[Trie] Built from {len(paths)} paths, nodes={node_count}, vocab={num_classes}, sid_depth={sid_depth}")
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
    ctx: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = enc_out.device
    B = enc_out.size(0)
    D = enc_out.size(-1)
    W = beam_width

    sequences = torch.zeros((B, W, 1), device=device, dtype=torch.long)
    nodes = torch.zeros((B, W), device=device, dtype=torch.long)
    logp = torch.full((B, W), -float("inf"), device=device)
    logp[:, 0] = 0.0

    ctx_flat = None
    if ctx is not None:
        ctx_flat = ctx.unsqueeze(1).expand(B, W, -1).reshape(B * W, -1)

    enc_flat = enc_out.unsqueeze(1).expand(B, W, -1, -1).reshape(B * W, enc_out.size(1), D)
    msk_flat = enc_mask.unsqueeze(1).expand(B, W, -1).reshape(B * W, enc_mask.size(1))

    for t in range(sid_depth):
        prev = sequences[:, :, 1:]
        BW = B * W
        T = t + 1

        x = torch.zeros((BW, T, D), device=device)
        x[:, 0, :] = model.bos
        if t > 0:
            prev_flat = prev.reshape(BW, t)
            x[:, 1:, :] = model.sid_embedding(prev_flat)

        pos = torch.arange(T, device=device)
        if hasattr(model, "dec_pos_emb"):
            x = x + model.dec_pos_emb(pos)[None, :, :]
        else:
            x = x + model.pos_embedding(pos)[None, :, :]

        if ctx_flat is not None:
            x = x + ctx_flat.unsqueeze(1)

        if hasattr(model, "_build_causal_mask"):
            tgt_mask = model._build_causal_mask(T, device=device)
        else:
            subsequent = torch.triu(torch.ones((T, T), device=device, dtype=torch.bool), diagonal=1)
            tgt_mask = torch.full((T, T), float("-inf"), device=device)
            tgt_mask = tgt_mask.masked_fill(~subsequent, 0.0)

        dec_feats = model.decoder(
            tgt=x,
            tgt_mask=tgt_mask,
            memory=enc_flat,
            memory_key_padding_mask=msk_flat,
        )
        last = dec_feats[:, -1, :]
        logits = model.out_proj(last).view(B, W, num_classes)
        logits = logits + trie_mask[nodes]
        lp = F.log_softmax(logits, dim=-1)

        total_lp = (logp.unsqueeze(-1) + lp).view(B, W * num_classes)
        topv, topi = torch.topk(total_lp, k=W, dim=-1)

        next_beam = topi // num_classes
        next_tok = topi % num_classes

        new_seq = torch.gather(
            sequences, 1, next_beam.unsqueeze(-1).expand(-1, -1, sequences.size(-1))
        )
        sequences = torch.cat([new_seq, next_tok.unsqueeze(-1)], dim=-1)
        logp = topv

        prev_nodes = torch.gather(nodes, 1, next_beam)
        nodes = trie_next[prev_nodes, next_tok]

    return sequences[:, :, 1:], logp


def encode_prompt(
    model: OneRecWithValue,
    user_feat: torch.Tensor,
    hist_sid: torch.Tensor,
    hist_len: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    user_feat = model._pad_or_trim_user_feat(
        user_feat, model.user_proj.in_features, batch_size=user_feat.size(0)
    ).to(user_feat.device)
    user_vec = model.user_proj(user_feat)
    enc_out, enc_mask = model.encode_history_seq(hist_sid, hist_len, user_ctx=user_vec)

    ctx: Optional[torch.Tensor] = None
    if getattr(model, "use_decoder_ctx", False):
        if hasattr(model, "build_ctx") and callable(getattr(model, "build_ctx")):
            ctx = model.build_ctx(enc_out, enc_mask, user_vec)  # type: ignore[attr-defined]
        else:
            valid = (~enc_mask).float().unsqueeze(-1)
            denom = valid.sum(dim=1).clamp_min(1.0)
            hist_vec = (enc_out * valid).sum(dim=1) / denom
            ctx = model.ctx_proj(torch.cat([hist_vec, user_vec], dim=-1))
    return enc_out, enc_mask, ctx


def compute_seq_logprobs(
    model: OneRecWithValue,
    enc_out: torch.Tensor,
    enc_mask: torch.Tensor,
    seq: torch.Tensor,  # [BW,L]
    trie_mask: torch.Tensor,
    trie_next: torch.Tensor,
    ctx: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = enc_out.device
    BW, L = seq.shape
    D = enc_out.size(-1)

    x = torch.zeros((BW, L, D), device=device)
    x[:, 0, :] = model.bos
    if L > 1:
        x[:, 1:, :] = model.sid_embedding(seq[:, :-1])

    pos = torch.arange(L, device=device)
    if hasattr(model, "dec_pos_emb"):
        x = x + model.dec_pos_emb(pos)[None, :, :]
    else:
        x = x + model.pos_embedding(pos)[None, :, :]

    if ctx is not None:
        x = x + ctx.unsqueeze(1)

    if hasattr(model, "_build_causal_mask"):
        tgt_mask = model._build_causal_mask(L, device=device)
    else:
        subsequent = torch.triu(torch.ones((L, L), device=device, dtype=torch.bool), diagonal=1)
        tgt_mask = torch.full((L, L), float("-inf"), device=device)
        tgt_mask = tgt_mask.masked_fill(~subsequent, 0.0)

    dec = model.decoder(tgt=x, tgt_mask=tgt_mask, memory=enc_out, memory_key_padding_mask=enc_mask)
    logits = model.out_proj(dec)  # [BW,L,V]

    node = torch.zeros((BW,), device=device, dtype=torch.long)
    nodes = []
    for t in range(L):
        nodes.append(node)
        node = trie_next[node, seq[:, t]]
    nodes = torch.stack(nodes, dim=1)

    logits = logits + trie_mask[nodes]
    log_probs = F.log_softmax(logits, dim=-1)
    logp_tok = log_probs.gather(-1, seq.unsqueeze(-1)).squeeze(-1)  # [BW,L]
    logp_sum = logp_tok.sum(dim=1)  # [BW]
    return logp_sum, logp_tok


def ntp_loss_on_gt(
    model: OneRecWithValue,
    target_sid: torch.Tensor,
    user_feat: torch.Tensor,
    hist_sid: torch.Tensor,
    hist_len: torch.Tensor,
    sample_w: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    logits, _, _, _, _, _ = model.forward_with_cache(
        target_sid=target_sid,
        user_feat=user_feat,
        hist_sid=hist_sid,
        hist_len=hist_len,
    )
    B, L = target_sid.shape
    V = logits.size(-1)
    ce = F.cross_entropy(logits.view(-1, V), target_sid.view(-1), reduction="none").view(B, L)
    per_sample = ce.mean(dim=1)
    if sample_w is None:
        return per_sample.mean()
    w = sample_w.float().view(B)
    denom = w.sum().clamp(min=1e-6)
    return (per_sample * w).sum() / denom


def pick_negatives_from_beam(
    gt: torch.Tensor,           # [B,L]
    beam_seq: torch.Tensor,     # [B,W,L]
    k: int,
    neg_pick: str = "top",
    rank_low: int = 0,
    rank_high: int = 0,
) -> torch.Tensor:
    '''
    neg_pick:
      - top: take best-ranked negatives (like v1)
      - random: random sample from all negatives in beam (excluding GT)
      - rank_range: random sample from [rank_low, rank_high) within the negative list (excluding GT)
    '''
    B, W, L = beam_seq.shape
    device = beam_seq.device
    neg = torch.zeros((B, k, L), device=device, dtype=torch.long)

    eq = (beam_seq == gt.unsqueeze(1)).all(dim=-1)  # [B,W]
    for i in range(B):
        idx = [j for j in range(W) if not bool(eq[i, j].item())]
        if len(idx) == 0:
            neg[i] = gt[i].unsqueeze(0).expand(k, -1)
            continue

        if neg_pick == "top":
            take = idx[:k]
        elif neg_pick == "random":
            take = np.random.choice(idx, size=k, replace=(len(idx) < k)).tolist()
        elif neg_pick == "rank_range":
            lo = max(0, int(rank_low))
            hi = int(rank_high) if int(rank_high) > 0 else len(idx)
            lo = min(lo, len(idx))
            hi = min(max(lo + 1, hi), len(idx))
            pool = idx[lo:hi]
            take = np.random.choice(pool, size=k, replace=(len(pool) < k)).tolist()
        else:
            raise ValueError(f"Unknown neg_pick={neg_pick}")

        picked = beam_seq[i, take]
        if picked.size(0) < k:
            pad = picked[-1:].expand(k - picked.size(0), -1)
            picked = torch.cat([picked, pad], dim=0)
        neg[i] = picked
    return neg


def softmax_dpo_loss(
    logp_pi_chosen: torch.Tensor,    # [B]
    logp_pi_rejected: torch.Tensor,  # [B,K]
    logp_ref_chosen: torch.Tensor,   # [B]
    logp_ref_rejected: torch.Tensor, # [B,K]
    beta: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    chosen_lr = logp_pi_chosen - logp_ref_chosen
    rejected_lr = logp_pi_rejected - logp_ref_rejected
    temp = torch.exp(beta * (rejected_lr - chosen_lr.unsqueeze(1))).sum(dim=1)
    loss = torch.log1p(temp)

    with torch.no_grad():
        pref_acc = (chosen_lr > rejected_lr.max(dim=1).values).float().mean()
        stats = {
            "chosen_lr": float(chosen_lr.mean().cpu()),
            "rej_lr": float(rejected_lr.mean().cpu()),
            "temp": float(temp.mean().cpu()),
            "pref_acc": float(pref_acc.cpu()),
        }
    return loss, stats


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--log_paths", type=str, required=True)
    p.add_argument("--sid_mapping_path", type=str, required=True)
    p.add_argument("--user_feat_path", type=str, required=True)
    p.add_argument("--label_col", type=str, default="is_click")
    p.add_argument("--gt_spec", type=str, default="")
    p.add_argument("--gt_gate", type=str, default="")
    p.add_argument("--hist_from_gt", type=int, default=0)
    p.add_argument("--gt_weight_norm", type=int, default=1)
    p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument("--min_hist_len", type=int, default=1)
    p.add_argument("--max_hist_len", type=int, default=50)

    p.add_argument("--model_size", type=str, default="small")
    p.add_argument("--num_layers", type=int, default=3)
    p.add_argument("--nhead", type=int, default=-1)
    p.add_argument("--sid_depth", type=int, default=4)
    p.add_argument("--num_classes", type=int, default=32)
    p.add_argument("--max_hist_len_model", type=int, default=50)
    p.add_argument("--use_decoder_ctx", action="store_true")
    p.add_argument("--init_ckpt", type=str, default="")

    p.add_argument("--num_neg", type=int, default=1)
    p.add_argument("--neg_beam", type=int, default=12)
    p.add_argument("--beta", type=float, default=0.01)
    p.add_argument("--dpo_coef", type=float, default=1.0)
    p.add_argument("--sft_coef", type=float, default=0.5)
    p.add_argument("--dpo_last_n", type=int, default=0, help="0=all tokens; e.g., 2 means DPO only on last 2 SID tokens")

    p.add_argument("--neg_pick", type=str, default="top", choices=["top", "random", "rank_range"])
    p.add_argument("--neg_rank_low", type=int, default=2)
    p.add_argument("--neg_rank_high", type=int, default=12)


    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--eval_every", type=int, default=200)
    p.add_argument("--model_dir", type=str, default="checkpoints_sdpo_v2")

    p.add_argument("--eval_beam", type=int, default=50)
    p.add_argument("--topk", type=str, default="1,5,10,20,50")
    p.add_argument("--sp_iters", type=int, default=3, help="Self-play iterations per epoch (each: SFT pass then DPO pass)")
    p.add_argument("--sft_steps", type=int, default=-1, help="Number of SFT update steps per iteration; -1 = full train_loader")
    p.add_argument("--dpo_steps", type=int, default=-1, help="Number of DPO update steps per iteration; -1 = full train_loader")
    p.add_argument("--eval_every_iter", type=int, default=1, help="Run evaluation every N self-play iterations")
    p.add_argument("--rand_neg_ratio", type=float, default=0.0, help="Replace this ratio of negatives with random SID paths (ablation)")

    return p.parse_args()


@torch.no_grad()
def evaluate_hit_ndcg(
    model: OneRecWithValue,
    data_loader: DataLoader,
    device: torch.device,
    trie_mask: torch.Tensor,
    trie_next: torch.Tensor,
    beam_width: int,
    topk_list: List[int],
):
    model.eval()
    V = model.sid_embedding.num_embeddings

    hits = {k: 0.0 for k in topk_list}
    ndcgs = {k: 0.0 for k in topk_list}
    total = 0

    for batch in data_loader:
        target_sid, user_feat, hist_sid, hist_len, _, _ = batch
        target_sid = target_sid.to(device).long()
        user_feat = user_feat.to(device).float()
        hist_sid = hist_sid.to(device).long()
        hist_len = hist_len.to(device).long()

        B, L = target_sid.shape
        total += B

        enc_out, enc_mask, ctx = encode_prompt(model, user_feat, hist_sid, hist_len)
        gen, _ = constrained_beam_search_sid(
            model=model,
            enc_out=enc_out,
            enc_mask=enc_mask,
            sid_depth=L,
            num_classes=V,
            trie_mask=trie_mask,
            trie_next=trie_next,
            beam_width=max(beam_width, max(topk_list)),
            ctx=ctx,
        )

        W = gen.size(1)
        eq = (gen == target_sid.unsqueeze(1).expand(-1, W, -1)).all(dim=-1)
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
            rank = first_idx[in_topk].float()
            nd[in_topk] = 1.0 / torch.log2(rank + 2.0)
            ndcgs[k] += nd.sum().item()

    out = {}
    for k in topk_list:
        out[f"hit@{k}"] = hits[k] / max(1, total)
        out[f"ndcg@{k}"] = ndcgs[k] / max(1, total)
    return out


@torch.no_grad()
def build_valid_tokens_per_node(trie_mask: torch.Tensor) -> List[torch.Tensor]:
    """Precompute valid token lists (where trie_mask[node, tok] == 0)."""
    valid = []
    for n in range(trie_mask.size(0)):
        valid.append(torch.nonzero(trie_mask[n] == 0.0, as_tuple=False).view(-1))
    return valid


@torch.no_grad()
def sample_random_paths_flat(
    valid_tokens: List[torch.Tensor],
    trie_next: torch.Tensor,
    n: int,
    sid_depth: int,
    device: torch.device,
) -> torch.Tensor:
    """Sample n random valid SID paths (shape: [n, sid_depth])."""
    out = torch.empty((n, sid_depth), dtype=torch.long, device=device)
    for i in range(n):
        node = 0
        for t in range(sid_depth):
            vt = valid_tokens[node]
            tok = vt[torch.randint(0, vt.numel(), (1,), device=device)].item()
            out[i, t] = tok
            node = int(trie_next[node, tok].item())
    return out


@torch.no_grad()
def apply_random_negative_replacement(
    neg_sid: torch.Tensor,     # [B,K,L]
    gt_sid: torch.Tensor,      # [B,L]
    rand_ratio: float,
    valid_tokens: List[torch.Tensor],
    trie_next: torch.Tensor,
) -> torch.Tensor:
    """Randomly replace a ratio of negatives with random valid paths (ablation)."""
    if rand_ratio <= 0:
        return neg_sid
    B, K, L = neg_sid.shape
    device = neg_sid.device
    mask = (torch.rand((B, K), device=device) < float(rand_ratio))
    if not mask.any():
        return neg_sid
    flat_mask = mask.view(-1)
    n = int(flat_mask.sum().item())
    rand_flat = sample_random_paths_flat(valid_tokens, trie_next, n=n, sid_depth=L, device=device)  # [n,L]
    neg_flat = neg_sid.view(-1, L).clone()
    neg_flat[flat_mask] = rand_flat
    gt_rep = gt_sid.unsqueeze(1).expand(B, K, L).reshape(-1, L)
    eq = (neg_flat == gt_rep).all(dim=1)
    if eq.any():
        m = int(eq.sum().item())
        rand2 = sample_random_paths_flat(valid_tokens, trie_next, n=m, sid_depth=L, device=device)
        neg_flat[eq] = rand2
    return neg_flat.view(B, K, L)

def main():
    args = parse_args()
    os.makedirs(args.model_dir, exist_ok=True)
    set_seed(args.seed)
    device = torch.device(args.device)

    if isinstance(args.gt_gate, str) and args.gt_gate.strip() != "":
        if re.fullmatch(r"[-+]?\d+(\.\d+)?([eE][-+]?\d+)?", args.gt_gate.strip()):
            raise ValueError(
                f"--gt_gate should be a column name (e.g., is_click), not a numeric value. Got: {args.gt_gate!r}. "
                "Fix: set --gt_gate is_click (same as your S-DPO script), or omit --gt_gate to disable gating."
            )

    log_paths = [x.strip() for x in args.log_paths.split(",") if x.strip()]
    full_dataset = KRPureValueDataset(
        log_paths=log_paths,
        sid_mapping_path=args.sid_mapping_path,
        user_feat_path=args.user_feat_path,
        sid_cols=[f"sid_{i+1}" for i in range(args.sid_depth)],
        label_col=args.label_col,
        gt_spec=args.gt_spec,
        gt_gate=args.gt_gate,
        hist_from_gt=args.hist_from_gt,
        gamma=float(args.gamma),
        min_hist_len=int(args.min_hist_len),
        max_hist_len=int(args.max_hist_len),
    )

    uid_last_idx = {}
    user_ids = full_dataset.sample_user_ids  # torch tensor [N]
    for idx in range(len(full_dataset)):
        uid = int(user_ids[idx].item())      # item() -> python int
        uid_last_idx[uid] = idx

    val_indices = sorted(uid_last_idx.values())
    val_set = set(val_indices)
    train_indices = [i for i in range(len(full_dataset)) if i not in val_set]

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
        num_workers=max(0, args.num_workers // 2),
        pin_memory=True,
        drop_last=False,
    )

    trie_mask, trie_next = build_trie_from_sid_mapping(
        sid_mapping_path=args.sid_mapping_path,
        sid_depth=args.sid_depth,
        num_classes=args.num_classes,
        device=device,
    )
    valid_tokens = build_valid_tokens_per_node(trie_mask)

    hid_dim, default_head = model_dim_from_size(args.model_size)
    nhead = args.nhead if args.nhead > 0 else default_head
    args.nhead = int(nhead)
    args.hid_dim = int(hid_dim)
    if hid_dim % nhead != 0:
        raise ValueError(f"hid_dim={hid_dim} must be divisible by nhead={nhead}.")

    model = OneRecWithValue(
        num_decoder_block=args.num_layers,
        hid_dim=hid_dim,
        nhead=nhead,
        sid_depth=args.sid_depth,
        num_classes=args.num_classes,
        user_feat_dim=full_dataset.user_feat_dim,
        max_hist_len=args.max_hist_len_model,
        hist_num_layers = 2,
        value_layers=2,
        detach_value_dec_feats=True,
        use_decoder_ctx=args.use_decoder_ctx,
    ).to(device)

    if args.init_ckpt:
        ckpt = torch.load(args.init_ckpt, map_location="cpu")
        sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        safe_load_state_dict(model, sd, strict=False)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    topk_list = [int(x) for x in args.topk.split(",") if x.strip()]

    step = 0
    best = -1.0

    metrics = evaluate_hit_ndcg(model, val_loader, device, trie_mask, trie_next, args.eval_beam, topk_list)
    print("[Val][init]", " ".join([f"{k}={v:.4f}" for k, v in metrics.items()]))

    for ep in range(1, args.epochs + 1):
        print(f"\n===== [Epoch {ep}/{args.epochs}] =====")

        for it in range(1, args.sp_iters + 1):
            print(f"\n--- [SPRec] iter {it}/{args.sp_iters} ---")

            ref_model = deepcopy(model).to(device)
            ref_model.eval()
            for p in ref_model.parameters():
                p.requires_grad_(False)

            model.train()
            sft_steps = int(args.sft_steps) if int(args.sft_steps) > 0 else len(train_loader)
            for b_idx, batch in enumerate(train_loader):
                if b_idx >= sft_steps:
                    break
                step += 1

                target_sid, user_feat, hist_sid, hist_len, sample_w, _ = batch
                target_sid = target_sid.to(device).long()
                user_feat = user_feat.to(device).float()
                hist_sid = hist_sid.to(device).long()
                hist_len = hist_len.to(device).long()
                sample_w = sample_w.to(device).float().view(-1)
                sample_w = torch.clamp(sample_w, min=0.0)
                if args.gt_weight_norm:
                    m = sample_w.mean().clamp(min=1e-6)
                    sample_w = sample_w / m

                sft_loss = ntp_loss_on_gt(model, target_sid, user_feat, hist_sid, hist_len, sample_w=sample_w)
                loss = float(args.sft_coef) * sft_loss

                optim.zero_grad(set_to_none=True)
                loss.backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optim.step()

                if step % args.log_every == 0:
                    print(f"[SFT][ep={ep} it={it} step={step}] loss={sft_loss.item():.6f}")

            model.train()
            dpo_steps = int(args.dpo_steps) if int(args.dpo_steps) > 0 else len(train_loader)
            for b_idx, batch in enumerate(train_loader):
                if b_idx >= dpo_steps:
                    break
                step += 1

                target_sid, user_feat, hist_sid, hist_len, sample_w, _ = batch
                target_sid = target_sid.to(device).long()
                user_feat = user_feat.to(device).float()
                hist_sid = hist_sid.to(device).long()
                hist_len = hist_len.to(device).long()

                sample_w = sample_w.to(device).float().view(-1)
                sample_w = torch.clamp(sample_w, min=0.0)
                if args.gt_weight_norm:
                    m = sample_w.mean().clamp(min=1e-6)
                    sample_w = sample_w / m
                denom = sample_w.sum().clamp(min=1e-6)

                B, L = target_sid.shape
                V = int(args.num_classes)
                K = int(args.num_neg)
                beam_w = max(int(args.neg_beam), K + 1)

                with temporary_mode(ref_model, train=False), torch.no_grad():
                    enc_out_s, enc_mask_s, ctx_s = encode_prompt(ref_model, user_feat, hist_sid, hist_len)
                    beam_seq, _ = constrained_beam_search_sid(
                        model=ref_model,
                        enc_out=enc_out_s,
                        enc_mask=enc_mask_s,
                        sid_depth=L,
                        num_classes=V,
                        trie_mask=trie_mask,
                        trie_next=trie_next,
                        beam_width=beam_w,
                        ctx=ctx_s,
                    )
                    neg_sid = pick_negatives_from_beam(
                        target_sid, beam_seq, k=K,
                        neg_pick=args.neg_pick,
                        rank_low=args.neg_rank_low,
                        rank_high=args.neg_rank_high,
                    )
                    neg_sid = apply_random_negative_replacement(
                        neg_sid, target_sid, rand_ratio=float(args.rand_neg_ratio),
                        valid_tokens=valid_tokens, trie_next=trie_next
                    )

                seq_all = torch.cat([target_sid.unsqueeze(1), neg_sid], dim=1)  # [B,1+K,L]
                BW = B * (1 + K)
                seq_flat = seq_all.reshape(BW, L)

                enc_out_pi, enc_mask_pi, ctx_pi = encode_prompt(model, user_feat, hist_sid, hist_len)
                enc_flat_pi = enc_out_pi.unsqueeze(1).expand(B, 1 + K, -1, -1).reshape(BW, enc_out_pi.size(1), enc_out_pi.size(2))
                msk_flat_pi = enc_mask_pi.unsqueeze(1).expand(B, 1 + K, -1).reshape(BW, enc_mask_pi.size(1))
                ctx_flat_pi = None
                if ctx_pi is not None:
                    ctx_flat_pi = ctx_pi.unsqueeze(1).expand(B, 1 + K, -1).reshape(BW, -1)
                logp_pi_sum, logp_pi_tok = compute_seq_logprobs(model, enc_flat_pi, msk_flat_pi, seq_flat, trie_mask, trie_next, ctx=ctx_flat_pi)

                with torch.no_grad():
                    enc_out_rf, enc_mask_rf, ctx_rf = encode_prompt(ref_model, user_feat, hist_sid, hist_len)
                    enc_flat_rf = enc_out_rf.unsqueeze(1).expand(B, 1 + K, -1, -1).reshape(BW, enc_out_rf.size(1), enc_out_rf.size(2))
                    msk_flat_rf = enc_mask_rf.unsqueeze(1).expand(B, 1 + K, -1).reshape(BW, enc_mask_rf.size(1))
                    ctx_flat_rf = None
                    if ctx_rf is not None:
                        ctx_flat_rf = ctx_rf.unsqueeze(1).expand(B, 1 + K, -1).reshape(BW, -1)
                    logp_ref_sum, logp_ref_tok = compute_seq_logprobs(ref_model, enc_flat_rf, msk_flat_rf, seq_flat, trie_mask, trie_next, ctx=ctx_flat_rf)

                if int(args.dpo_last_n) > 0:
                    n = int(args.dpo_last_n)
                    logp_pi_used = logp_pi_tok[:, -n:].sum(dim=1)
                    logp_ref_used = logp_ref_tok[:, -n:].sum(dim=1)
                else:
                    logp_pi_used = logp_pi_sum
                    logp_ref_used = logp_ref_sum

                logp_pi = logp_pi_used.view(B, 1 + K)
                logp_ref = logp_ref_used.view(B, 1 + K)

                logp_pi_ch = logp_pi[:, 0]
                logp_pi_rj = logp_pi[:, 1:]
                logp_ref_ch = logp_ref[:, 0]
                logp_ref_rj = logp_ref[:, 1:]

                per_sample_dpo, dpo_stats = softmax_dpo_loss(
                    logp_pi_ch, logp_pi_rj,
                    logp_ref_ch, logp_ref_rj,
                    beta=float(args.beta)
                )
                dpo_loss = (per_sample_dpo * sample_w).sum() / denom
                loss = float(args.dpo_coef) * dpo_loss

                optim.zero_grad(set_to_none=True)
                loss.backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optim.step()

                if step % args.log_every == 0:
                    msg = " ".join([f"{k}={v:.4f}" for k, v in dpo_stats.items()])
                    print(f"[DPO][ep={ep} it={it} step={step}] loss={dpo_loss.item():.6f} {msg}")

            del ref_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if int(args.eval_every_iter) > 0 and (it % int(args.eval_every_iter) == 0):
                metrics = evaluate_hit_ndcg(model, val_loader, device, trie_mask, trie_next, args.eval_beam, topk_list)
                print("[Val]", " ".join([f"{k}={v:.4f}" for k, v in metrics.items()]))

                score = metrics.get("hit@20", None)
                if score is None:
                    score = list(metrics.values())[-1]
                if score > best:
                    best = float(score)
                    save_path = os.path.join(args.model_dir, "best.pt")
                    torch.save({"model": model.state_dict(), "args": vars(args), "step": step, "epoch": ep, "sp_iter": it}, save_path)
                    print(f"[Save] best -> {save_path} (score={best:.4f})")

        save_path = os.path.join(args.model_dir, f"epoch_{ep}.pt")
        torch.save({"model": model.state_dict(), "args": vars(args), "step": step, "epoch": ep}, save_path)
        print(f"[Save] {save_path}")

    print("[Done]")


if __name__ == "__main__":
    main()
