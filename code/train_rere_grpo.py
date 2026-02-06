# -*- coding: utf-8 -*-
"""ReRe-style GRPO/PPO-like post-train for OneRec SID generation on offline logs.

Key fixes vs your current train_rere_grpo.py (v3):
5) Old-policy snapshot update happens AFTER optimizer step (avoids ratio==1 on snapshot steps).
6) Per-token normalization for ratio/approx_kl/KL (stabilizes clip_frac).

Original fixes:
1) Safe checkpoint loading: skips shape-mismatched tensors instead of crashing.
2) Optional old policy snapshot (old_model) for ratio denom.
3) Dropout control for policy ratio: default uses eval-mode (dropout off) for beam/logp_old/logp_new.
4) Explicit policy_coef so you can truly turn RL on/off (policy term), independent of KL / SFT.
"""

import argparse
import os
import time
from copy import deepcopy
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

try:
    from dataset_krpure_value_gt import KRPureValueDataset  # patched version (multi-behavior GT weights)
except Exception:
    from dataset_krpure_value import KRPureValueDataset
from model.onerec_value import OneRecWithValue


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def model_dim_from_size(model_size: str) -> Tuple[int, int]:
    model_size = model_size.lower()
    if model_size in ["mini", "s"]:
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


def safe_load_state_dict(model: torch.nn.Module, state_dict: dict, *, verbose: bool = True):
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
    model.load_state_dict(filtered, strict=False)

    if verbose:
        print(f"[SafeLoad] loaded={len(filtered)} missing={len(missing)} unexpected={len(unexpected)} mismatched={len(mismatched)}")
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
            valid = (~enc_mask).float().unsqueeze(-1)  # [B,S,1]
            denom = valid.sum(dim=1).clamp_min(1.0)
            hist_vec = (enc_out * valid).sum(dim=1) / denom  # [B,D]
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
    logp_tok = log_probs.gather(-1, seq.unsqueeze(-1)).squeeze(-1)
    logp_sum = logp_tok.sum(dim=1)
    return logp_sum, logp_tok


def ntp_loss_on_gt(
    model: OneRecWithValue,
    target_sid: torch.Tensor,
    user_feat: torch.Tensor,
    hist_sid: torch.Tensor,
    hist_len: torch.Tensor,
    sample_w: Optional[torch.Tensor] = None,  # [B] or [B,1]
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
    per_sample = ce.mean(dim=1)  # [B]
    if sample_w is None:
        return per_sample.mean()
    w = sample_w.float().view(B)
    denom = w.sum().clamp(min=1e-6)
    return (per_sample * w).sum() / denom


def compute_rere_rewards(
    seq_all: torch.Tensor,          # (B, W, L)
    target_sid: torch.Tensor,       # (B, L)
    gen_logp: Optional[torch.Tensor] = None,  # (B, W) log P_\pi(e_k | prompt), used to compute rank rho_k
    w_rank: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Paper-faithful ReRe reward (Eq. 7-9 in 'Reinforced Preference Optimization for Recommendation').

    - Rule-based reward:
        R_rule(e_k, e_t) = 1 if e_k == e_t else 0

    - Ranking reward:
        \hat{R}_rank(e_k, e_t) = 0 if e_k == e_t else -1 / log(rho_k + 1)
        R_rank(e_k, e_t) = - \hat{R}_rank(e_k, e_t) / sum_j \hat{R}_rank(e_j, e_t)

      where rho_k is the *generation rank* (1 = highest probability) among the W generated items.

    - Overall:
        R = R_rule + R_rank
      (we keep an optional scalar w_rank, default 1.0, for ablations; set w_rank=1 to match the paper.)
    """
    assert seq_all.dim() == 3, f"seq_all should be (B,W,L), got {tuple(seq_all.shape)}"
    B, W, L = seq_all.shape
    assert target_sid.shape == (B, L), f"target_sid should be (B,L), got {tuple(target_sid.shape)}"

    eq = (seq_all == target_sid.unsqueeze(1)).all(dim=-1)  # (B,W) bool
    r_rule = eq.to(torch.float32)
    hit_any = eq.any(dim=1, keepdim=True)  # (B,1)

    if gen_logp is None:
        rho = torch.arange(1, W + 1, device=seq_all.device, dtype=torch.float32).unsqueeze(0).expand(B, -1)
    else:
        assert gen_logp.shape == (B, W), f"gen_logp should be (B,W), got {tuple(gen_logp.shape)}"
        order = torch.argsort(gen_logp, dim=1, descending=True)              # (B,W)
        inv_rank = torch.empty_like(order)
        inv_rank.scatter_(1, order, torch.arange(W, device=seq_all.device).unsqueeze(0).expand(B, -1))
        rho = (inv_rank + 1).to(torch.float32)                               # (B,W), 1..W

    rhat = torch.zeros((B, W), device=seq_all.device, dtype=torch.float32)
    neg_mask = ~eq
    if neg_mask.any():
        rhat[neg_mask] = -1.0 / torch.log(rho[neg_mask] + 1.0)

    denom = rhat.sum(dim=1, keepdim=True)  # (B,1) typically negative
    safe = denom.abs() >= eps
    r_rank = torch.zeros_like(rhat)
    r_rank[safe.expand_as(rhat)] = (-rhat / denom.clamp(min=-1e9, max=1e9))[safe.expand_as(rhat)]

    r_rank = r_rank * hit_any.to(torch.float32)

    return r_rule + (w_rank * r_rank)



@torch.no_grad()
def update_reference_policy(ref_model: torch.nn.Module, model: torch.nn.Module, *, mode: str = "ema", tau: float = 0.05) -> None:
    """
    Dynamically update the reference policy used for KL penalty (Appendix B.3 mentions following Gorbatovski et al., 2024).

    - mode="copy": hard copy current policy -> reference
    - mode="ema" : exponential moving average in weight space: ref <- (1-tau)*ref + tau*model
    """
    if mode == "none" or ref_model is None:
        return
    if mode == "copy":
        ref_model.load_state_dict(model.state_dict(), strict=True)
        return
    if mode != "ema":
        raise ValueError(f"Unknown ref_update_mode: {mode}")

    sd_ref = ref_model.state_dict()
    sd = model.state_dict()
    for k, v_ref in sd_ref.items():
        v = sd[k]
        if torch.is_floating_point(v_ref):
            v_ref.mul_(1.0 - float(tau)).add_(v, alpha=float(tau))
        else:
            v_ref.copy_(v)

def grpo_step(
    model: OneRecWithValue,
    old_model: Optional[OneRecWithValue],
    ref_model: Optional[OneRecWithValue],
    batch,
    device: torch.device,
    trie_mask: torch.Tensor,
    trie_next: torch.Tensor,
    group_size: int,
    clip_eps: float,
    policy_coef: float,
    kl_coef: float,
    w_rank: float,
    sft_coef: float,
    gt_weight_norm: int,
    dropout_in_policy: int,
    add_gt: int = 0,
    skip_nohit: int = 0,
    kl_estimator: str = 'exp',
):
    target_sid, user_feat, hist_sid, hist_len, sample_w, _ = batch
    target_sid = target_sid.to(device).long()
    user_feat = user_feat.to(device).float()
    hist_sid = hist_sid.to(device).long()
    hist_len = hist_len.to(device).long()

    sample_w = sample_w.to(device).float().view(-1)
    sample_w = torch.clamp(sample_w, min=0.0)
    if gt_weight_norm:
        m = sample_w.mean().clamp(min=1e-6)
        sample_w = sample_w / m

    B, L = target_sid.shape
    W = max(2, group_size)
    beam_w = W
    V = model.sid_embedding.num_embeddings

    sampler = old_model if old_model is not None else model
    policy_train_mode = bool(dropout_in_policy)

    with temporary_mode(sampler, train=policy_train_mode), torch.no_grad():
        enc_out_old, enc_mask_old, ctx_old = encode_prompt(sampler, user_feat, hist_sid, hist_len)
        gen_seq, _ = constrained_beam_search_sid(
            model=sampler,
            enc_out=enc_out_old,
            enc_mask=enc_mask_old,
            sid_depth=L,
            num_classes=V,
            trie_mask=trie_mask,
            trie_next=trie_next,
            beam_width=beam_w,
            ctx=ctx_old,
        )

    seq_all = gen_seq

    if add_gt:
        eq0 = (seq_all == target_sid.unsqueeze(1)).all(dim=-1)  # (B,W)
        need = ~eq0.any(dim=1)  # (B,)
        if need.any():
            seq_all[need, -1, :] = target_sid[need]
    BW = B * W
    seq_flat = seq_all.reshape(BW, L)

    with temporary_mode(sampler, train=policy_train_mode), torch.no_grad():
        enc_out_old, enc_mask_old, ctx_old = encode_prompt(sampler, user_feat, hist_sid, hist_len)
        enc_flat_old = enc_out_old.unsqueeze(1).expand(B, W, -1, -1).reshape(BW, enc_out_old.size(1), enc_out_old.size(2))
        msk_flat_old = enc_mask_old.unsqueeze(1).expand(B, W, -1).reshape(BW, enc_mask_old.size(1))
        ctx_flat_old = None
        if ctx_old is not None:
            ctx_flat_old = ctx_old.unsqueeze(1).expand(B, W, -1).reshape(BW, -1)
        logp_old_flat, _ = compute_seq_logprobs(sampler, enc_flat_old, msk_flat_old, seq_flat, trie_mask, trie_next, ctx=ctx_flat_old)

    logp_old_all = logp_old_flat.view(B, W).detach()

    r = compute_rere_rewards(seq_all, target_sid, logp_old_all, w_rank=w_rank)
    r_mean = r.mean(dim=1, keepdim=True)
    r_std = r.std(dim=1, keepdim=True)
    adv = (r - r_mean) / (r_std + 1e-6)
    adv = torch.where(r_std < 1e-6, torch.zeros_like(adv), adv).detach()
    eq_in_group = (seq_all == target_sid.unsqueeze(1)).all(dim=-1)  # (B,W)
    hit_any = eq_in_group.any(dim=1)  # (B,)
    nohit_frac = (~hit_any).float().mean().detach()
    hit_any_rate = hit_any.float().mean().detach()
    if skip_nohit:
        adv = adv * hit_any.float().unsqueeze(1)  # zero-out no-hit groups
    adv_flat = adv.reshape(BW)

    with temporary_mode(model, train=policy_train_mode):
        enc_out_new, enc_mask_new, ctx_new = encode_prompt(model, user_feat, hist_sid, hist_len)
        enc_flat_new = enc_out_new.unsqueeze(1).expand(B, W, -1, -1).reshape(BW, enc_out_new.size(1), enc_out_new.size(2))
        msk_flat_new = enc_mask_new.unsqueeze(1).expand(B, W, -1).reshape(BW, enc_mask_new.size(1))
        ctx_flat_new = None
        if ctx_new is not None:
            ctx_flat_new = ctx_new.unsqueeze(1).expand(B, W, -1).reshape(BW, -1)
        logp_new_flat, logp_tok_new_flat = compute_seq_logprobs(model, enc_flat_new, msk_flat_new, seq_flat, trie_mask, trie_next, ctx=ctx_flat_new)

    w_flat = sample_w.unsqueeze(1).expand(B, W).reshape(BW)  # (BW,)
    denom_w = w_flat.sum().clamp(min=1e-6)

    score_tok = torch.exp(logp_tok_new_flat - logp_tok_new_flat.detach())  # (BW,L)
    seq_score = (score_tok * adv_flat.unsqueeze(1)).mean(dim=1)            # (BW,)
    policy_loss_raw = -((seq_score * w_flat).sum() / denom_w)
    policy_loss = policy_coef * policy_loss_raw

    with torch.no_grad():
        ratio = torch.tensor(1.0, device=device)  # for logging compatibility
        clip_frac = torch.tensor(0.0, device=device)
        approx_kl = ((logp_old_flat - logp_new_flat).mean() / float(L)).clamp_min(0.0)

    kl_ref = torch.tensor(0.0, device=device)
    kl = torch.tensor(0.0, device=device)
    if (ref_model is not None) and (kl_coef > 0):
        with temporary_mode(ref_model, train=False), torch.no_grad():
            ref_enc_out, ref_enc_mask, ref_ctx = encode_prompt(ref_model, user_feat, hist_sid, hist_len)
            ref_enc_flat = ref_enc_out.unsqueeze(1).expand(B, W, -1, -1).reshape(BW, ref_enc_out.size(1), ref_enc_out.size(2))
            ref_msk_flat = ref_enc_mask.unsqueeze(1).expand(B, W, -1).reshape(BW, ref_enc_mask.size(1))
            ref_ctx_flat = None
            if ref_ctx is not None:
                ref_ctx_flat = ref_ctx.unsqueeze(1).expand(B, W, -1).reshape(BW, -1)
            _, logp_tok_ref_flat = compute_seq_logprobs(ref_model, ref_enc_flat, ref_msk_flat, seq_flat, trie_mask, trie_next, ctx=ref_ctx_flat)

        delta_ref = (logp_tok_ref_flat - logp_tok_new_flat)  # (BW, L)
        per_tok_kl = torch.exp(delta_ref) - delta_ref - 1.0
        seq_kl = per_tok_kl.mean(dim=1)  # (BW,)
        kl_ref = (seq_kl * w_flat).sum() / denom_w
        kl = kl_ref
    else:
        kl = approx_kl.detach()

    kl_loss = kl_coef * kl

    sft = torch.tensor(0.0, device=device)
    if sft_coef > 0:
        sft = ntp_loss_on_gt(model, target_sid, user_feat, hist_sid, hist_len, sample_w=sample_w)

    total = policy_loss + kl_loss + sft_coef * sft

    stats = {
        "loss": float(total.detach().cpu()),
        "policy": float(policy_loss.detach().cpu()),
        "policy_raw": float(policy_loss_raw.detach().cpu()),
        "kl": float(kl.detach().cpu()),
        "kl_ref": float(kl_ref.detach().cpu()),
        "sft": float(sft.detach().cpu()),
        "r_mean": float(r.mean().detach().cpu()),
        "hit_rate_in_group": float((r > 0.5).float().mean().detach().cpu()),
        "nohit_frac": float(nohit_frac.cpu()),
        "hit_any_rate": float(hit_any_rate.cpu()),
        "ratio": float(ratio.mean().detach().cpu()),
        "clip_frac": float(clip_frac.detach().cpu()),
        "approx_kl": float(approx_kl.detach().cpu()),
    }
    return total, stats


@torch.no_grad()
def evaluate_hit_ndcg(
    model: OneRecWithValue,
    data_loader: DataLoader,
    device: torch.device,
    trie_mask: torch.Tensor,
    trie_next: torch.Tensor,
    beam_width: int,
    topk_list: List[int],
    gt_weight_norm: int,
):
    model.eval()
    V = model.sid_embedding.num_embeddings

    hits = {k: 0.0 for k in topk_list}
    ndcgs = {k: 0.0 for k in topk_list}
    total = 0

    for batch in data_loader:
        target_sid, user_feat, hist_sid, hist_len, sample_w, _ = batch
        target_sid = target_sid.to(device).long()
        user_feat = user_feat.to(device).float()
        hist_sid = hist_sid.to(device).long()
        hist_len = hist_len.to(device).long()

        sample_w = sample_w.to(device).float().view(-1)
        sample_w = torch.clamp(sample_w, min=0.0)
        if gt_weight_norm:
            m = sample_w.mean().clamp(min=1e-6)
            sample_w = sample_w / m

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


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--log_paths", type=str, required=True)
    p.add_argument("--sid_mapping_path", type=str, required=True)
    p.add_argument("--user_feat_path", type=str, required=True)
    p.add_argument("--label_col", type=str, default="is_click")
    p.add_argument("--gt_spec", type=str, default="", help="Behavior weights spec for GT, e.g. 'is_click:1,long_view:2,is_like:5' (weight=0 ignored)")
    p.add_argument("--gt_gate", type=str, default="", help="Optional GT gate behavior, e.g. 'is_click' -> only gate==1 rows can be GT")
    p.add_argument("--hist_from_gt", type=int, default=0, help="1: history only uses GT events; 0: history uses all exposures (default)")
    p.add_argument("--gt_weight_norm", type=int, default=1, help="1: normalize sample weights to mean=1 inside each batch (default); 0: use raw weights")
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

    p.add_argument("--group_size", type=int, default=16)
    p.add_argument("--add_gt", type=int, default=0, help="Force include GT path into each group (replaces last candidate if missing).")
    p.add_argument("--clip_eps", type=float, default=0.2)
    p.add_argument("--policy_coef", type=float, default=1.0)
    p.add_argument("--kl_coef", type=float, default=0.0)
    p.add_argument("--ref_update_mode", type=str, default="ema",
                   choices=["none", "copy", "ema"],
                   help="Dynamic reference-policy update for KL term (paper Appendix B.3 Eq.21 idea). "
                        "'ema' updates ref weights as EMA of current policy; 'copy' hard-copies.")
    p.add_argument("--ref_update_tau", type=float, default=0.05,
                   help="EMA coefficient for ref update (only used when ref_update_mode='ema').")
    p.add_argument("--ref_update_every", type=int, default=1,
                   help="Update reference model every N optimizer steps (only if kl_coef>0 and ref_update_mode!='none').")
    p.add_argument("--w_rank", type=float, default=1.0)
    p.add_argument("--sft_coef", type=float, default=0.0)
    p.add_argument("--dropout_in_policy", type=int, default=0, help="0: dropout OFF for beam/logp; 1: keep dropout ON")

    p.add_argument("--use_old_model", type=int, default=1)
    p.add_argument("--old_model_update_every", type=int, default=1)
    p.add_argument("--skip_nohit", type=int, default=0, help="If 1, skip GRPO updates on groups where GT is absent (no hit).")
    p.add_argument("--kl_estimator", type=str, default="exp", choices=["exp","mean_log_ratio"], help="How to estimate KL(pi||ref) from samples. exp is non-negative (recommended).")

    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--eval_every", type=int, default=500)
    p.add_argument("--model_dir", type=str, default="checkpoints_rere_grpo")

    p.add_argument("--eval_beam", type=int, default=50)
    p.add_argument("--topk", type=str, default="1,5,10,20,50")

    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.model_dir, exist_ok=True)
    set_seed(args.seed)
    device = torch.device(args.device)

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
        gamma=args.gamma,
        min_hist_len=args.min_hist_len,
        max_hist_len=args.max_hist_len,
    )

    num_samples = len(full_dataset)
    sample_uids = full_dataset.sample_user_ids.numpy()
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
    if nhead <= 0:
        raise ValueError(f"Invalid nhead={nhead}. Use --nhead > 0 or a valid model_size.")
    if hid_dim % nhead != 0:
        raise ValueError(f"hid_dim={hid_dim} must be divisible by nhead={nhead}.")
    
    args.nhead = int(nhead)        # : , ckpt  -1 
    args.hid_dim = int(hid_dim)    # :  eval/debug(eval )
    model = OneRecWithValue(
        num_decoder_block=args.num_layers,
        hid_dim=hid_dim,
        nhead=nhead,
        sid_depth=args.sid_depth,
        num_classes=args.num_classes,
        user_feat_dim=full_dataset.user_feat_dim,
        max_hist_len=args.max_hist_len_model,
        value_layers=2,
        detach_value_dec_feats=True,
        use_decoder_ctx=args.use_decoder_ctx,
    ).to(device)

    if args.use_decoder_ctx and (not hasattr(model, 'build_ctx')):
        print('[WARN] --use_decoder_ctx was set, but model has no build_ctx(). Auto-disabling decoder ctx to match checkpoint behavior.')
        model.use_decoder_ctx = False
        args.use_decoder_ctx = False

    if args.init_ckpt:
        ckpt = torch.load(args.init_ckpt, map_location="cpu")
        sd = ckpt["model"] if (isinstance(ckpt, dict) and "model" in ckpt) else ckpt
        safe_load_state_dict(model, sd, verbose=True)
        print(f"[Load] init_ckpt={args.init_ckpt}")

    ref_model = deepcopy(model).eval().requires_grad_(False)

    old_model = None
    if args.use_old_model:
        old_model = deepcopy(model).eval().requires_grad_(False)

    trie_mask, trie_next = build_trie_from_sid_mapping(
        args.sid_mapping_path, args.sid_depth, args.num_classes, device
    )

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    topk_list = [int(x) for x in args.topk.split(",") if x.strip()]

    step = 0
    best = -1.0
    t0 = time.time()

    for ep in range(1, args.epochs + 1):
        model.train()
        for batch in train_loader:
            if step == 0:
                metrics = evaluate_hit_ndcg(
                    model=model,
                    data_loader=val_loader,
                    device=device,
                    trie_mask=trie_mask,
                    trie_next=trie_next,
                    beam_width=args.eval_beam,
                    topk_list=topk_list,
                    gt_weight_norm=args.gt_weight_norm,
                )
                model.train()
                print("[Val]", " ".join([f"{k}={v:.4f}" for k, v in metrics.items()]))
                
            step += 1
            update_old_snapshot = (old_model is not None) and (step % max(1, args.old_model_update_every) == 0)

            optim.zero_grad(set_to_none=True)
            loss, stats = grpo_step(
                model=model,
                old_model=old_model,
                ref_model=ref_model,
                batch=batch,
                device=device,
                trie_mask=trie_mask,
                trie_next=trie_next,
                group_size=args.group_size,
                clip_eps=args.clip_eps,
                policy_coef=args.policy_coef,
                kl_coef=args.kl_coef,
                w_rank=args.w_rank,
                sft_coef=args.sft_coef,
                dropout_in_policy=args.dropout_in_policy,
                add_gt=args.add_gt,
                gt_weight_norm=args.gt_weight_norm,
            )
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optim.step()

            if (ref_model is not None) and (args.kl_coef > 0) and (args.ref_update_mode != "none"):
                if step % max(1, args.ref_update_every) == 0:
                    update_reference_policy(ref_model, model, mode=args.ref_update_mode, tau=args.ref_update_tau)

            if update_old_snapshot:
                old_model.load_state_dict(model.state_dict(), strict=True)

            if step % args.log_every == 0:
                dt = time.time() - t0
                msg = " ".join([f"{k}={v:.4f}" for k, v in stats.items()])
                print(f"[Train][ep={ep} step={step}] {msg} | dt={dt:.1f}s")
                t0 = time.time()

            if step % args.eval_every == 0:
                metrics = evaluate_hit_ndcg(
                    model=model,
                    data_loader=val_loader,
                    device=device,
                    trie_mask=trie_mask,
                    trie_next=trie_next,
                    beam_width=args.eval_beam,
                    topk_list=topk_list,
                    gt_weight_norm=args.gt_weight_norm,
                )
                model.train()
                print("[Val]", " ".join([f"{k}={v:.4f}" for k, v in metrics.items()]))
                score = metrics.get("hit@20", 0.0)
                if score > best:
                    best = score
                    save_path = os.path.join(args.model_dir, "best.pt")
                    torch.save({"model": model.state_dict(), "args": vars(args), "step": step}, save_path)
                    print(f"[Save] best -> {save_path} (hit@20={best:.4f})")

        save_path = os.path.join(args.model_dir, f"epoch_{ep}.pt")
        torch.save({"model": model.state_dict(), "args": vars(args), "step": step}, save_path)
        print(f"[Save] {save_path}")

    print("[Done]")


if __name__ == "__main__":
    main()