# -*- coding: utf-8 -*-
"""HRPO-RRPO on NTP (no rev/ltv heads)

What this script does
---------------------
* Offline, log-based post-train for OneRec-style hierarchical SID generation.
* Reward is **dense per-token HRPO residuals** from a pre-built `hrpo_table.pkl`.
* RL update is **token-level PPO/GRPO** (one ratio/clip per token position).
* Optional KL-to-reference (SFT checkpoint) and small SFT mix to prevent degeneration.

Key design choices
------------------
* We keep an `old_model` (behavior policy) frozen for sampling + logp_old.
  We update `old_model <- model` every `--old_update_freq` steps (default 20).
* We disable dropout during RL forward (set `model.eval()`), because PPO ratios are
  otherwise dominated by stochasticity.
* We **freeze value heads** by default (`--freeze_value_decoder=1`) because we are
  not using rev/ltv for this training target.

Expected repo layout
--------------------
This script is meant to live under your repo `code/` and reuse:
  - model/onerec_value.py  (OneRecWithValue)
  - dataset_krpure_value.py (KRPureValueDataset)

If you run it elsewhere, adjust imports accordingly.
"""

from __future__ import annotations

import argparse
import os
import pickle
import random
import time
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Union, Sequence
import math
import numpy as np
import torch
import torch.nn.functional as F

def ensure_build_ctx(model):
    """Ensure `model.build_ctx` exists and returns a context vector of shape (B, D).

    Some repo versions of `model/onerec_value.py` call `self.build_ctx(...)` inside
    `forward_with_cache`, expecting (B, D) so they can do `x = x + ctx.unsqueeze(1)`
    where x is (B, sid_depth, D). However, older snapshots may (a) miss the method,
    or (b) return (B, T, D) (e.g., encoder outputs), which will crash during beam search.

    This helper adds the method if missing, and *wraps* it if present to guarantee output
    shape (B, D) by applying a masked mean pool when needed.
    """
    import types

    if getattr(model, "_build_ctx_wrapped", False):
        return model

    def _masked_mean(enc_out, enc_mask):
        if enc_mask is None:
            return enc_out.mean(dim=1)
        m = enc_mask
        if m.dtype == torch.bool:
            first_true_ratio = m[:, 0].float().mean().item()
            valid = m if first_true_ratio > 0.5 else (~m)
        else:
            valid = m > 0
        vf = valid.float()  # (B,T)
        denom = vf.sum(dim=1, keepdim=True).clamp_min(1.0)  # (B,1)
        return (enc_out * vf.unsqueeze(-1)).sum(dim=1) / denom  # (B,D)

    orig = getattr(model, "build_ctx", None)

    if orig is None:
        def _default_build_ctx(self, enc_out, enc_mask, user_vec):
            ctx = _masked_mean(enc_out, enc_mask)
            use = getattr(self, "use_decoder_ctx", True)
            if (user_vec is None) or (not use):
                return ctx
            try:
                if user_vec.dim() == 2:
                    return ctx + user_vec
                if user_vec.dim() == 3 and user_vec.size(1) == 1:
                    return ctx + user_vec.squeeze(1)
            except Exception:
                pass
            return ctx
        model.build_ctx = types.MethodType(_default_build_ctx, model)
    else:
        def _wrapped_build_ctx(self, enc_out, enc_mask, user_vec):
            ctx = orig(enc_out, enc_mask, user_vec)
            try:
                if isinstance(ctx, torch.Tensor) and ctx.dim() == 3:
                    ctx = _masked_mean(ctx, enc_mask)
                elif isinstance(ctx, torch.Tensor) and ctx.dim() == 2:
                    pass
                else:
                    ctx = _masked_mean(enc_out, enc_mask)
            except Exception:
                ctx = _masked_mean(enc_out, enc_mask)
            return ctx
        model.build_ctx = types.MethodType(_wrapped_build_ctx, model)

    model._build_ctx_wrapped = True
    return model

from torch.utils.data import DataLoader, Subset


try:
    from model.onerec_value import OneRecWithValue
except Exception:
    from onerec_value import OneRecWithValue  # type: ignore

try:
    from dataset_krpure_value import KRPureValueDataset
except Exception:
    from dataset_krpure_value import KRPureValueDataset  # type: ignore





import json
class _StepMetricsPlotter:
    """Lightweight step-based metrics recorder + plotter (matplotlib Agg).

    This is intentionally isolated so it does not change training logic.
    """

    def __init__(self, out_dir: str, smooth_window: int = 0, clear: bool = False):
        self.out_dir = out_dir
        self.smooth_window = int(max(0, smooth_window))
        self.train_path = os.path.join(out_dir, "train.jsonl")
        self.val_path = os.path.join(out_dir, "val.jsonl")
        self.plot_dir_train = os.path.join(out_dir, "plots", "train")
        self.plot_dir_val = os.path.join(out_dir, "plots", "val")
        os.makedirs(self.out_dir, exist_ok=True)
        os.makedirs(self.plot_dir_train, exist_ok=True)
        os.makedirs(self.plot_dir_val, exist_ok=True)

        if clear:
            for p in (self.train_path, self.val_path):
                try:
                    if os.path.exists(p):
                        os.remove(p)
                except Exception:
                    pass

        self._train_series = {}  # key -> list[(step, value)]
        self._val_series = {}    # key -> list[(step, value)]
        self._mpl_ok = None

    def _ensure_mpl(self):
        if self._mpl_ok is not None:
            return self._mpl_ok
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt  # noqa: F401
            self._mpl_ok = True
        except Exception:
            self._mpl_ok = False
        return self._mpl_ok

    @staticmethod
    def _to_float(v):
        try:
            return float(v)
        except Exception:
            return None

    def _append_series(self, series_dict, step: int, d: dict):
        for k, v in d.items():
            if k in ("step", "ep", "time", "dt"):
                continue
            fv = self._to_float(v)
            if fv is None or not (math.isfinite(fv)):
                continue
            series_dict.setdefault(k, []).append((int(step), float(fv)))

    def log_train(self, step: int, ep: int, stats: dict):
        rec = {"step": int(step), "ep": int(ep)}
        for k, v in stats.items():
            fv = self._to_float(v)
            rec[k] = fv if fv is not None else v
        try:
            with open(self.train_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception:
            pass
        self._append_series(self._train_series, step, stats)

    def log_val(self, step: int, ep: int, stats: dict):
        rec = {"step": int(step), "ep": int(ep)}
        for k, v in stats.items():
            fv = self._to_float(v)
            rec[k] = fv if fv is not None else v
        try:
            with open(self.val_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception:
            pass
        self._append_series(self._val_series, step, stats)

    def _smooth(self, ys):
        w = self.smooth_window
        if w <= 1 or len(ys) < w:
            return ys
        out = []
        s = 0.0
        for i, y in enumerate(ys):
            s += y
            if i >= w:
                s -= ys[i - w]
            if i >= w - 1:
                out.append(s / w)
            else:
                out.append(y)
        return out

    def _plot_series(self, series, out_path: str, title: str):
        if not self._ensure_mpl():
            return
        import matplotlib.pyplot as plt
        xs = [x for x, _ in series]
        ys = [y for _, y in series]
        ys2 = self._smooth(ys)
        plt.figure()
        plt.plot(xs, ys2)
        plt.xlabel("step")
        plt.ylabel(title)
        plt.title(title)
        plt.tight_layout()
        try:
            plt.savefig(out_path, dpi=160)
        finally:
            plt.close()

    def plot_train(self):
        for k, series in self._train_series.items():
            if len(series) < 2:
                continue
            out_path = os.path.join(self.plot_dir_train, f"{k}.png")
            self._plot_series(series, out_path, f"train/{k}")

    def plot_val(self):
        for k, series in self._val_series.items():
            if len(series) < 2:
                continue
            out_path = os.path.join(self.plot_dir_val, f"{k}.png")
            self._plot_series(series, out_path, f"val/{k}")

    def plot_all(self):
        self.plot_train()
        self.plot_val()
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def model_dim_from_size(model_size: str) -> Tuple[int, int]:
    """Return (hid_dim, default_nhead)"""
    model_size = str(model_size).lower()
    if model_size in ["mini", "small"]:
        return 128, 4
    if model_size in ["base", "medium"]:
        return 256, 8
    if model_size in ["large"]:
        return 512, 8
    return 256, 8


def build_trie_from_sid_mapping(
    sid_mapping_path: str,
    sid_depth: int,
    num_classes: int,
    device: torch.device,
):
    """Build trie constraints for hierarchical SID decoding.

    Returns:
      trie_mask: [num_nodes, num_classes] bool
      trie_next: [num_nodes, num_classes] long (next_node or -1)
    """
    import pandas as pd

    df = pd.read_csv(sid_mapping_path)
    sid_cols = [f"sid_{i+1}" for i in range(sid_depth)]
    for c in sid_cols:
        if c not in df.columns:
            raise ValueError(f"sid mapping missing column: {c}")
    paths = df[sid_cols].astype(int).values.tolist()

    trie: List[Dict[int, int]] = [dict()]
    for p in paths:
        node = 0
        for tok in p:
            tok = int(tok)
            if tok < 0 or tok >= num_classes:
                continue
            nxt = trie[node].get(tok, None)
            if nxt is None:
                nxt = len(trie)
                trie[node][tok] = nxt
                trie.append(dict())
            node = nxt

    num_nodes = len(trie)
    trie_mask = torch.zeros((num_nodes, num_classes), dtype=torch.bool, device=device)
    trie_next = torch.full((num_nodes, num_classes), -1, dtype=torch.long, device=device)
    for i, mp in enumerate(trie):
        for tok, nxt in mp.items():
            trie_mask[i, tok] = True
            trie_next[i, tok] = int(nxt)
    print(f"[Trie] Built from {len(paths)} paths, nodes={num_nodes}, vocab={num_classes}, sid_depth={sid_depth}")
    return trie_mask, trie_next


@dataclass
class HRPOTable:
    sid_depth: int
    uid2ctx: Dict[int, int]
    ctx_count: Dict[int, int]
    ctx_sum: Dict[int, float]
    prefix_count: Dict[Tuple[int, Tuple[int, ...]], int]
    prefix_sum: Dict[Tuple[int, Tuple[int, ...]], float]
    global_mean: float
    smoothing_alpha: float
    min_prefix_count: int

    @staticmethod
    def load(path: str) -> "HRPOTable":
        with open(path, "rb") as f:
            d = pickle.load(f)
        prefix_count = {(int(k[0]), tuple(k[1])): int(v) for k, v in d["prefix_count"].items()}
        prefix_sum = {(int(k[0]), tuple(k[1])): float(v) for k, v in d["prefix_sum"].items()}
        return HRPOTable(
            sid_depth=int(d["sid_depth"]),
            uid2ctx={int(k): int(v) for k, v in d["uid2ctx"].items()},
            ctx_count={int(k): int(v) for k, v in d["ctx_count"].items()},
            ctx_sum={int(k): float(v) for k, v in d["ctx_sum"].items()},
            prefix_count=prefix_count,
            prefix_sum=prefix_sum,
            global_mean=float(d["global_mean"]),
            smoothing_alpha=float(d.get("smoothing_alpha", 100.0)),
            min_prefix_count=int(d.get("min_prefix_count", 1)),
        )

    def _ctx_mean(self, ctx: int) -> float:
        cnt = self.ctx_count.get(ctx, 0)
        if cnt <= 0:
            return self.global_mean
        a = self.smoothing_alpha
        return (self.ctx_sum.get(ctx, 0.0) + a * self.global_mean) / (cnt + a)

    def _shrink_mean(self, sum_r: float, cnt: int, prior: float) -> float:
        a = self.smoothing_alpha
        return (sum_r + a * prior) / (cnt + a)

    def mean_reward(self, ctx: int, prefix: Tuple[int, ...]) -> float:
        """Empirical-Bayes mean with (ctx,prefix)->global(prefix)->shorter backoff."""
        ctx_prior = self._ctx_mean(ctx)
        p = prefix
        while True:
            if len(p) == 0:
                return ctx_prior

            key = (ctx, p)
            cnt = self.prefix_count.get(key, 0)
            if cnt > 0 and (cnt >= self.min_prefix_count or self.min_prefix_count <= 1):
                return self._shrink_mean(self.prefix_sum.get(key, 0.0), cnt, prior=ctx_prior)

            key_g = (-1, p)
            cnt_g = self.prefix_count.get(key_g, 0)
            if cnt_g > 0 and (cnt_g >= self.min_prefix_count or self.min_prefix_count <= 1):
                return self._shrink_mean(self.prefix_sum.get(key_g, 0.0), cnt_g, prior=ctx_prior)

            p = p[:-1]

    def residuals_for_path(self, ctx: int, path: Tuple[int, ...]) -> np.ndarray:
        assert len(path) == self.sid_depth
        res = np.zeros((self.sid_depth,), dtype=np.float32)
        prev = self.mean_reward(ctx, tuple())
        pref: Tuple[int, ...] = tuple()
        for i, tok in enumerate(path):
            pref = pref + (int(tok),)
            cur = self.mean_reward(ctx, pref)
            res[i] = cur - prev
            prev = cur
        return res


class KRPureValueDatasetWithUID(KRPureValueDataset):
    """Just like KRPureValueDataset, but returns user_id per sample."""

    def __getitem__(self, idx):
        base = super().__getitem__(idx)
        uid = self.sample_user_ids[idx]
        return base + (uid,)


def compute_seq_logprobs_token(
    model: OneRecWithValue,
    target_sid: torch.LongTensor,
    user_feat: torch.Tensor,
    hist_sid: torch.LongTensor,
    hist_len: torch.LongTensor,
):
    """Return (logp_sum [B], logp_tok [B,L])."""
    logits, *_ = model.forward_with_cache(
        target_sid=target_sid,
        user_feat=user_feat,
        hist_sid=hist_sid,
        hist_len=hist_len,
    )
    logp = F.log_softmax(logits, dim=-1)  # [B,L,V]
    tok = target_sid.unsqueeze(-1)
    logp_tok = logp.gather(-1, tok).squeeze(-1)  # [B,L]
    logp_sum = logp_tok.sum(dim=1)  # [B]
    return logp_sum, logp_tok



def constrained_beam_search_sid(
    model: OneRecWithValue,
    user_feat: torch.Tensor,
    hist_sid: torch.LongTensor,
    hist_len: torch.LongTensor,
    trie_mask: torch.Tensor,
    trie_next: torch.Tensor,
    beam_width: int,
    sid_depth: int,
    temperature: float = 1.0,
):
    """Beam search under trie constraints.

    Important implementation detail:
      - We start with **one** active beam, then grow up to beam_width.
        If you start with beam_width identical beams, the first step produces
        many *tied duplicates* (same token copied across parents), which
        collapses beam diversity and makes hit@k identical for all k.

    Returns:
      seq: [B, W, L] where W <= beam_width (can grow to beam_width)
      logp: [B, W]   (sum log-prob)
    """
    device = hist_sid.device
    B = hist_sid.size(0)

    if hist_sid.dtype not in (torch.int64, torch.long):
        hist_sid = hist_sid.long()
    if hist_len.dtype not in (torch.int64, torch.long):
        hist_len = hist_len.long()

    user_feat = model._pad_or_trim_user_feat(user_feat, model.user_proj.in_features, batch_size=B)
    user_vec = model.user_proj(user_feat)
    _enc_out, _enc_mask = model.encode_history_seq(hist_sid, hist_len, user_ctx=user_vec)

    beams = torch.zeros((B, 1, sid_depth), dtype=torch.long, device=device)
    beam_scores = torch.zeros((B, 1), dtype=torch.float32, device=device)
    trie_nodes = torch.zeros((B, 1), dtype=torch.long, device=device)  # root=0
    active = 1

    for t in range(sid_depth):
        flat_beams = beams.view(B * active, sid_depth)
        flat_nodes = trie_nodes.view(B * active)

        logits, *_ = model.forward_with_cache(
            target_sid=flat_beams,
            user_feat=user_feat.repeat_interleave(active, dim=0),
            hist_sid=hist_sid.repeat_interleave(active, dim=0),
            hist_len=hist_len.repeat_interleave(active, dim=0),
        )
        step_logits = logits[:, t, :] / max(1e-6, float(temperature))
        step_logp = F.log_softmax(step_logits, dim=-1)  # [B*active, V]

        valid = trie_mask[flat_nodes]  # [B*active, V]
        step_logp = step_logp.masked_fill(~valid, -1e9)

        V = int(step_logp.size(-1))
        k_step = min(int(beam_width), V)

        topk_logp, topk_tok = torch.topk(step_logp, k=k_step, dim=-1)  # [B*active, k_step]

        cand_scores = beam_scores.view(B * active, 1) + topk_logp  # [B*active, k_step]
        cand_scores = cand_scores.view(B, active * k_step)
        cand_tok = topk_tok.view(B, active * k_step)
        cand_parent = (
            torch.arange(active, device=device).view(1, active, 1).expand(B, active, k_step).reshape(B, active * k_step)
        )

        new_active = min(int(beam_width), int(active * k_step))
        new_scores, idx = torch.topk(cand_scores, k=new_active, dim=-1)
        new_parent = cand_parent.gather(1, idx)
        new_tok = cand_tok.gather(1, idx)

        beams = beams.gather(1, new_parent.unsqueeze(-1).expand(-1, -1, sid_depth))
        beams[:, :, t] = new_tok
        beam_scores = new_scores

        parent_nodes = trie_nodes.gather(1, new_parent)
        trie_nodes = trie_next[parent_nodes, new_tok]

        active = new_active

    return beams, beam_scores


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
    hits = {k: 0 for k in topk_list}
    ndcgs = {k: 0.0 for k in topk_list}
    total = 0
    uniq_sum = 0.0

    for batch in data_loader:
        target_sid, user_feat, hist_sid, hist_len, *_ = batch
        target_sid = target_sid.to(device)
        user_feat = user_feat.to(device)
        hist_sid = hist_sid.to(device)
        hist_len = hist_len.to(device)

        seq, _ = constrained_beam_search_sid(
            model=model,
            user_feat=user_feat,
            hist_sid=hist_sid,
            hist_len=hist_len,
            trie_mask=trie_mask,
            trie_next=trie_next,
            beam_width=beam_width,
            sid_depth=target_sid.size(1),
            temperature=1.0,
        )

        B = target_sid.size(0)
        total += B

        with torch.no_grad():
            for b in range(B):
                uniq = set(tuple(int(x) for x in seq[b, r].tolist()) for r in range(seq.size(1)))
                uniq_sum += len(uniq) / max(1, seq.size(1))

        for b in range(B):
            tgt = target_sid[b]
            rank = None
            for r in range(seq.size(1)):
                if torch.equal(seq[b, r], tgt):
                    rank = r
                    break
            for k in topk_list:
                if rank is not None and rank < k:
                    hits[k] += 1
                    ndcgs[k] += 1.0 / np.log2(rank + 2)

    metrics = {}
    for k in topk_list:
        metrics[f"hit@{k}"] = hits[k] / max(1, total)
        metrics[f"ndcg@{k}"] = ndcgs[k] / max(1, total)
    metrics["beam_uniq_ratio"] = uniq_sum / max(1, total)
    return metrics


def _dbg_stats(name: str, t: torch.Tensor, max_print: int = 8):
    """Lightweight tensor stats for debugging."""
    try:
        x = t.detach()
        if x.numel() == 0:
            print(f"[DBG][{name}] empty")
            return
        x = x.float().flatten().cpu()
        qs = [0, 1, 5, 25, 50, 75, 95, 99, 100]
        qv = torch.quantile(x, torch.tensor([q/100 for q in qs], dtype=torch.float32)).tolist()
        print(f"[DBG][{name}] n={x.numel()} mean={x.mean().item():.6g} std={x.std(unbiased=False).item():.6g} min={x.min().item():.6g} max={x.max().item():.6g}")
        print(f"[DBG][{name}] pct " + " ".join([f"p{q}={v:.6g}" for q,v in zip(qs,qv)]))
    except Exception as e:
        print(f"[DBG][{name}] stats failed: {type(e).__name__}: {e}")



def _dist_entropy_top1(values: torch.LongTensor, vocab: int):
    """Return (entropy, entropy_norm, top1_share, uniq) for a 1D LongTensor."""
    if values.numel() == 0 or vocab <= 0:
        return 0.0, 0.0, 0.0, 0
    v = values.detach().view(-1)
    v = v.clamp(min=0)
    counts = torch.bincount(v, minlength=vocab).float()
    total = counts.sum().clamp(min=1.0)
    p = counts / total
    mask = p > 0
    ent = float(-(p[mask] * torch.log(p[mask])).sum().item())
    ent_norm = float(ent / math.log(vocab)) if vocab > 1 else 0.0
    top1 = float(p.max().item())
    uniq = int((counts > 0).sum().item())
    return ent, ent_norm, top1, uniq

def _dbg_print_seq(prefix: str, seq: torch.Tensor, n: int = 2):
    """Print first n sequences (int tokens)."""
    try:
        x = seq.detach().cpu()
        n = min(n, x.size(0))
        for i in range(n):
            print(f"[DBG][{prefix}][{i}] {x[i].tolist()}")
    except Exception as e:
        print(f"[DBG][{prefix}] print_seq failed: {type(e).__name__}: {e}")

def ntp_loss(
    model: OneRecWithValue,
    target_sid: torch.LongTensor,
    user_feat: torch.Tensor,
    hist_sid: torch.LongTensor,
    hist_len: torch.LongTensor,
):
    logits, *_ = model.forward_with_cache(
        target_sid=target_sid,
        user_feat=user_feat,
        hist_sid=hist_sid,
        hist_len=hist_len,
    )
    B, L = target_sid.shape
    return F.cross_entropy(logits.view(B * L, -1), target_sid.view(B * L), ignore_index=-1)


def hrpo_rewards_for_candidates(
    hrpo: Union[HRPOTable, Sequence[HRPOTable]],
    uids: torch.LongTensor,
    seq_bw: torch.LongTensor,
    reward_scale: float,
    device: torch.device,
    hrpo_weights: Optional[Sequence[float]] = None,
):
    """Compute HRPO residual reward per token for sequences.

    Args:
      uids: [B]
      seq_bw: [B,W,L]
    Returns:
      r_tok: [B,W,L] float32
    """
    hrpo_list: List[HRPOTable] = list(hrpo) if isinstance(hrpo, (list, tuple)) else [hrpo]
    if hrpo_weights is None:
        wts = [1.0] * len(hrpo_list)
    else:
        wts = list(float(x) for x in hrpo_weights)
        if len(wts) != len(hrpo_list):
            raise ValueError(f"hrpo_weights length {len(wts)} != hrpo tables {len(hrpo_list)}")

    uids_np = uids.detach().cpu().numpy().astype(np.int64)
    seq_np = seq_bw.detach().cpu().numpy().astype(np.int64)
    B, W, L = seq_np.shape
    out = np.zeros((B, W, L), dtype=np.float32)

    cache: Dict[Tuple[int, int, Tuple[int, ...]], np.ndarray] = {}

    for b in range(B):
        uid = int(uids_np[b])
        for w in range(W):
            path = tuple(int(x) for x in seq_np[b, w].tolist())
            rr_sum = None
            for ti, tbl in enumerate(hrpo_list):
                ctx = int(tbl.uid2ctx.get(uid, 0))
                key = (ti, ctx, path)
                rr = cache.get(key)
                if rr is None:
                    rr = tbl.residuals_for_path(ctx, path)
                    cache[key] = rr
                wt = float(wts[ti])
                if rr_sum is None:
                    rr_sum = (wt * rr).astype(np.float32, copy=False)
                else:
                    rr_sum = rr_sum + (wt * rr)
            if rr_sum is None:
                rr_sum = np.zeros((L,), dtype=np.float32)
            out[b, w, :] = rr_sum

    out *= float(reward_scale)
    return torch.from_numpy(out).to(device=device, dtype=torch.float32)


def rrpo_step_hrpo_token(
    model: OneRecWithValue,
    old_model: OneRecWithValue,
    ref_model: Optional[OneRecWithValue],
    hrpo: Union[HRPOTable, Sequence[HRPOTable]],
    batch,
    device: torch.device,
    trie_mask: torch.Tensor,
    trie_next: torch.Tensor,
    group_size: int,
    clip_eps: float,
    kl_coef: float,
    sft_coef: float,
    reward_scale: float,
    hrpo_weights: Optional[Sequence[float]] = None,
    global_step: int = 0,
    debug_every: int = 0,
    debug_samples: int = 2,
    debug_print_seq: bool = False,
    debug_reward_range: bool = False,
    no_residual_credit: bool = False,
    no_credit_to_go: bool = False,
):
    """One PPO/GRPO update step using HRPO dense rewards."""
    target_sid, user_feat, hist_sid, hist_len, *_rest, uids = batch
    target_sid = target_sid.to(device)
    user_feat = user_feat.to(device)
    hist_sid = hist_sid.to(device)
    hist_len = hist_len.to(device)
    uids = uids.to(device)

    if target_sid.dtype not in (torch.int64, torch.long):
        target_sid = target_sid.long()
    if hist_sid.dtype not in (torch.int64, torch.long):
        hist_sid = hist_sid.long()
    if hist_len.dtype not in (torch.int64, torch.long):
        hist_len = hist_len.long()

    B, L = target_sid.shape
    W = max(2, int(group_size))
    beam_w = W - 1

    model.eval()
    old_model.eval()
    if ref_model is not None:
        ref_model.eval()

    with torch.no_grad():
        gen_seq, _ = constrained_beam_search_sid(
            model=old_model,
            user_feat=user_feat,
            hist_sid=hist_sid,
            hist_len=hist_len,
            trie_mask=trie_mask,
            trie_next=trie_next,
            beam_width=beam_w,
            sid_depth=L,
            temperature=1.0,
        )  # [B, W-1, L]

        seq_all = torch.cat([target_sid.unsqueeze(1), gen_seq], dim=1)  # [B,W,L]

        flat_seq = seq_all.view(B * W, L)
        flat_user = user_feat.unsqueeze(1).expand(B, W, user_feat.size(1)).contiguous().view(B * W, -1)
        flat_hist = hist_sid.unsqueeze(1).expand(B, W, *hist_sid.shape[1:]).contiguous().view(B * W, *hist_sid.shape[1:])
        flat_hlen = hist_len.unsqueeze(1).expand(B, W).contiguous().view(B * W)

        _, logp_old_tok = compute_seq_logprobs_token(
            old_model, flat_seq, flat_user, flat_hist, flat_hlen
        )  # [B*W, L]
        logp_old_tok = logp_old_tok.view(B, W, L)

        logp_ref_tok = None
        if ref_model is not None and kl_coef > 0:
            _, lr = compute_seq_logprobs_token(ref_model, flat_seq, flat_user, flat_hist, flat_hlen)
            logp_ref_tok = lr.view(B, W, L)

        r_tok = hrpo_rewards_for_candidates(
            hrpo,
            uids,
            seq_all,
            reward_scale,
            device,
            hrpo_weights=hrpo_weights,
        )

        if no_residual_credit:
            r_seq = r_tok.sum(dim=-1, keepdim=True)  # [B,W,1]
            base_tok = r_seq.expand(-1, -1, L).contiguous()  # [B,W,L]
        else:
            base_tok = r_tok

        if no_credit_to_go or no_residual_credit:
            rtg_tok = base_tok
        else:
            rtg_tok = torch.flip(torch.cumsum(torch.flip(base_tok, dims=[2]), dim=2), dims=[2])

        mu = rtg_tok.mean(dim=1, keepdim=True)
        std = rtg_tok.std(dim=1, keepdim=True)
        std = torch.clamp(std, min=1e-3)
        adv_tok = (rtg_tok - mu) / (std + 1e-6)
        adv_tok = adv_tok.clamp(-5.0, 5.0)

        r_sum = r_tok.sum(dim=-1)  # [B,W]
        best_idx = r_sum.argmax(dim=1)
        hit_rate_in_group = (best_idx == 0).float().mean().item()
        r_mean = r_sum.mean().item()

        best_r_sum = r_sum.max(dim=1).values  # [B]
        best_seq = seq_all[torch.arange(B, device=seq_all.device), best_idx]  # [B,L]
        vocab = None
        for _attr in ("num_classes", "vocab_size"):
            if hasattr(model, _attr):
                try:
                    vocab = int(getattr(model, _attr))
                    break
                except Exception:
                    vocab = None
        if vocab is None:
            vocab = int(getattr(trie_mask, "shape", [32])[-1]) if trie_mask is not None else 32

        sid1_gen = gen_seq[..., 0].reshape(-1)
        sid1_best = best_seq[..., 0].reshape(-1)
        ent1g, ent1g_n, top1_1g, uniq1g = _dist_entropy_top1(sid1_gen, vocab)
        ent1b, ent1b_n, top1_1b, uniq1b = _dist_entropy_top1(sid1_best, vocab)

        ent2g = ent2g_n = top1_2g = uniq2g = 0.0
        ent2b = ent2b_n = top1_2b = uniq2b = 0.0
        if L >= 2:
            sid2_gen = gen_seq[..., 1].reshape(-1)
            sid2_best = best_seq[..., 1].reshape(-1)
            ent2g, ent2g_n, top1_2g, uniq2g = _dist_entropy_top1(sid2_gen, vocab)
            ent2b, ent2b_n, top1_2b, uniq2b = _dist_entropy_top1(sid2_best, vocab)

        if (debug_every is not None) and (debug_every > 0) and (global_step % debug_every == 0):
            gt_r_sum = r_sum[:, 0]
            best_r_sum = r_sum.max(dim=1).values
            gap = best_r_sum - gt_r_sum
            _dbg_stats("r_sum_gt", gt_r_sum)
            _dbg_stats("r_sum_best", best_r_sum)
            _dbg_stats("r_gap_best_minus_gt", gap)
            adv_gt = adv_tok[:, 0, :].reshape(-1)
            _dbg_stats("adv_gt", adv_gt)
            dup = (gen_seq == target_sid.unsqueeze(1)).all(dim=-1)  # [B, W-1]
            print(f"[DBG][beam_dup_gt] frac={dup.float().mean().item():.4f}")
            print(
                f"[DBG][sid1][gen] uniq={int(uniq1g)} top1={top1_1g:.4f} H={ent1g:.3f} Hn={ent1g_n:.3f} | "
                f"[best] uniq={int(uniq1b)} top1={top1_1b:.4f} H={ent1b:.3f} Hn={ent1b_n:.3f}"
            )
            if L >= 2:
                print(
                    f"[DBG][sid2][gen] uniq={int(uniq2g)} top1={float(top1_2g):.4f} H={float(ent2g):.3f} Hn={float(ent2g_n):.3f} | "
                    f"[best] uniq={int(uniq2b)} top1={float(top1_2b):.4f} H={float(ent2b):.3f} Hn={float(ent2b_n):.3f}"
                )


    flat_seq = seq_all.view(B * W, L)
    flat_user = user_feat.unsqueeze(1).expand(B, W, user_feat.size(1)).contiguous().view(B * W, -1)
    flat_hist = hist_sid.unsqueeze(1).expand(B, W, *hist_sid.shape[1:]).contiguous().view(B * W, *hist_sid.shape[1:])
    flat_hlen = hist_len.unsqueeze(1).expand(B, W).contiguous().view(B * W)

    _, logp_new_tok = compute_seq_logprobs_token(model, flat_seq, flat_user, flat_hist, flat_hlen)
    logp_new_tok = logp_new_tok.view(B, W, L)

    ratio = torch.exp(logp_new_tok - logp_old_tok)  # [B,W,L]
    ratio_clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
    obj1 = ratio * adv_tok
    obj2 = ratio_clipped * adv_tok
    policy_obj = torch.minimum(obj1, obj2)
    policy_loss = -policy_obj.mean()

    clip_frac = ((ratio - ratio_clipped).abs() > 1e-8).float().mean().item()
    ratio_mean = ratio.mean().item()
    dlogp = (logp_new_tok - logp_old_tok).abs().mean().item()
    dratio = (ratio - 1.0).abs().mean().item()

    kl = torch.tensor(0.0, device=device)
    if kl_coef > 0:
        kl = (logp_old_tok.detach() - logp_new_tok).mean()

    sft = torch.tensor(0.0, device=device)
    if sft_coef > 0:
        sft = ntp_loss(model, target_sid, user_feat, hist_sid, hist_len)

    total_loss = policy_loss + kl_coef * kl + sft_coef * sft

    stats = {
        "loss": float(total_loss.detach().cpu()),
        "policy": float(policy_loss.detach().cpu()),
        "policy_raw": float(policy_loss.detach().cpu()),
        "kl": float(kl.detach().cpu()),
        "sft": float(sft.detach().cpu()),
        "r_mean": float(r_mean),
        "r_best_mean": float(best_r_sum.mean().item()),
        "sid1_top1_gen": float(top1_1g),
        "sid1_Hn_gen": float(ent1g_n),
        "sid1_top1_best": float(top1_1b),
        "sid1_Hn_best": float(ent1b_n),
        "sid2_top1_gen": float(top1_2g) if L >= 2 else 0.0,
        "sid2_Hn_gen": float(ent2g_n) if L >= 2 else 0.0,
        "sid2_top1_best": float(top1_2b) if L >= 2 else 0.0,
        "sid2_Hn_best": float(ent2b_n) if L >= 2 else 0.0,
        "hit_rate_in_group": float(hit_rate_in_group),
        "ratio": float(ratio_mean),
        "clip_frac": float(clip_frac),
        "dlogp": float(dlogp),
        "dratio": float(dratio),
        "approx_kl": float(kl.detach().cpu()),
    }
    if debug_every and (global_step % debug_every == 0):
        print(f"[DBG][GRPO] step={global_step} B={B} W={W} L={L} | clip_eps={clip_eps} kl_coef={kl_coef} sft_coef={sft_coef} reward_scale={reward_scale}")
        _dbg_stats("ratio", ratio)
        _dbg_stats("adv_tok", adv_tok)
        _dbg_stats("reward_tok_scaled", r_tok)
        if debug_reward_range and (reward_scale is not None) and float(reward_scale) != 0.0:
            _dbg_stats("reward_tok_raw", r_tok / float(reward_scale))
        try:
            with torch.no_grad():
                _, gt_logp_tok = compute_seq_logprobs_token(model, target_sid, user_feat, hist_sid, hist_len)
                ce_layer = (-gt_logp_tok).mean(dim=0)
                print("[DBG][SFT/GT] per-layer CE:", [round(x.item(), 4) for x in ce_layer])
        except Exception as e:
            print(f"[DBG][SFT/GT] failed: {type(e).__name__}: {e}")

        if debug_print_seq:
            _dbg_print_seq("target_sid", target_sid, n=debug_samples)
            try:
                best_seq = seq_all[torch.arange(B, device=seq_all.device), best_idx]  # [B,L]
            except Exception:
                best_seq = seq_all[:, 0, :]
            _dbg_print_seq("best_seq", best_seq, n=debug_samples)
            _dbg_print_seq("gt_in_group", seq_all[:, 0, :], n=debug_samples)
            _nc = None
            for _attr in ("num_classes", "vocab_size"):
                if hasattr(model, _attr):
                    try:
                        _nc = int(getattr(model, _attr))
                        break
                    except Exception:
                        _nc = None
            for i in range(min(debug_samples, B)):
                rb = r_tok[i, best_idx[i].item(), :].detach().cpu().tolist()
                rg = r_tok[i, 0, :].detach().cpu().tolist()
                print(f"[DBG][sample {i}] reward_tok(best)={rb} | reward_tok(gt)={rg}")
                if _nc is not None:
                    ts = target_sid[i].detach().cpu()
                    if (ts.numel() > 0) and ((ts.min().item() < 0) or (ts.max().item() >= _nc)):
                        print(f"[DBG][sample {i}] WARNING target_sid out of range: min={ts.min().item()} max={ts.max().item()} num_classes={_nc}")

    return total_loss, stats


def freeze_module(m: torch.nn.Module):
    for p in m.parameters():
        p.requires_grad = False


def main():
    code_dir = os.path.dirname(os.path.abspath(__file__))  # <repo>/code
    root_main = os.path.abspath(os.path.join(code_dir, os.pardir))  # <repo>

    hrpo_pkl_root = os.path.join(code_dir, 'dataset', 'kuairand', 'kuairand-Pure', 'hrpo_bucket')
    default_log_csv = os.path.join(root_main, 'dataset', 'kuairand', 'kuairand-Pure', 'data', 'log_session_4_08_to_5_08_Pure.csv')
    default_sid_map = os.path.join(code_dir, 'dataset', 'kuairand', 'kuairand-Pure', 'sid', '32_mask', 'video_sid_mapping.csv')
    default_user_feat = os.path.join(root_main, 'dataset', 'kuairand', 'kuairand-Pure', 'data', 'user_features_Pure_fillna.csv')
    default_hrpo_table = os.path.join(code_dir, 'dataset', 'kuairand', 'kuairand-Pure', 'hrpo', 'hrpo_table.pkl')
    default_init_ckpt = os.path.join(code_dir, 'checkpoints', 'checkpoints', 'onerec_value_v2_32_mask_mini', 'epoch_5.pt')
    default_model_dir = os.path.join(code_dir, 'checkpoints', 'checkpoints', 'onerec_value_v2_32_mask_mini', 'hrpo_rrpo_ntp')
    default_metrics_dir = os.path.join(code_dir, 'output')
    default_hrpo_table_paths = [
        os.path.join(hrpo_pkl_root, 'hrpo_click.pkl'),
        os.path.join(hrpo_pkl_root, 'hrpo_long_view.pkl'),
        os.path.join(hrpo_pkl_root, 'hrpo_like.pkl'),
        os.path.join(hrpo_pkl_root, 'hrpo_comment.pkl'),
        os.path.join(hrpo_pkl_root, 'hrpo_forward.pkl'),
        os.path.join(hrpo_pkl_root, 'hrpo_follow.pkl'),
        os.path.join(hrpo_pkl_root, 'hrpo_hate.pkl'),
    ]
    ap = argparse.ArgumentParser()

    ap.add_argument('--log_paths', nargs='+', default=[default_log_csv],
                    help='One or more log csv paths. Default matches train_hrpo_rrpo_ntp.sh')
    ap.add_argument('--sid_mapping_path', type=str, default=default_sid_map)
    ap.add_argument('--user_feat_path', type=str, default=default_user_feat)
    ap.add_argument('--label_col', type=str, default='is_click')
    ap.add_argument('--sid_depth', type=int, default=4)
    ap.add_argument('--num_classes', type=int, default=32)
    ap.add_argument('--max_hist_len', type=int, default=50)
    ap.add_argument('--max_hist_len_model', type=int, default=50)

    ap.add_argument('--hrpo_table_path', type=str, default=None,
                    help='Path to a single hrpo_table.pkl.')
    ap.add_argument('--hrpo_table_paths', type=str, nargs='+', default=None,
                    help='One or more HRPO table paths for multi-objective reward.')
    ap.add_argument('--reward_weights', type=str, default='1.0,0.7,0.5,0.5,0.5,0.5,0.0',
                    help='Comma-separated weights for each table in --hrpo_table_paths.')
    ap.add_argument('--reward_scale', type=float, default=30.0)

    ap.add_argument('--model_size', type=str, default='mini')
    ap.add_argument('--num_layers', type=int, default=3)
    ap.add_argument('--nhead', type=int, default=-1)
    ap.add_argument('--use_decoder_ctx', action='store_true')
    ap.add_argument('--init_ckpt', type=str, default=default_init_ckpt)
    ap.add_argument('--freeze_value_decoder', type=int, default=1)

    ap.add_argument('--group_size', type=int, default=18)
    ap.add_argument('--clip_eps', type=float, default=0.2)
    ap.add_argument('--kl_coef', type=float, default=0.1)
    ap.add_argument('--sft_coef', type=float, default=0.4)
    ap.add_argument('--old_update_freq', type=int, default=20)
    ap.add_argument('--no_residual_credit', action='store_true',
                    help='Ablation: disable residual-credit assignment; use sequence-level HRPO reward broadcast to all token positions.')
    ap.add_argument('--no_credit_to_go', action='store_true',
                    help='Ablation: disable credit-to-go; use immediate token rewards (no RTG accumulation).')
    ap.add_argument('--debug_every', type=int, default=50, help='Print debug stats every N train steps (0 disables).')
    ap.add_argument('--debug_samples', type=int, default=2, help='How many samples to print in debug block.')
    ap.add_argument('--debug_print_seq', action='store_true', help='Print token sequences in debug blocks.')
    ap.add_argument('--debug_reward_range', action='store_true', help='Print raw reward range (before reward_scale).')

    ap.add_argument('--lr', type=float, default=1e-5)
    ap.add_argument('--weight_decay', type=float, default=0.01)
    ap.add_argument('--grad_clip', type=float, default=1.0)
    ap.add_argument('--batch_size', type=int, default=256)
    ap.add_argument('--num_workers', type=int, default=8)
    ap.add_argument('--epochs', type=int, default=1)

    ap.add_argument('--model_dir', type=str, default=default_model_dir)
    ap.add_argument('--eval_every', type=int, default=40)
    ap.add_argument('--eval_at_start', type=int, default=1)

    ap.add_argument('--metrics_dir', type=str, default=default_metrics_dir,
                    help='Write jsonl metrics + PNG plots here (default matches train_hrpo_rrpo_ntp.sh).')
    ap.add_argument('--plot_every', type=int, default=200,
                    help='If >0, refresh plots every N steps (also plots at end). 0 disables plotting.')
    ap.add_argument('--plot_smooth', type=int, default=30,
                    help='Moving-average window for plotting (0/1 disables).')
    ap.add_argument('--clear_metrics', type=int, default=1,
                    help='If 1, delete existing train.jsonl/val.jsonl before writing new metrics.')
    ap.add_argument('--log_every', type=int, default=50)
    ap.add_argument('--eval_beam', type=int, default=50)
    ap.add_argument('--topk', type=str, default='1,5,10,20,50')

    ap.add_argument('--seed', type=int, default=2025)
    ap.add_argument('--device', type=str, default='cuda')
    args = ap.parse_args()
    print("[Args] kl_coef=", args.kl_coef, "sft_coef=", args.sft_coef, "clip_eps=", args.clip_eps, "reward_scale=", args.reward_scale, "old_update_freq=", args.old_update_freq)

    plotter = None
    if (getattr(args, "plot_every", 0) and int(args.plot_every) > 0) or (getattr(args, "metrics_dir", "") and str(args.metrics_dir).strip()):
        _metrics_dir = str(getattr(args, "metrics_dir", "")).strip()
        if not _metrics_dir:
            _metrics_dir = os.path.join(args.model_dir, "metrics")
        plotter = _StepMetricsPlotter(
            out_dir=_metrics_dir,
            smooth_window=int(getattr(args, "plot_smooth", 0)),
            clear=bool(int(getattr(args, "clear_metrics", 0))),
        )
    print("[Args] debug_every=", args.debug_every, "debug_samples=", args.debug_samples, "debug_print_seq=", args.debug_print_seq, "debug_reward_range=", args.debug_reward_range)

    os.makedirs(args.model_dir, exist_ok=True)
    seed_everything(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")

    hrpo_paths: List[str] = []
    if args.hrpo_table_paths is not None and len(args.hrpo_table_paths) > 0:
        hrpo_paths = list(args.hrpo_table_paths)
    elif args.hrpo_table_path is not None:
        hrpo_paths = [args.hrpo_table_path]

    if not hrpo_paths:
        if all(os.path.isfile(p) for p in default_hrpo_table_paths):
            hrpo_paths = list(default_hrpo_table_paths)
        elif os.path.isfile(default_hrpo_table):
            hrpo_paths = [default_hrpo_table]

    if not hrpo_paths:
        raise ValueError("Please provide --hrpo_table_path or --hrpo_table_paths")

    hrpo_weights: List[float]
    if args.reward_weights is None or str(args.reward_weights).strip() == "":
        hrpo_weights = [1.0 for _ in hrpo_paths]
    else:
        hrpo_weights = [float(x) for x in str(args.reward_weights).split(",")]
        if len(hrpo_weights) == 1 and len(hrpo_paths) > 1:
            hrpo_weights = hrpo_weights * len(hrpo_paths)
        if len(hrpo_weights) != len(hrpo_paths):
            raise ValueError(f"reward_weights length {len(hrpo_weights)} must match hrpo_paths length {len(hrpo_paths)}")

    hrpo_tables: List[HRPOTable] = [HRPOTable.load(p) for p in hrpo_paths]
    for p, w, t in zip(hrpo_paths, hrpo_weights, hrpo_tables):
        if t.sid_depth != args.sid_depth:
            print(f"[WARN] {p}: hrpo.sid_depth={t.sid_depth} != args.sid_depth={args.sid_depth}")
        print(f"[HRPO] loaded: {p} (weight={w})")

    full_dataset = KRPureValueDatasetWithUID(
        log_paths=args.log_paths,
        sid_mapping_path=args.sid_mapping_path,
        user_feat_path=args.user_feat_path,
        sid_cols=[f"sid_{i+1}" for i in range(args.sid_depth)],
        label_col=args.label_col,
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


    args.nhead = int(nhead)
    args.hid_dim = int(hid_dim)
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

    if args.init_ckpt:
        ckpt = torch.load(args.init_ckpt, map_location="cpu")
        if isinstance(ckpt, dict) and "model" in ckpt:
            model.load_state_dict(ckpt["model"], strict=False)
        else:
            model.load_state_dict(ckpt, strict=False)
        print(f"[Load] init_ckpt={args.init_ckpt}")

    if int(args.freeze_value_decoder) == 1 and hasattr(model, "value_decoder"):
        freeze_module(model.value_decoder)
        print("[Freeze] value_decoder frozen (HRPO-RRPO targets NTP only)")

    ref_model = None
    if args.kl_coef > 0:
        ref_model = deepcopy(model).eval().requires_grad_(False)

    old_model = deepcopy(model).eval().requires_grad_(False)


    ensure_build_ctx(model)
    ensure_build_ctx(old_model)
    if ref_model is not None:
        ensure_build_ctx(ref_model)

    trie_mask, trie_next = build_trie_from_sid_mapping(
        args.sid_mapping_path, args.sid_depth, args.num_classes, device
    )

    params = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in params)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[Params] trainable={n_train:,} total={n_total:,}")
    if n_train == 0:
        raise RuntimeError("No trainable parameters: check freeze flags.")
    optim = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    topk_list = [int(x) for x in args.topk.split(",") if x.strip()]
    step = 0
    best = -1.0
    t0 = time.time()
    ema = {}
    ema_beta = float(getattr(args, "ema_beta", 0.98))


    if args.eval_at_start:
        metrics0 = evaluate_hit_ndcg(
            model=model,
            data_loader=val_loader,
            device=device,
            trie_mask=trie_mask,
            trie_next=trie_next,
            beam_width=args.eval_beam,
            topk_list=topk_list,
        )
        print("[Val@start]", " ".join([f"{k}={v:.4f}" for k, v in metrics0.items()]))
        best = metrics0.get("hit@50", best)

    for ep in range(1, args.epochs + 1):
        for batch in train_loader:
            optim.zero_grad(set_to_none=True)

            loss, stats = rrpo_step_hrpo_token(
                model=model,
                old_model=old_model,
                ref_model=ref_model,
                hrpo=hrpo_tables,
                hrpo_weights=hrpo_weights,
                batch=batch,
                device=device,
                trie_mask=trie_mask,
                trie_next=trie_next,
                group_size=args.group_size,
                clip_eps=args.clip_eps,
                kl_coef=args.kl_coef,
                sft_coef=args.sft_coef,
                reward_scale=args.reward_scale,
                no_residual_credit=args.no_residual_credit,
                no_credit_to_go=args.no_credit_to_go,
                global_step=step,
                debug_every=args.debug_every,
                debug_samples=args.debug_samples,
                debug_print_seq=args.debug_print_seq,
                debug_reward_range=args.debug_reward_range,
            )

            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optim.step()

            step += 1
            if args.old_update_freq > 0 and (step % args.old_update_freq == 0):
                old_model.load_state_dict(model.state_dict(), strict=False)

            if step % args.log_every == 0:
                dt = time.time() - t0
                for _k in ("r_mean", "r_best_mean"):
                    if _k in stats:
                        v = float(stats[_k])
                        if _k not in ema:
                            ema[_k] = v
                        else:
                            ema[_k] = ema_beta * float(ema[_k]) + (1.0 - ema_beta) * v
                if "r_mean" in ema:
                    stats["r_mean_ema"] = float(ema["r_mean"])
                if "r_best_mean" in ema:
                    stats["r_best_ema"] = float(ema["r_best_mean"])

                msg = " ".join([f"{k}={v:.4f}" for k, v in stats.items()])
                print(f"[Train][ep={ep} step={step}] {msg} | dt={dt:.1f}s")
                t0 = time.time()

                if plotter is not None:
                    try:
                        plotter.log_train(step=step, ep=ep, stats=stats)
                        if int(getattr(args, "plot_every", 0)) > 0 and (step % int(getattr(args, "plot_every", 0)) == 0):
                            plotter.plot_train()
                    except Exception:
                        pass

            if step % args.eval_every == 0:
                metrics = evaluate_hit_ndcg(
                    model=model,
                    data_loader=val_loader,
                    device=device,
                    trie_mask=trie_mask,
                    trie_next=trie_next,
                    beam_width=args.eval_beam,
                    topk_list=topk_list,
                )
                print("[Val]", " ".join([f"{k}={v:.4f}" for k, v in metrics.items()]))

                if plotter is not None:
                    try:
                        plotter.log_val(step=step, ep=ep, stats=metrics)
                        plotter.plot_val()
                    except Exception:
                        pass

                score = metrics.get("hit@50", 0.0)
                if score > best:
                    best = score
                    save_path = os.path.join(args.model_dir, "best.pt")
                    torch.save({"model": model.state_dict(), "args": vars(args), "step": step}, save_path)
                    print(f"[Save] best -> {save_path} (hit@50={best:.4f})")

        save_path = os.path.join(args.model_dir, f"epoch_{ep}.pt")
        torch.save({"model": model.state_dict(), "args": vars(args), "step": step}, save_path)
        print(f"[Save] {save_path}")
    if plotter is not None:
        try:
            plotter.plot_all()
        except Exception:
            pass



    print("[Done]")


if __name__ == "__main__":
    main()
