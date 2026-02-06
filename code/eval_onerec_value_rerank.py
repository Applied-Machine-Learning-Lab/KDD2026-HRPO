# -*- coding: utf-8 -*-
"""
eval_onerec_value_rerank.py

Evaluate OneRec-with-Value in KuaiSim with optional LTV reranking.

Flow:
1) Generate SID candidates with beam search.
2) Score candidates with click probability and value prediction.
3) Rerank by either multiplicative or additive rule.
4) Execute top results in simulator and report metrics.
"""
import os
import argparse
import json
import time
import random
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, Any, Tuple, List, Optional

import numpy as np
import math
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.onerec_value import OneRecWithValue
from model.reward import get_immediate_reward
from env.KREnvironment_WholeSession_GPU import KREnvironment_WholeSession_GPU
import utils  #  HAC random seed

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def infer_hist_layers(state_dict, prefix="hist_encoder.net.layers."):
    idx = set()
    for k in state_dict.keys():
        if k.startswith(prefix):
            parts = k.split(".")
            if len(parts) > 3 and parts[3].isdigit():
                idx.add(int(parts[3]))
    return (max(idx) + 1) if idx else 1

def report_stats(self):
    """Print selection statistics to help diagnose beam/candidate mismatches."""
    if not getattr(self.args, "report_pick_stats", False):
        return
    users = max(1, int(self._stat_users))
    steps = max(1, int(self._stat_steps))
    fb_rate = float(self._stat_fallback_users) / float(users)
    if self._stat_first_hit_rank_cnt > 0:
        avg_rank = float(self._stat_first_hit_rank_sum) / float(self._stat_first_hit_rank_cnt)
    else:
        avg_rank = None
    print("=" * 40)
    print("[PickStats] steps=", steps, " users=", users)
    print(f"[PickStats] fallback_users={int(self._stat_fallback_users)}  fallback_rate={fb_rate:.4f}")
    if avg_rank is None:
        print("[PickStats] no beam hits recorded (check sid mapping / candidate pool / history filtering)")
    else:
        print(f"[PickStats] avg_first_hit_rank={avg_rank:.2f}  (0=top beam after scoring)")
    print("=" * 40)

def parse_args():
    parser = argparse.ArgumentParser()
    parser = KREnvironment_WholeSession_GPU.parse_model_args(parser)

    parser.add_argument(
        "--onerec_ckpt",
        type=str,
        required=True,
        help="OneRecWithValue trained checkpoint",
    )
    parser.add_argument(
        "--sid_mapping_path",
        type=str,
        default=str(PROJECT_ROOT / "code/dataset/kuairand/kuairand-Pure/sid/32_mask/video_sid_mapping.csv"),
    )

    parser.add_argument("--num_episodes", type=int, default=200)
    parser.add_argument(
        "--max_steps_per_episode",
        type=int,
        default=20,
        help="Max steps per episode.",
    )
    parser.add_argument(
        "--eval_log_path",
        type=str,
        default="../output/KuaiRand_Pure/eval/onerec_rerank.log",
    )
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda:0")

    parser.add_argument(
        "--beam_width", type=int, default=50, help="Beam width for candidate SID generation."
    )
    parser.add_argument(
        "--rerank_alpha",
        type=float,
        default=1.0,
        help="LTV rerank weight (alpha).",
    )
    parser.add_argument(
        "--rerank_formula",
        type=str,
        default="mul",
        choices=["mul", "add"],
        help="Rerank rule: mul => Prob*LTV*alpha, add => Prob+LTV*alpha.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=2025,
        help="Random seed for reproducible evaluation.",
    )
    parser.add_argument(
        "--sid_depth",
        type=int,
        default=4,
        help="",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=32,
        help="",
    )
    parser.add_argument(
        "--max_hist_len",
        type=int,
        default=50,
        help="",
    )
    parser.add_argument(

        "--hist_reverse_valid",

        action="store_true",

        help=" right-pad history ( hist_len )(forhistorytraining)",

    )

    parser.add_argument(

        "--report_pick_stats",

        action="store_true",

        help="evaluation beam /fallback()",

    )
    parser.add_argument(
        "--allow_repeat",
        action="store_true",
        help="history item(default;  fallback  history)",
    )





    

    
    parser.add_argument(
        "--report_debias_metrics",
        action="store_true",
        help="(evaluation, /environment): coverage, entropy/gini, long-tail, fairness. ",
    )
    parser.add_argument(
        "--longtail_q",
        type=float,
        default=0.2,
        help=" q:  popularity  bottom-q  item  long-tail( reader  popularity ). ",
    )
    parser.add_argument(
        "--random_policy",
        action="store_true",
        help=" baseline: candidate item(historydedupe)",
    )
    parser.add_argument(
        "--debug_urm",
        action="store_true",
        help=" URM output,  double-sigmoid (, )",
    )
    parser.add_argument(
        "--debug_urm_steps",
        type=int,
        default=1,
        help="debug_urm  step (default 1)",
    )

    parser.add_argument(
        "--debug_hist",
        action="store_true",
        help=" history (+leftpad->rightpad)(, )",
    )
    parser.add_argument(
        "--debug_hist_steps",
        type=int,
        default=1,
        help="debug_hist  step (default 1)",
    )


    args = parser.parse_args()
    return args



def load_onerec_value_model(ckpt_path: str, device: torch.device):
    """
    Load OneRecWithValue checkpoint.

    Supports:
      - New trainer format: {"model": state_dict, "args": {...}}
      - Older format: {"model_state_dict": ..., "config": ...}

    Returns:
      model, cfg(dict)
    """
    ckpt = torch.load(ckpt_path, map_location=device)

    if isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
        ckpt_args = ckpt.get("args", {}) or {}
        cfg = {"from_ckpt_args": True, "ckpt_args": ckpt_args}
    else:
        state_dict = ckpt.get("model_state_dict", ckpt)
        cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}

        ckpt_args = cfg.get("args", {}) if isinstance(cfg, dict) else {}

    sd = ckpt.get("model", ckpt)
    
    if "sid_embedding.weight" not in state_dict:
        raise KeyError("sid_embedding.weight not found in checkpoint state_dict")
    num_classes, hid_dim = state_dict["sid_embedding.weight"].shape
    user_feat_dim = state_dict["user_proj.weight"].shape[1]

    def _get_num_emb(key: str, default: int):
        w = state_dict.get(key, None)
        return int(w.shape[0]) if w is not None else int(default)
    def _coerce_int(v, default: int) -> int:
        if v is None:
            return int(default)
        if isinstance(v, (bool, np.bool_)):
            return int(bool(v))
        if isinstance(v, (int, np.integer)):
            return int(v)
        if isinstance(v, (float, np.floating)):
            return int(v)
        if isinstance(v, str):
            s = v.strip().lower()
            if s in ("", "none", "null"):
                return int(default)
            try:
                return int(float(s))  # handles "2.0"
            except Exception:
                return int(default)
        try:
            return int(v)
        except Exception:
            return int(default)

    def _coerce_float(v, default: float) -> float:
        if v is None:
            return float(default)
        if isinstance(v, (float, np.floating, int, np.integer)):
            return float(v)
        if isinstance(v, str):
            s = v.strip().lower()
            if s in ("", "none", "null"):
                return float(default)
            try:
                return float(s)
            except Exception:
                return float(default)
        try:
            return float(v)
        except Exception:
            return float(default)

    def _coerce_bool(v, default: bool) -> bool:
        if v is None:
            return bool(default)
        if isinstance(v, (bool, np.bool_)):
            return bool(v)
        if isinstance(v, (int, np.integer)):
            return bool(int(v))
        if isinstance(v, str):
            s = v.strip().lower()
            if s in ("1", "true", "t", "yes", "y", "on"):
                return True
            if s in ("0", "false", "f", "no", "n", "off", "", "none", "null"):
                return False
            return bool(default)
        return bool(v)

    def _infer_value_layers(sd_: Dict[str, Any]) -> int:
        idx = set()
        for k in sd_.keys():
            if k.startswith("value_decoder.blocks.") or k.startswith("value_decoder.layers."):
                parts = k.split(".")
                if len(parts) > 2 and parts[2].isdigit():
                    idx.add(int(parts[2]))
        return (max(idx) + 1) if idx else 2

    def _infer_decoder_layers(sd_: Dict[str, Any]) -> int:
        """Infer decoder Transformer depth from checkpoint state_dict keys.

        Common patterns:
          - decoder.net.layers.{i}.*
          - decoder.layers.{i}.*

        Falls back to 2 if not found.
        """
        idx = set()
        prefixes = ["decoder.net.layers.", "decoder.layers."]
        for k in sd_.keys():
            for p in prefixes:
                if k.startswith(p):
                    rest = k[len(p):]
                    head = rest.split(".", 1)[0]
                    if head.isdigit():
                        idx.add(int(head))
        return (max(idx) + 1) if idx else 2

    sid_depth = _coerce_int(
        ckpt_args.get("sid_depth", None),
        _get_num_emb("dec_pos_emb.weight", 4),
    )
    max_hist_len = _coerce_int(
        ckpt_args.get("max_hist_len", ckpt_args.get("max_seq_len", None)),
        _get_num_emb("hist_item_pos_emb.weight", 50),
    )

    default_dec_layers = _infer_decoder_layers(state_dict)
    num_layers = _coerce_int(
        ckpt_args.get("num_layers", ckpt_args.get("num_decoder_block", None)),
        default_dec_layers,
    )
    hist_num_layers = ckpt_args.get("hist_num_layers", None)
    hist_num_layers = _coerce_int(hist_num_layers, infer_hist_layers(sd))

    nhead = _coerce_int(ckpt_args.get("nhead", None), 4)
    dropout = _coerce_float(ckpt_args.get("dropout", ckpt_args.get("dropout_ratio", None)), 0.1)

    use_user_token = _coerce_bool(ckpt_args.get("use_user_token", True), True)
    use_decoder_ctx = _coerce_bool(ckpt_args.get("use_decoder_ctx", False), False)

    value_layers = ckpt_args.get("value_layers", ckpt_args.get("num_value_layers", ckpt_args.get("value_num_layers", None)))
    value_layers = _coerce_int(value_layers, _infer_value_layers(state_dict))

    model = OneRecWithValue(
        num_decoder_block=num_layers,
        hid_dim=hid_dim,
        nhead=nhead,
        sid_depth=sid_depth,
        num_classes=num_classes,
        user_feat_dim=user_feat_dim,
        max_hist_len=max_hist_len,
        hist_num_layers=hist_num_layers,
        dropout_ratio=dropout,
        use_user_token=use_user_token,
        use_decoder_ctx=use_decoder_ctx,
        value_layers=value_layers,
    ).to(device)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if len(missing) > 0 or len(unexpected) > 0:
        print(f"[LoadDiag] missing={len(missing)} unexpected={len(unexpected)}")
        if len(missing) > 0:
            print("[LoadDiag] missing(head):", missing[:12])
        if len(unexpected) > 0:
            print("[LoadDiag] unexpected(head):", unexpected[:12])

    model.eval()

    cfg = cfg or {}
    cfg["ckpt_path"] = ckpt_path
    cfg["hid_dim"] = hid_dim
    cfg["num_classes"] = num_classes
    cfg["sid_depth"] = sid_depth
    cfg["max_hist_len"] = max_hist_len
    cfg["user_feat_dim"] = user_feat_dim
    cfg["nhead"] = nhead
    cfg["num_layers"] = num_layers
    cfg["hist_num_layers"] = hist_num_layers
    cfg["use_user_token"] = use_user_token
    cfg["use_decoder_ctx"] = use_decoder_ctx
    cfg["value_layers"] = value_layers

    return model, cfg

def build_iid2sid(reader, mapping_path: str, sid_depth: int, device: torch.device):
    """
    Returns:
      iid2sid:     [max_iid+1, L] long tensor on device (0 row is PAD)
      sid2iid_map: dict[tuple(sid_1..sid_L)] -> list[iid]  (handles collisions)

    Notes:
      - KuaiRand env uses encoded item ids (item_id_enc) in [1..n_item]. Mapping csv may contain raw video_id.
      - This function prefers encoded id columns (item_id/iid/item_id_enc). If only raw id exists (video_id),
        it converts raw->encoded using reader.item_id_vocab when available.
    """
    df = pd.read_csv(mapping_path)

    sid_cols = [c for c in df.columns if c.startswith("sid_")]
    sid_cols = sid_cols[:sid_depth]
    if len(sid_cols) != sid_depth:
        raise ValueError(f"Need {sid_depth} sid cols, got {sid_cols}")

    id_col = None
    for c in ["item_id_enc", "item_id", "iid", "itemid", "itemId", "video_id", "videoId"]:
        if c in df.columns:
            id_col = c
            break
    if id_col is None:
        raise ValueError(f"Cannot find item id column in mapping. Columns: {list(df.columns)[:20]}")

    n_items = None
    try:
        stats = reader.get_statistics()
        n_items = int(stats.get("n_item", None))
    except Exception:
        n_items = None

    item_id_vocab = getattr(reader, "item_id_vocab", None)
    id_col_eff = id_col

    if id_col in ["video_id", "videoId"] and isinstance(item_id_vocab, dict) and len(item_id_vocab) > 0:
        raw = df[id_col].to_numpy()
        in_keys = float(sum((x in item_id_vocab) for x in raw)) / max(1, len(raw))
        in_vals = float(sum((x in set(item_id_vocab.values())) for x in raw)) / max(1, len(raw))
        max_raw = int(raw.max()) if len(raw) else -1
        if n_items is None:
            try:
                n_items = int(max(item_id_vocab.values()))
            except Exception:
                n_items = None

        df["_iid_enc"] = df[id_col].map(item_id_vocab).fillna(0).astype(int)
        id_col_eff = "_iid_enc"
        kept = int((df["_iid_enc"] > 0).sum())
        total = int(len(df))
        print(f"[SID] id_col='{id_col}' looks RAW (in_keys={in_keys:.2f}, in_vals={in_vals:.2f}, max_raw={max_raw}, n_items={n_items}); converted -> encoded. kept {kept}/{total} rows.")
        df = df[df["_iid_enc"] > 0].copy()

    if n_items is not None:
        max_iid = int(n_items)
    else:
        max_iid = int(df[id_col_eff].max())

    iid2sid = torch.zeros((max_iid + 1, sid_depth), dtype=torch.long, device=device)

    sid2iid_map: Dict[tuple, List[int]] = {}
    collision_cnt = 0

    iids = df[id_col_eff].to_numpy()
    sid_mat = df[sid_cols].to_numpy()

    for iid, sid_row in zip(iids, sid_mat):
        iid = int(iid)
        if iid <= 0 or iid > max_iid:
            continue
        sid = tuple(int(x) for x in sid_row.tolist())
        iid2sid[iid] = torch.tensor(sid, dtype=torch.long, device=device)

        if sid not in sid2iid_map:
            sid2iid_map[sid] = [iid]
        else:
            sid2iid_map[sid].append(iid)
            if len(sid2iid_map[sid]) == 2:
                collision_cnt += 1

    if collision_cnt > 0:
        print(f"[SID][WARN] collisions: {collision_cnt} SID tuples map to >1 iid (will choose among candidates at rerank time).")

    return iid2sid, sid2iid_map


class OneRecValuePolicy:
    def __init__(self, model, iid2sid, sid2iid_map, cfg, device, slate_size, reader, args):
        self.model = model
        self.iid2sid = iid2sid
        self.sid2iid_map = sid2iid_map

        self.cfg = cfg
        self.device = device
        self.slate_size = int(slate_size)
        self.reader = reader
        self.args = args

        self.sid_depth = int(args.sid_depth)
        self.num_classes = int(args.num_classes)
        self.max_hist_len = int(args.max_hist_len)

        self.beam_width = int(args.beam_width)
        self.alpha = float(args.rerank_alpha)
        self.formula = str(args.rerank_formula).lower()

        self._build_trie()
        self._dbg_hist_step = 0  # for --debug_hist
        self._stat_users = 0
        self._stat_steps = 0
        self._stat_fallback_users = 0
        self._stat_first_hit_rank_sum = 0
        self._stat_first_hit_rank_cnt = 0

    @staticmethod
    def _reverse_valid_rightpad(hist_iids: torch.Tensor, hist_len: torch.Tensor) -> torch.Tensor:
        """Reverse the order of the valid prefix (length hist_len) in a RIGHT-PADDED [B,H] history."""
        if hist_iids is None:
            return hist_iids
        if not torch.is_tensor(hist_iids):
            hist_iids = torch.as_tensor(hist_iids)
        if not torch.is_tensor(hist_len):
            hist_len = torch.as_tensor(hist_len, device=hist_iids.device)
        B, H = hist_iids.shape
        t = torch.arange(H, device=hist_iids.device).view(1, H).expand(B, H)
        hl = hist_len.view(B, 1).clamp(min=0, max=H)
        rev_t = (hl - 1 - t).clamp(min=0)  # for t>=hl, value doesn't matter
        idx = torch.where(t < hl, rev_t, t)
        return hist_iids.gather(1, idx)

    @staticmethod
    def _pad_or_trim_user_feat(x: torch.Tensor, target_dim: int) -> torch.Tensor:
        """Pad with zeros on the RIGHT or trim to match target_dim."""
        if x is None:
            return x
        if not torch.is_tensor(x):
            x = torch.as_tensor(x)
        if x.dim() == 1:
            x = x.view(1, -1)
        if x.size(1) > target_dim:
            return x[:, :target_dim]
        if x.size(1) < target_dim:
            pad = torch.zeros(
                x.size(0),
                target_dim - x.size(1),
                device=x.device,
                dtype=x.dtype,
            )
            return torch.cat([x, pad], dim=1)
        return x

    @staticmethod
    def _extract_hist_len(user_history: dict, hist_iids: torch.Tensor) -> torch.Tensor:
        """Extract history length from env observation with key fallbacks."""
        hist_len = None
        if isinstance(user_history, dict):
            for k in ("len", "history_length", "length", "hist_len", "seq_len"):
                if k in user_history and user_history[k] is not None:
                    hist_len = user_history[k]
                    break
        if hist_len is None:
            return (hist_iids > 0).sum(dim=1).long()

        if isinstance(hist_len, np.ndarray):
            hist_len = torch.from_numpy(hist_len)
        if not torch.is_tensor(hist_len):
            hist_len = torch.tensor(hist_len, dtype=torch.long)

        if hist_len.dim() == 0:
            hist_len = hist_len.expand(hist_iids.size(0))
        elif hist_len.dim() > 1:
            hist_len = hist_len.view(hist_len.size(0))

        return hist_len.long()

    def _user_profile_to_tensor(self, user_profile, batch_size: int) -> torch.Tensor:
        """Convert user_profile from Tensor/ndarray/dict to a [B,F] float tensor."""
        device = self.device
        if torch.is_tensor(user_profile):
            x = user_profile
        elif isinstance(user_profile, np.ndarray):
            x = torch.from_numpy(user_profile)
        elif isinstance(user_profile, dict):
            feat_names = None
            try:
                feat_names = self.reader.get_statistics().get("user_features", None)
                if feat_names is not None:
                    feat_names = list(feat_names)
            except Exception:
                feat_names = None
            if feat_names is None:
                feat_names = sorted(list(user_profile.keys()))

            cols = []
            for name in feat_names:
                v = user_profile.get(name, None)
                if v is None:
                    cols.append(torch.zeros(batch_size, dtype=torch.float32))
                    continue
                if torch.is_tensor(v):
                    t = v
                elif isinstance(v, np.ndarray):
                    t = torch.from_numpy(v)
                else:
                    t = torch.tensor(v)
                if t.dim() == 0:
                    t = t.expand(batch_size)
                elif t.dim() > 1:
                    t = t.view(t.size(0))
                cols.append(t.float().cpu())
            x = torch.stack(cols, dim=1)
        else:
            x = torch.zeros((batch_size, 0), dtype=torch.float32)

        return x.to(device).float()

    @staticmethod
    def _leftpad_to_rightpad(hist_iids: torch.Tensor, hist_len: torch.Tensor) -> torch.Tensor:
        """Shift valid tail to front if history is left-padded (KuaiRand reader style)."""
        B, H = hist_iids.shape
        if H == 0:
            return hist_iids
        z0 = (hist_iids[:, 0] == 0).float().mean().item()
        z1 = (hist_iids[:, -1] == 0).float().mean().item()
        if z0 <= z1:
            return hist_iids

        L = hist_len.clamp(min=0, max=H)
        idx = torch.arange(H, device=hist_iids.device).unsqueeze(0).expand(B, H)
        base = (H - L).unsqueeze(1)
        src = base + idx
        mask = idx < L.unsqueeze(1)
        src = torch.where(mask, src, torch.zeros_like(src))
        out = hist_iids.gather(1, src)
        out = torch.where(mask, out, torch.zeros_like(out))
        return out


    def _build_trie(self):
        """
        Build trie over all SID tuples in sid2iid_map (keys).
        """
        device = self.device
        trie_graph: Dict[int, Dict[int, int]] = {0: {}}
        node_count = 1

        for sid in self.sid2iid_map.keys():
            node = 0
            for tok in sid:
                tok = int(tok)
                if tok < 0 or tok >= self.num_classes:
                    node = None
                    break
                if tok not in trie_graph[node]:
                    trie_graph[node][tok] = node_count
                    trie_graph[node_count] = {}
                    node_count += 1
                node = trie_graph[node][tok]
            if node is None:
                continue

        self.trie_mask = torch.full((node_count, self.num_classes), -float("inf"), device=device)
        self.trie_next = torch.zeros((node_count, self.num_classes), dtype=torch.long, device=device)
        for u, trans in trie_graph.items():
            for tok, v in trans.items():
                self.trie_mask[u, tok] = 0.0
                self.trie_next[u, tok] = v

        print(f"[Trie] nodes={node_count}, vocab={self.num_classes}")

    @torch.no_grad()
    def _get_context_and_history(self, hist_iids, hist_len, user_profile):
        """Build encoder memory from user history + user profile.
    
        Padding convention:
          - Env commonly returns LEFT-PADDED history (0s on the left, recent items on the right).
          - OneRec encoder expects RIGHT-PADDED history (valid first, PAD=0 at end) + hist_len.
    
        Returns: (ctx, enc_out, enc_mask, user_vec, hist_sids, hist_iids_proc, hist_len_proc)
        """
        device = self.device
        hist_iids = torch.as_tensor(hist_iids, device=device).long()
        hist_len = torch.as_tensor(hist_len, device=device).long()
        B, Henv = hist_iids.shape
        H = int(self.max_hist_len)
    
        if Henv >= H:
            hist_iids = hist_iids[:, -H:]
        else:
            pad_left = torch.zeros((B, H - Henv), dtype=torch.long, device=device)
            hist_iids = torch.cat([pad_left, hist_iids], dim=1)
    
        hist_len = hist_len.clamp(min=0, max=H)
    
        hist_iids = self._leftpad_to_rightpad(hist_iids, hist_len)
    
        hist_len = (hist_iids > 0).sum(dim=1).long()
    
        user_feat = self._user_profile_to_tensor(user_profile, hist_iids.shape[0])
        user_feat = self._pad_or_trim_user_feat(user_feat, self.model.user_proj.in_features)
        user_vec = self.model.user_proj(user_feat)
    
        if getattr(self.args, "hist_reverse_valid", False):
            hist_iids = self._reverse_valid_rightpad(hist_iids, hist_len)

        hist_sids = self.iid2sid[hist_iids]  # [B,H,L]
    
        enc_out, enc_mask = self.model.encode_history_seq(hist_sids, hist_len, user_ctx=user_vec)
    
        ctx = None
        if getattr(self.model, "use_decoder_ctx", False) and hasattr(self.model, "build_ctx"):
            ctx = self.model.build_ctx(enc_out, enc_mask, user_vec)
        else:
            ctx = None

    
        return ctx, enc_out, enc_mask, user_vec, hist_sids, hist_iids, hist_len

    @torch.no_grad()
    def _beam_search(self, enc_out, enc_mask, ctx=None):
        """
        Returns:
          sequences: [B,W,L]
          log_probs: [B,W]
        """
        model = self.model
        device = self.device

        B = enc_out.size(0)
        W = max(self.beam_width, 1)
        D = model.hid_dim

        sequences = torch.zeros((B, W, 1), dtype=torch.long, device=device)
        logp = torch.full((B, W), -float("inf"), device=device)
        logp[:, 0] = 0.0
        nodes = torch.zeros((B, W), dtype=torch.long, device=device)

        enc_flat = enc_out.unsqueeze(1).expand(B, W, -1, -1).reshape(B * W, enc_out.size(1), D)
        msk_flat = enc_mask.unsqueeze(1).expand(B, W, -1).reshape(B * W, enc_mask.size(1))

        if ctx is not None:
            ctx_flat = ctx.unsqueeze(1).expand(B, W, -1).reshape(B * W, -1)
        else:
            ctx_flat = None

        for t in range(self.sid_depth):
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
            logits = model.out_proj(dec[:, -1, :]).view(B, W, self.num_classes)

            mask = self.trie_mask[nodes]  # [B,W,V]
            logits = logits + mask

            step_logp = torch.log_softmax(logits, dim=-1)  # [B,W,V]
            total = logp.unsqueeze(-1) + step_logp

            flat = total.view(B, W * self.num_classes)
            topv, topi = torch.topk(flat, k=W, dim=-1)

            next_beam = topi // self.num_classes
            next_tok = topi % self.num_classes

            new_seq = torch.gather(sequences, 1, next_beam.unsqueeze(-1).expand(-1, -1, sequences.size(-1)))
            new_seq = torch.cat([new_seq, next_tok.unsqueeze(-1)], dim=-1)
            sequences = new_seq

            logp = topv
            prev_nodes = torch.gather(nodes, 1, next_beam)
            nodes = self.trie_next[prev_nodes, next_tok]

        return sequences[:, :, 1:], logp

    @torch.no_grad()
    def _predict_ltv(self, enc_out, enc_mask, seqs, ctx=None):
        """
        seqs: [B,W,L] (SID tokens)
        Returns:
          ltv_scores: [B,W]
        """
        model = self.model
        device = self.device

        B, W, L = seqs.shape
        D = model.hid_dim
        BW = B * W

        flat = seqs.reshape(BW, L)
        inp = torch.zeros_like(flat)
        if L > 1:
            inp[:, 1:] = flat[:, :-1]

        x = model.sid_embedding(inp)  # [BW,L,D]
        x[:, 0, :] = model.bos

        if hasattr(model, "dec_pos_emb"):
            pos = torch.arange(L, device=device)
            x = x + model.dec_pos_emb(pos)[None, :, :]
        else:
            pos = torch.arange(L, device=device)
            x = x + model.pos_embedding(pos)[None, :, :]

        if ctx is not None:
            ctx_flat = ctx.unsqueeze(1).expand(B, W, -1).reshape(BW, -1)
            x = x + ctx_flat.unsqueeze(1)

        enc_flat = enc_out.unsqueeze(1).expand(B, W, -1, -1).reshape(BW, enc_out.size(1), D)
        msk_flat = enc_mask.unsqueeze(1).expand(B, W, -1).reshape(BW, enc_mask.size(1))

        tgt_mask = model._build_causal_mask(L, device=device)
        dec_feats = model.decoder(tgt=x, tgt_mask=tgt_mask, memory=enc_flat, memory_key_padding_mask=msk_flat)

        _, ltv_pred = model.value_decoder(sequence_emb=enc_flat, dec_feats=dec_feats, enc_padding_mask=msk_flat)
        ltv_last = ltv_pred[:, -1]  # [BW]
        return ltv_last.view(B, W)

    @torch.no_grad()
    def act(self, observation, candidate_info):
        """
        observation:
          - observation["user_profile"]: [B,F] float
          - observation["user_history"]["history"]: [B,Henv] long
          - observation["user_history"]["history_length"] (or "len"): [B] long
        candidate_info:
          - candidate_info["item_id"]: [B,n_candidate] long
        """
        device = self.device
        user_profile = observation["user_profile"]
        uh = observation["user_history"]
        hist_iids = uh["history"]
        hist_len = self._extract_hist_len(uh, torch.as_tensor(hist_iids))

        hist_iids = torch.as_tensor(hist_iids, device=device).long()
        hist_len = torch.as_tensor(hist_len, device=device).long()

        cand_raw = candidate_info.get('item_id', None) if isinstance(candidate_info, dict) else None
        if cand_raw is None and isinstance(candidate_info, dict):
            cand_raw = candidate_info.get('item_ids', None)
        if cand_raw is None:
            try:
                cand_raw = candidate_info['item_id']
            except Exception as e:
                raise KeyError("candidate_info must contain 'item_id' (or 'item_ids')") from e
        cand_iids = torch.as_tensor(cand_raw, device=device).long()
        B = int(hist_iids.shape[0])
        if cand_iids.dim() == 1:
            cand_iids = cand_iids.unsqueeze(0).expand(B, -1)
        elif cand_iids.dim() == 2:
            if cand_iids.size(0) == 1 and B > 1:
                cand_iids = cand_iids.expand(B, -1)
            elif cand_iids.size(0) != B:
                cand_iids = cand_iids[:1].expand(B, -1)
        else:
            raise ValueError(f"candidate_info['item_id'] must be 1D or 2D, got shape={tuple(cand_iids.shape)}")

        ctx, enc_out, enc_mask, user_vec, hist_sids, hist_iids_proc, hist_len_proc = self._get_context_and_history(hist_iids, hist_len, user_profile)

        if getattr(self.args, "debug_hist", False) and self._dbg_hist_step < int(getattr(self.args, "debug_hist_steps", 1)):
            nz = (hist_iids_proc > 0).sum(dim=1)
            print("[DBG-HIST] hist_len(head)=", hist_len_proc[:8].tolist(),
                  " nonzero(head)=", nz[:8].tolist(),
                  " first_is_zero_frac=", (hist_iids_proc[:, 0] == 0).float().mean().item(),
                  " last_is_zero_frac=", (hist_iids_proc[:, -1] == 0).float().mean().item())
            self._dbg_hist_step += 1



        seqs, logp = self._beam_search(enc_out, enc_mask, ctx=ctx)  # [B,W,L] & [B,W]

        ltv = self._predict_ltv(enc_out, enc_mask, seqs, ctx=ctx)  # [B,W]

        prob = torch.exp(logp).clamp(min=1e-12)  # [B,W]

        if self.formula == "mul":
            scores = prob * (self.alpha * ltv)
        else:
            scores = prob + (self.alpha * ltv)

        B, N = cand_iids.shape
        actions = torch.zeros((B, self.slate_size), dtype=torch.long, device=device)

        hist_iids = hist_iids.to(device).long()

        do_stats = getattr(self.args, "report_pick_stats", False)
        for b in range(B):
            first_hit_rank = None
            cand_list = cand_iids[b].tolist()
            cand_pos = {int(iid): j for j, iid in enumerate(cand_list)}

            h = hist_iids[b].tolist()
            h_set = set([int(x) for x in h if int(x) > 0])

            picked = []
            used_pos = set()

            for w in torch.argsort(scores[b], descending=True).tolist():
                sid = tuple(int(x) for x in seqs[b, w].tolist())
                if sid not in self.sid2iid_map:
                    continue
                for iid in self.sid2iid_map[sid]:
                    iid = int(iid)
                    if (not getattr(self.args, "allow_repeat", False)) and (iid in h_set):
                        continue
                    if iid not in cand_pos:
                        continue
                    pos = cand_pos[iid]
                    if pos in used_pos:
                        continue
                    if do_stats and (first_hit_rank is None):
                        first_hit_rank = int(w)
                    picked.append(pos)
                    used_pos.add(pos)
                    if len(picked) >= self.slate_size:
                        break
                if len(picked) >= self.slate_size:
                    break

            if do_stats:
                if first_hit_rank is not None:
                    self._stat_first_hit_rank_sum += first_hit_rank
                    self._stat_first_hit_rank_cnt += 1
                if len(picked) < self.slate_size:
                    self._stat_fallback_users += 1

            if len(picked) < self.slate_size:
                for pos in range(N):
                    if pos in used_pos:
                        continue
                    if (not getattr(self.args, "allow_repeat", False)) and (int(cand_list[pos]) in h_set):
                        continue
                    picked.append(pos)
                    used_pos.add(pos)
                    if len(picked) >= self.slate_size:
                        break

            actions[b, :] = torch.tensor(picked[: self.slate_size], device=device, dtype=torch.long)

        return actions




class RandomPolicy:
    """
 candidate slate_size  item(history; ). 
 :  action  candidate pool  local index( env.candidate_iids align). 
 """
    def __init__(self, slate_size: int, device: torch.device):
        self.slate_size = int(slate_size)
        self.device = device

    def act(self, observation, candidate_info):
        cand_iids = candidate_info["item_id"]
        if torch.is_tensor(cand_iids):
            cand_iids = cand_iids.detach().cpu().numpy()
        cand_iids = np.asarray(cand_iids)
        n_cand = int(len(cand_iids))

        hist = observation["user_history"]["history"]
        if torch.is_tensor(hist):
            hist_np = hist.detach().cpu().numpy()
        else:
            hist_np = np.asarray(hist)

        B = int(hist_np.shape[0])
        actions = []

        all_idx = np.arange(n_cand)

        for i in range(B):
            user_hist = set(hist_np[i].tolist())

            mask = np.array([gid not in user_hist for gid in cand_iids], dtype=bool)
            remain = all_idx[mask]

            pool = remain if remain.size >= self.slate_size else all_idx

            if pool.size <= 0:
                pick = np.zeros((self.slate_size,), dtype=np.int64)
            else:
                if pool.size >= self.slate_size:
                    pick = np.random.choice(pool, size=self.slate_size, replace=False)
                else:
                    pick = np.random.choice(pool, size=self.slate_size, replace=True)

            actions.append(pick.tolist())

        return torch.tensor(actions, dtype=torch.long, device=self.device)


def _safe_to_tensor(x, device):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    if not torch.is_tensor(x):
        x = torch.tensor(x)
    return x.to(device)

def _pretty_stats(name: str, t: torch.Tensor):
    t_det = t.detach()
    return (
        f"{name}: shape={tuple(t_det.shape)}, dtype={t_det.dtype}, device={t_det.device}, "
        f"min={t_det.min().item():.6f}, max={t_det.max().item():.6f}, "
        f"mean={t_det.mean().item():.6f}, std={t_det.std(unbiased=False).item():.6f}"
    )

def _ratio_to_posrate(r: float) -> float:
    rr = abs(float(r))
    if rr < 0:
        rr = -rr
    return rr / (1.0 + rr) if rr >= 0 else 0.0

def _debug_urm_once(env, observation, action, step_id: int, beh_names):
    """
    Diagnostics to detect whether env is doing double-sigmoid:
      - env.get_response uses out_dict['probs'] then torch.sigmoid(behavior_scores) then bernoulli. 
    We rebuild the same URM batch (without changing env state) and print ranges.
    """
    try:
        device = env.device if hasattr(env, "device") else action.device
        B = int(action.shape[0])
        slate = int(action.shape[1]) if action.dim() == 2 else 1

        batch = {"item_id": env.candidate_iids[action]}
        batch.update(observation["user_profile"])
        batch.update(observation["user_history"])
        batch.update({k: v[action] for k, v in env.candidate_item_meta.items()})

        with torch.no_grad():
            out = env.immediate_response_model(batch)

        if "probs" not in out:
            print(f"[URM-DBG] step={step_id} out_dict has no 'probs'. keys={list(out.keys())}")
            return

        probs = out["probs"]
        print(f"[URM-DBG] step={step_id} out_dict.keys={list(out.keys())}")
        print("[URM-DBG] " + _pretty_stats("out['probs']", probs))

        pmin = float(probs.min().item())
        pmax = float(probs.max().item())
        looks_like_prob = (pmin >= -1e-4) and (pmax <= 1.0 + 1e-4)

        sig = torch.sigmoid(probs)
        print("[URM-DBG] " + _pretty_stats("sigmoid(out['probs'])", sig))

        item_enc = env.candidate_item_encoding[action].view(B, slate, -1)
        item_enc_norm = F.normalize(item_enc, p=2.0, dim=-1)
        corr_factor = env.get_intra_slate_similarity(item_enc_norm)  # (B, slate)

        rho = float(getattr(env, "rho", 0.0))
        point_scores_env = (sig - corr_factor.view(B, slate, 1) * rho).clamp(0.0, 1.0)
        point_scores_nosig = (probs - corr_factor.view(B, slate, 1) * rho).clamp(0.0, 1.0)

        print("[URM-DBG] " + _pretty_stats("point_scores_env(sigmoid-penalty)", point_scores_env))
        print("[URM-DBG] " + _pretty_stats("point_scores_nosig(probs-penalty)", point_scores_nosig))

        K = point_scores_env.size(-1)
        names = beh_names[:K]
        env_means = point_scores_env.mean(dim=(0, 1)).detach().cpu().numpy()
        nosig_means = point_scores_nosig.mean(dim=(0, 1)).detach().cpu().numpy()

        print("[URM-DBG] mean prob per behavior (env-way vs no-sigmoid):")
        for i, n in enumerate(names):
            print(f"  {n}: env={env_means[i]:.6f} | nosig={nosig_means[i]:.6f}")

        try:
            stats = env.reader.get_statistics()
            pri = stats.get("feedback_negative_sample_rate", None)
            if pri is not None:
                print("[URM-DBG] prior pos-rate from data (ratio pos/neg -> pos_rate=r/(1+r)):")
                for n in names:
                    if n in pri:
                        pr = _ratio_to_posrate(float(pri[n]))
                        print(f"  {n}: ratio={float(pri[n]):.6f} -> pos_rate~{100.0*pr:.4f}%")
        except Exception as e:
            print(f"[URM-DBG] prior rate read failed: {e}")

        if looks_like_prob:
            print(
                "[URM-DBG][WARN] out['probs']  [0,1], . \n"
                " environment get_response  torch.sigmoid(out['probs'])  bernoulli,  double-sigmoid, \n"
                "  50%+( 55%~70% ). \n"
                " fix: environmentif out['probs'] ,  sigmoid;  logits,  sigmoid. \n"
            )
        else:
            print(
                "[URM-DBG] out['probs'] ( logits),  env  sigmoid . \n"
                " ,  URM /training,  corr_factor/rho settings. "
            )

    except Exception as e:
        print(f"[URM-DBG][ERROR] {e}")



def _entropy_from_counter(counter: Counter) -> float:
    total = float(sum(counter.values()))
    if total <= 0:
        return 0.0
    p = np.array([v / total for v in counter.values()], dtype=np.float64)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())

def _gini_from_counter(counter: Counter) -> float:
    x = np.array(list(counter.values()), dtype=np.float64)
    if x.size == 0:
        return 0.0
    if np.allclose(x.sum(), 0.0):
        return 0.0
    x = np.sort(x)
    n = x.size
    cumx = np.cumsum(x)
    g = 1.0 - 2.0 * float(cumx.sum()) / (n * float(cumx[-1])) + 1.0 / n
    return float(max(0.0, min(1.0, g)))

def _top_share_from_counter(counter: Counter, topk: int = 10) -> float:
    total = float(sum(counter.values()))
    if total <= 0:
        return 0.0
    vals = sorted(counter.values(), reverse=True)
    return float(sum(vals[: max(1, int(topk))]) / total)

def _maybe_get_item_popularity(reader):
    """Best-effort: try to obtain item popularity counts from reader or reader.get_statistics().

    Returns:
      pop: dict[int->count] OR 1D np.ndarray indexed by encoded iid (len>=max_iid+1) OR None
    """
    for attr in [
        "item_popularity", "item_pop", "item_freq", "item_count", "item_cnt",
        "item_interaction_count", "item_exposure_count", "item_pv",
    ]:
        if hasattr(reader, attr):
            pop = getattr(reader, attr)
            if pop is not None:
                return pop

    try:
        stats = reader.get_statistics()
    except Exception:
        stats = None
    if isinstance(stats, dict):
        for k in [
            "item_popularity", "item_pop", "item_freq", "item_count", "item_cnt",
            "item_interaction_count", "item_exposure_count", "item_pv",
        ]:
            if k in stats and stats[k] is not None:
                return stats[k]
    return None

def _pop_get(pop, iid: int) -> float:
    if pop is None:
        return 0.0
    try:
        if isinstance(pop, dict):
            return float(pop.get(int(iid), 0.0))
        arr = np.asarray(pop)
        iid = int(iid)
        if 0 <= iid < arr.shape[0]:
            return float(arr[iid])
    except Exception:
        return 0.0
    return 0.0

def _compute_longtail_metrics(item_counter: Counter, pop, q: float = 0.2):
    """Compute long-tail share and average novelty if popularity is available.

    long-tail: items in bottom-q quantile by popularity (among items with pop>0).
    novelty: average -log2(pop/total_pop) over exposures.
    """
    if pop is None or len(item_counter) == 0:
        return None

    iids = list(item_counter.keys())
    pops = np.array([_pop_get(pop, iid) for iid in iids], dtype=np.float64)
    pops_pos = pops[pops > 0]
    if pops_pos.size == 0:
        return None

    thr = float(np.quantile(pops_pos, max(0.0, min(1.0, q))))
    tail_mask = (pops > 0) & (pops <= thr)
    tail_iids = set([int(iids[i]) for i in range(len(iids)) if bool(tail_mask[i])])

    total_exp = float(sum(item_counter.values()))
    tail_exp = float(sum(v for iid, v in item_counter.items() if int(iid) in tail_iids))
    longtail_share = tail_exp / total_exp if total_exp > 0 else 0.0

    total_pop = float(pops_pos.sum())
    novelty_sum = 0.0
    for iid, cnt in item_counter.items():
        p = _pop_get(pop, iid)
        if p <= 0:
            continue
        novelty_sum += float(cnt) * (-math.log2(p / total_pop))
    avg_novelty = novelty_sum / total_exp if total_exp > 0 else 0.0

    return {
        "longtail_q": float(q),
        "pop_threshold": thr,
        "longtail_share": float(longtail_share),
        "avg_novelty": float(avg_novelty),
    }



def run_eval(args):
    device = torch.device(args.device)
    print(f"[Info] Using device: {device}")

    env = KREnvironment_WholeSession_GPU(args)
    if hasattr(env, "set_seed"):
        env.set_seed(args.seed)
        print(f"[Seed] env.set_seed({args.seed}) called.")

    model, cfg = load_onerec_value_model(args.onerec_ckpt, device)

    iid2sid, sid2iid_map = build_iid2sid(
        env.reader, args.sid_mapping_path, cfg["sid_depth"], device
    )

    policy = OneRecValuePolicy(
        model,
        iid2sid,
        sid2iid_map,
        cfg,
        device,
        args.slate_size,
        env.reader,
        args,
    )

    rand_policy = RandomPolicy(args.slate_size, device)
    if args.random_policy:
        print('[Eval] RandomPolicy ENABLED: ignore OneRec+Value rerank, sample random items each step.')

    print(
        f"[Eval] Rerank Alpha={args.rerank_alpha}, "
        f"Beam Width={args.beam_width}, Seed={args.seed}"
    )

    observation = env.reset({"batch_size": args.episode_batch_size})

    cur_returns = torch.zeros(args.episode_batch_size, device=device)
    cur_lengths = torch.zeros(args.episode_batch_size, device=device)
    finished = 0
    all_ret, all_len = [], []

    beh_names = None
    if hasattr(env, "response_types"):
        try:
            beh_names = list(env.response_types)
        except Exception:
            beh_names = None
    if beh_names is None:
        try:
            beh_names = env.reader.get_statistics().get("feedback_type", None)
        except Exception:
            beh_names = None
    if beh_names is None and hasattr(env, "response_weights"):
        try:
            K_tmp = int(env.response_weights.shape[0])
            beh_names = [f"fb{i}" for i in range(K_tmp)]
        except Exception:
            beh_names = None
    if beh_names is None:
        beh_names = ["is_click", "long_view", "is_like", "is_comment", "is_forward", "is_follow", "is_hate"]

    K = len(beh_names)
    B = int(args.episode_batch_size)
    cur_beh_counts = torch.zeros(B, K, device=device)   # (B, K)
    cur_impr = torch.zeros(B, device=device)            # (B,)
    total_beh_counts = torch.zeros(K, device=device)    # (K,)
    total_impr = 0.0

    response_weights = env.response_weights


    do_debias = bool(getattr(args, "report_debias_metrics", False))
    cur_item_lists = [[] for _ in range(B)]
    cur_sid1_lists = [[] for _ in range(B)]
    cur_sid2_lists = [[] for _ in range(B)]
    author_key = None
    try:
        for _k in getattr(env, "candidate_item_meta", {}).keys():
            if "author" in str(_k).lower():
                author_key = _k
                break
    except Exception:
        author_key = None
    cur_author_lists = [[] for _ in range(B)]

    item_counter = Counter()
    sid1_counter = Counter()
    sid2_counter = Counter()
    author_counter = Counter()

    pop = _maybe_get_item_popularity(getattr(env, "reader", None)) if do_debias else None
    dbg_urm_step = 0

    try:
        while finished < args.num_episodes:
            try:
                cand = env.get_candidate_info(feed_dict=None)
            except Exception:
                cand = env.get_candidate_info()

            action = (rand_policy.act(observation, cand) if args.random_policy else policy.act(observation, cand))
            if getattr(args, "debug_urm", False) and dbg_urm_step < int(getattr(args, "debug_urm_steps", 1)):
                _debug_urm_once(env, observation, action, dbg_urm_step, beh_names)
                dbg_urm_step += 1

            next_obs, resp, _ = env.step({"action": action})


            if do_debias:
                try:
                    shown_iids = env.candidate_iids[action]  # (B, slate) encoded iid
                    if torch.is_tensor(shown_iids):
                        shown_iids_cpu = shown_iids.detach().cpu().numpy()
                    else:
                        shown_iids_cpu = np.asarray(shown_iids)
                    try:
                        sid_tok = iid2sid[_safe_to_tensor(shown_iids, device).long()]
                        if torch.is_tensor(sid_tok) and sid_tok.dim() == 2:
                            sid_tok = sid_tok.unsqueeze(1)
                        sid1 = sid_tok[:, :, 0]
                        sid2 = sid_tok[:, :, 1] if sid_tok.size(-1) > 1 else None
                        sid1_cpu = sid1.detach().cpu().numpy()
                        sid2_cpu = sid2.detach().cpu().numpy() if sid2 is not None else None
                    except Exception:
                        sid1_cpu, sid2_cpu = None, None

                    author_cpu = None
                    if author_key is not None:
                        try:
                            a = env.candidate_item_meta[author_key][action]
                            if torch.is_tensor(a):
                                a = a.detach().cpu().numpy()
                            else:
                                a = np.asarray(a)

                            slate_sz = int(getattr(env, "slate_size", 1))
                            if a.ndim == 1:
                                author_cpu = a
                            elif a.ndim == 2:
                                if a.shape[1] == slate_sz:
                                    author_cpu = a
                                elif a.shape[1] == 1:
                                    author_cpu = a.squeeze(1)
                                else:
                                    author_cpu = a.argmax(axis=-1)
                            elif a.ndim == 3:
                                if a.shape[-1] == 1:
                                    author_cpu = a.squeeze(-1)
                                else:
                                    author_cpu = a.argmax(axis=-1)
                        except Exception:
                            author_cpu = None


                    for bi in range(int(shown_iids_cpu.shape[0])):
                        cur_item_lists[bi].extend([int(x) for x in shown_iids_cpu[bi].reshape(-1).tolist()])
                        if sid1_cpu is not None:
                            cur_sid1_lists[bi].extend([int(x) for x in sid1_cpu[bi].reshape(-1).tolist()])
                        if sid2_cpu is not None:
                            cur_sid2_lists[bi].extend([int(x) for x in sid2_cpu[bi].reshape(-1).tolist()])
                        if author_cpu is not None:
                            cur_author_lists[bi].extend([int(x) for x in author_cpu[bi].reshape(-1).tolist()])
                except Exception as _e:
                    pass
            if getattr(args, "debug_urm", False) and dbg_urm_step <= int(getattr(args, "debug_urm_steps", 1)):
                im_dbg = resp.get("immediate_response", None)
                if im_dbg is not None:
                    if isinstance(im_dbg, np.ndarray):
                        im_t = torch.from_numpy(im_dbg).to(device).float()
                    else:
                        im_t = im_dbg.to(device).float() if torch.is_tensor(im_dbg) else torch.tensor(im_dbg, device=device).float()
                    non_bin = ((im_t - im_t.round()).abs() > 1e-6).float().mean().item()
                    print("[RESP-DBG] " + _pretty_stats("resp['immediate_response']", im_t))
                    print(f"[RESP-DBG] non_binary_frac={non_bin:.6f} (0 means pure 0/1)")
            im = resp.get("immediate_response", None)
            if im is None:
                raise RuntimeError("resp no 'immediate_response' , ")
            if isinstance(im, np.ndarray):
                im = torch.from_numpy(im)
            im = im.to(device).float()  # (B, slate, K)

            cur_impr += float(im.size(1))
            K_eff = min(K, int(im.size(2)))
            if K_eff < K:
                cur_beh_counts[:, :K_eff] += im[:, :, :K_eff].sum(dim=1)
            else:
                cur_beh_counts += im.sum(dim=1)

            rw_dict = {
                "immediate_response": resp["immediate_response"],
                "immediate_response_weight": response_weights,
            }
            step_r = get_immediate_reward(rw_dict)

            cur_returns += step_r
            cur_lengths += 1

            done = resp["done"]
            if done.any():
                idxs = torch.nonzero(done).squeeze(-1)
                for idx in idxs:
                    if finished < args.num_episodes:
                        all_ret.append(cur_returns[idx].item())
                        all_len.append(cur_lengths[idx].item())
                        finished += 1
                        total_beh_counts += cur_beh_counts[idx]
                        total_impr += float(cur_impr[idx].item())

                        if do_debias:
                            try:
                                item_counter.update(cur_item_lists[int(idx)])
                                sid1_counter.update(cur_sid1_lists[int(idx)])
                                sid2_counter.update(cur_sid2_lists[int(idx)])
                                if author_key is not None:
                                    author_counter.update(cur_author_lists[int(idx)])
                            except Exception:
                                pass
                        if finished % args.log_every == 0:
                            print(
                                f"Progress {finished}/{args.num_episodes} | "
                                f"Ret: {np.mean(all_ret):.4f} | "
                                f"Len: {np.mean(all_len):.2f}"
                            )
                cur_returns[done] = 0
                cur_lengths[done] = 0
                cur_beh_counts[done] = 0
                cur_impr[done] = 0


                if do_debias:
                    for _idx in torch.nonzero(done).squeeze(-1).tolist():
                        cur_item_lists[int(_idx)] = []
                        cur_sid1_lists[int(_idx)] = []
                        cur_sid2_lists[int(_idx)] = []
                        cur_author_lists[int(_idx)] = []
            observation = next_obs

    except KeyboardInterrupt:
        print("[Eval] Interrupted by user.")

    if all_ret:
        print("=" * 40)
        print(f"Alpha: {args.rerank_alpha}")

        total_ret = float(np.mean(all_ret))
        depth = float(np.mean(all_len))
        avg_step_r = total_ret / depth if depth > 0 else 0.0

        print(f"Total Reward: {total_ret:.4f}")
        print(f"Depth: {depth:.2f}")
        print(f"Avg Step Reward: {avg_step_r:.4f}")

        if hasattr(env, "get_report"):
            env_report = env.get_report()
            coverage = float(env_report.get("coverage", 0.0))
            ild = float(env_report.get("ILD", 0.0))
            print(f"Coverage: {coverage:.2f}")
            print(f"ILD: {ild:.4f}")

            print("Table-style metrics:")
            print(f"Depth: {depth:.2f}")
            print(f"Average reward: {avg_step_r:.4f}")
            print(f"Total reward: {total_ret:.4f}")
            print(f"Coverage: {coverage:.2f}")
            print(f"ILD: {ild:.4f}")

            print("Behavior rates (count / impressions):")
            if total_impr <= 0:
                print("  [WARN] total_impr=0")
            else:
                for k, name in enumerate(beh_names):
                    cnt = float(total_beh_counts[k].item())
                    rate = 100.0 * cnt / total_impr
                    print(f"  {name}: {int(round(cnt))}/{int(round(total_impr))} ({rate:.4f}%)")


        if bool(getattr(args, "report_debias_metrics", False)):
            try:
                stats = {}
                n_unique = len(item_counter)
                total_exp = float(sum(item_counter.values()))
                try:
                    n_item_total = int(getattr(env, "n_candidate", 0))
                    st = env.reader.get_statistics() if hasattr(env, "reader") else {}
                    if isinstance(st, dict) and st.get("n_item", None) is not None:
                        n_item_total = int(st["n_item"])
                except Exception:
                    n_item_total = int(n_unique)
                cov = float(n_unique) / float(max(1, n_item_total))
                ent = _entropy_from_counter(item_counter)
                ent_norm = ent / float(max(1e-12, math.log(max(2, n_unique)))) if n_unique > 1 else 0.0
                gini = _gini_from_counter(item_counter)
                top10_share = _top_share_from_counter(item_counter, topk=10)
                stats["item"] = {
                    "unique": int(n_unique),
                    "total_exposures": float(total_exp),
                    "coverage": float(cov),
                    "entropy": float(ent),
                    "entropy_norm": float(ent_norm),
                    "gini": float(gini),
                    "top10_share": float(top10_share),
                }

                for name, ctr in [("sid1", sid1_counter), ("sid2", sid2_counter)]:
                    if len(ctr) > 0:
                        u = len(ctr)
                        e = _entropy_from_counter(ctr)
                        e_norm = e / float(max(1e-12, math.log(max(2, u)))) if u > 1 else 0.0
                        stats[name] = {
                            "unique": int(u),
                            "entropy": float(e),
                            "entropy_norm": float(e_norm),
                            "gini": float(_gini_from_counter(ctr)),
                            "top10_share": float(_top_share_from_counter(ctr, topk=10)),
                        }

                if author_key is not None and len(author_counter) > 0:
                    u = len(author_counter)
                    e = _entropy_from_counter(author_counter)
                    e_norm = e / float(max(1e-12, math.log(max(2, u)))) if u > 1 else 0.0
                    stats["author"] = {
                        "key": str(author_key),
                        "unique": int(u),
                        "entropy": float(e),
                        "entropy_norm": float(e_norm),
                        "gini": float(_gini_from_counter(author_counter)),
                        "top10_share": float(_top_share_from_counter(author_counter, topk=10)),
                    }

                lt = _compute_longtail_metrics(item_counter, pop, q=float(getattr(args, "longtail_q", 0.2)))
                if lt is not None:
                    stats["longtail"] = lt
                else:
                    once = sum(1 for v in item_counter.values() if int(v) <= 1)
                    stats["longtail_proxy"] = {
                        "": "popularity unavailable; proxy uses exposure counts in this eval run",
                        "share_items_exposed_once": float(once) / float(max(1, len(item_counter))),
                    }

                print("=" * 40)
                print("Debias distribution metrics (evaluation-only):")
                print(f"[Item] unique={stats['item']['unique']} total_exp={stats['item']['total_exposures']:.0f} "
                      f"coverage={stats['item']['coverage']:.6f} entropy={stats['item']['entropy']:.4f} "
                      f"entropy_norm={stats['item']['entropy_norm']:.4f} gini={stats['item']['gini']:.4f} "
                      f"top10_share={stats['item']['top10_share']:.4f}")
                if "sid1" in stats:
                    s = stats["sid1"]
                    print(f"[SID1] unique={s['unique']} entropy={s['entropy']:.4f} entropy_norm={s['entropy_norm']:.4f} "
                          f"gini={s['gini']:.4f} top10_share={s['top10_share']:.4f}")
                if "sid2" in stats:
                    s = stats["sid2"]
                    print(f"[SID2] unique={s['unique']} entropy={s['entropy']:.4f} entropy_norm={s['entropy_norm']:.4f} "
                          f"gini={s['gini']:.4f} top10_share={s['top10_share']:.4f}")
                if "author" in stats:
                    s = stats["author"]
                    print(f"[Author] key={s['key']} unique={s['unique']} entropy={s['entropy']:.4f} entropy_norm={s['entropy_norm']:.4f} "
                          f"gini={s['gini']:.4f} top10_share={s['top10_share']:.4f}")
                if "longtail" in stats:
                    s = stats["longtail"]
                    print(f"[LongTail] q={s['longtail_q']:.2f} thr={s['pop_threshold']:.2f} longtail_share={s['longtail_share']:.4f} "
                          f"avg_novelty={s['avg_novelty']:.4f}")
                else:
                    s = stats["longtail_proxy"]
                    print(f"[LongTail-Proxy] {s['']} | share_items_exposed_once={s['share_items_exposed_once']:.4f}")
                print("=" * 40)

            except Exception as e:
                print(f"[DebiasMetrics][WARN] failed to compute metrics: {e}")

        if hasattr(policy, "report_stats"):
            policy.report_stats()

        print("=" * 40)


if __name__ == "__main__":
    args = parse_args()
    utils.set_random_seed(args.seed)
    run_eval(args)
