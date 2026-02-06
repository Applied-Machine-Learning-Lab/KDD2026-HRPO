
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HRPO / RRPO Online Continuation (Simulator)
-------------------------------------------
One-file orchestrator that:
  1) Collects fresh interaction records by rolling out a snapshot policy in a simulator env.
  2) Incrementally refreshes HRPO prefix tables from the new records.
  3) Runs a limited number of RRPO/GRPO token-level updates, periodically syncing the snapshot policy.

This script is designed to plug into your existing KuaiSim/KuaiRand codebase:
- Environment: KREnvironment_WholeSession_GPU
- Offline trainer utilities: train_hrpo_rrpo_ntp.py (HRPOTable, constrained_beam_search_sid, rrpo_step_hrpo_token, etc.)

Usage example (adjust paths/args for your repo):
  python hrpo_online.py \
    --init_ckpt checkpoints/hrpo_offline.pt \
    --sid_mapping_path data/video_sid_mapping.csv \
    --user_feat_path data/user_features.csv \
    --base_log_paths data/offline_log.csv \
    --hrpo_table_path checkpoints/hrpo_table.pkl \
    --output_dir checkpoints/online_runs \
    --Q 10 --collect_steps 2000 --train_steps 1000 \
    --reward_weights "is_click:1,long_view:2,is_like:3,is_hate:-2" \
    ... (env args, see --help)

Notes:
- This script assumes your env returns observations in the same structure as KREnvironment_WholeSession_GPU:
    obs["user_profile"]["user_id"], obs["user_history"]["history"], obs["user_history"]["history_length"]
- It logs per-item rows (one row per slate position) with time_ms/session_id/position plus feedback columns.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import pickle
import random
import time
from collections import defaultdict
from copy import deepcopy
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch



def _try_import_env():
    try:
        from env.KREnvironment_WholeSession_GPU import KREnvironment_WholeSession_GPU  # type: ignore
        return KREnvironment_WholeSession_GPU
    except Exception:
        from KREnvironment_WholeSession_GPU import KREnvironment_WholeSession_GPU  # type: ignore
        return KREnvironment_WholeSession_GPU


def _try_import_offline_utils():
    """
    We reuse your offline RRPO implementation to avoid re-implementing training details.
    """
    try:
        import train_hrpo_rrpo_ntp as offline  # type: ignore
        return offline
    except Exception as e:
        raise ImportError(
            "Cannot import train_hrpo_rrpo_ntp.py. "
            "Put the trainer script in the same folder or ensure it is in PYTHONPATH."
        ) from e



def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def now_str() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def parse_reward_weights(s: str) -> Dict[str, float]:
    """
    Support:
      - JSON: '{"is_click": 1, "long_view": 2}'
      - kv pairs: "is_click:1,long_view:2,is_hate:-2"
    """
    s = (s or "").strip()
    if not s:
        return {}
    if s.startswith("{"):
        obj = json.loads(s)
        return {str(k): float(v) for k, v in obj.items()}
    out: Dict[str, float] = {}
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"Bad reward weight token: {part!r}. Use k:v or JSON.")
        k, v = part.split(":", 1)
        out[k.strip()] = float(v.strip())
    return out


def safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default



@dataclasses.dataclass
class SidMapping:
    """
    video_id <-> SID path (sid_1..sid_L)
    """
    sid_depth: int
    video2sid: Dict[int, Tuple[int, ...]]
    sid2video: Dict[Tuple[int, ...], int]
    sid_matrix: np.ndarray  # shape [max_video_id+1, sid_depth], row 0 = zeros

    @staticmethod
    def load(path: str, video_id_col: str = "video_id") -> "SidMapping":
        df = pd.read_csv(path)
        sid_cols = [c for c in df.columns if c.startswith("sid_")]
        sid_cols = sorted(sid_cols, key=lambda x: int(x.split("_")[1]))
        if not sid_cols:
            raise ValueError(f"No sid_* columns found in sid mapping file: {path}")
        sid_depth = len(sid_cols)

        video2sid: Dict[int, Tuple[int, ...]] = {}
        sid2video: Dict[Tuple[int, ...], int] = {}

        vids = df[video_id_col].astype(int).to_numpy()
        sids = df[sid_cols].astype(int).to_numpy()

        for vid, row in zip(vids, sids):
            tup = tuple(int(t) for t in row.tolist())
            video2sid[int(vid)] = tup
            sid2video[tup] = int(vid)

        max_vid = int(np.max(vids)) if len(vids) else 0
        sid_matrix = np.zeros((max_vid + 1, sid_depth), dtype=np.int64)
        for vid, tup in video2sid.items():
            if 0 <= vid <= max_vid:
                sid_matrix[vid, :] = np.asarray(tup, dtype=np.int64)

        return SidMapping(
            sid_depth=sid_depth,
            video2sid=video2sid,
            sid2video=sid2video,
            sid_matrix=sid_matrix,
        )




def build_trie_from_sid_mapping_fallback(sid_mapping_path: str, sid_depth: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build (trie_mask, trie_next) from sid mapping CSV.
    - trie_mask: [num_nodes, num_classes] bool
    - trie_next: [num_nodes, num_classes] long (child node id, 0 if none)
    Node 0 is root.
    """
    df = pd.read_csv(sid_mapping_path)
    sid_cols = [c for c in df.columns if c.startswith('sid_')]
    sid_cols = sorted(sid_cols, key=lambda x: int(x.split('_')[1]))
    if len(sid_cols) != sid_depth:
        sid_depth = len(sid_cols)
    paths = df[sid_cols].astype(int).to_numpy().tolist()
    max_tok = 0
    for p in paths:
        for t in p:
            if t > max_tok:
                max_tok = int(t)
    num_classes = max_tok + 1  # include 0

    children: List[Dict[int, int]] = []
    children.append({})  # root
    def new_node() -> int:
        children.append({})
        return len(children) - 1

    for p in paths:
        node = 0
        for t in p[:sid_depth]:
            t = int(t)
            if t not in children[node]:
                children[node][t] = new_node()
            node = children[node][t]

    num_nodes = len(children)
    trie_mask = torch.zeros((num_nodes, num_classes), dtype=torch.bool)
    trie_next = torch.zeros((num_nodes, num_classes), dtype=torch.long)
    for node, ch in enumerate(children):
        for tok, nxt in ch.items():
            if 0 <= tok < num_classes:
                trie_mask[node, tok] = True
                trie_next[node, tok] = int(nxt)

    return trie_mask, trie_next

def _infer_feedback_cols(response_types: Sequence[str]) -> List[str]:
    return [str(x) for x in response_types]


def model_dim_from_size(model_size: str):
    model_size = str(model_size).lower()
    if model_size in ["mini", "small"]:
        return 128, 4
    if model_size in ["base", "medium"]:
        return 256, 8
    if model_size in ["large"]:
        return 512, 8
    return 256, 8


def compute_item_rewards(
    immediate_response: torch.Tensor,
    response_types: Sequence[str],
    reward_w: Dict[str, float],
) -> torch.Tensor:
    """
    immediate_response: [B, slate, R] (0/1 or 0/1/..)
    return: [B, slate] float
    """
    if not reward_w:
        return torch.zeros(immediate_response.shape[0], immediate_response.shape[1], device=immediate_response.device)

    idx = {name: i for i, name in enumerate(response_types)}
    rew = torch.zeros(immediate_response.shape[0], immediate_response.shape[1], device=immediate_response.device, dtype=torch.float32)
    for name, w in reward_w.items():
        if name not in idx:
            continue
        rew = rew + float(w) * immediate_response[..., idx[name]].float()
    return rew


def hrpo_incremental_update_from_df(
    hrpo_table,
    df_new: pd.DataFrame,
    sidmap: SidMapping,
    reward_w: Dict[str, float],
    user_id_col: str = "user_id",
    video_id_col: str = "video_id",
) -> None:
    """
    Update HRPO table in-place using newly collected (u,v,b) rows.

    Assumptions:
      - `hrpo_table` contains:
            hrpo_table.uid2ctx: Dict[int,int]
            hrpo_table.prefix_sum: Dict[(ctx,prefix_tuple), float]
            hrpo_table.prefix_count: Dict[(ctx,prefix_tuple), int]
            hrpo_table.ctx_sum: Dict[ctx, float]
            hrpo_table.ctx_count: Dict[ctx, int]
            hrpo_table.global_mean: float
      - Prefix keys are python tuples.
    """
    if df_new.empty:
        return

    fb_cols = [c for c in df_new.columns if c in reward_w.keys()]
    if not fb_cols:
        fb_cols = [c for c in df_new.columns if c.startswith("is_") or c in ("click", "long_view")]

    r = np.zeros(len(df_new), dtype=np.float32)
    for k, w in reward_w.items():
        if k in df_new.columns:
            r += float(w) * df_new[k].astype(np.float32).to_numpy()

    uids = df_new[user_id_col].astype(int).to_numpy()
    vids = df_new[video_id_col].astype(int).to_numpy()

    total_count_old = int(sum(hrpo_table.ctx_count.values())) if getattr(hrpo_table, "ctx_count", None) else 0
    total_sum_old = float(sum(hrpo_table.ctx_sum.values())) if getattr(hrpo_table, "ctx_sum", None) else float(hrpo_table.global_mean * max(total_count_old, 1))

    if getattr(hrpo_table, "prefix_sum", None) is None:
        hrpo_table.prefix_sum = {}
    if getattr(hrpo_table, "prefix_count", None) is None:
        hrpo_table.prefix_count = {}
    if getattr(hrpo_table, "ctx_sum", None) is None:
        hrpo_table.ctx_sum = {}
    if getattr(hrpo_table, "ctx_count", None) is None:
        hrpo_table.ctx_count = {}
    if getattr(hrpo_table, "uid2ctx", None) is None:
        hrpo_table.uid2ctx = {}

    prefix_sum = hrpo_table.prefix_sum
    prefix_count = hrpo_table.prefix_count
    ctx_sum = hrpo_table.ctx_sum
    ctx_count = hrpo_table.ctx_count
    uid2ctx = hrpo_table.uid2ctx

    sid_depth = sidmap.sid_depth

    n_new = 0
    sum_new = 0.0

    for uid, vid, rr in zip(uids, vids, r):
        uid_i = int(uid)
        vid_i = int(vid)
        rr_f = float(rr)

        sid = sidmap.video2sid.get(vid_i, None)
        if sid is None:
            continue

        ctx = int(uid2ctx.get(uid_i, 0))
        ctx_sum[ctx] = float(ctx_sum.get(ctx, 0.0) + rr_f)
        ctx_count[ctx] = int(ctx_count.get(ctx, 0) + 1)

        for t in range(1, sid_depth + 1):
            pref = tuple(sid[:t])
            key_c = (ctx, pref)
            key_g = (-1, pref)  # global backoff bucket

            prefix_sum[key_c] = float(prefix_sum.get(key_c, 0.0) + rr_f)
            prefix_count[key_c] = int(prefix_count.get(key_c, 0) + 1)

            prefix_sum[key_g] = float(prefix_sum.get(key_g, 0.0) + rr_f)
            prefix_count[key_g] = int(prefix_count.get(key_g, 0) + 1)

        n_new += 1
        sum_new += rr_f

    total_count_new = total_count_old + n_new
    total_sum_new = total_sum_old + sum_new
    if total_count_new > 0:
        hrpo_table.global_mean = float(total_sum_new / total_count_new)


def obs_to_model_inputs(
    obs: Dict[str, Any],
    sidmap: SidMapping,
    user_feat_cols: Sequence[str],
    max_hist_len: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build:
      uids:      [B] long (cpu)
      user_feat: [B, D] float (device)
      hist_sid:  [B, H, L] long (device)
      hist_len:  [B] long (device)
    """
    user_profile = obs["user_profile"]
    user_history = obs["user_history"]

    uids = user_profile["user_id"]
    if isinstance(uids, torch.Tensor):
        uids_t = uids.detach().cpu().long()
    else:
        uids_t = torch.as_tensor(uids, dtype=torch.long)

    feat_list = []
    for col in user_feat_cols:
        key = f"uf_{col}"
        if key not in user_profile:
            continue
        v = user_profile[key]
        if not isinstance(v, torch.Tensor):
            v = torch.as_tensor(v)
        feat_list.append(v.float())
    if not feat_list:
        raise KeyError(
            "No user feature vectors found in obs['user_profile']. "
            "Check user_feat_cols or your env observation keys."
        )
    user_feat = torch.cat(feat_list, dim=1).to(device)

    hist_item = user_history["history"]
    hist_len = user_history.get("history_length", None)
    if hist_len is None:
        if isinstance(hist_item, torch.Tensor):
            hist_len = (hist_item != 0).sum(dim=1)
        else:
            hist_len = np.sum(np.asarray(hist_item) != 0, axis=1)

    if not isinstance(hist_item, torch.Tensor):
        hist_item = torch.as_tensor(hist_item, dtype=torch.long)
    if not isinstance(hist_len, torch.Tensor):
        hist_len = torch.as_tensor(hist_len, dtype=torch.long)

    if hist_item.shape[1] > max_hist_len:
        hist_item = hist_item[:, -max_hist_len:]
    hist_item_cpu = hist_item.detach().cpu().numpy()
    max_vid = sidmap.sid_matrix.shape[0] - 1
    hist_item_cpu = np.clip(hist_item_cpu, 0, max_vid)

    hist_sid_np = sidmap.sid_matrix[hist_item_cpu]  # [B,H,L]
    hist_sid = torch.from_numpy(hist_sid_np).long().to(device)

    hist_len = torch.clamp(hist_len, 0, max_hist_len).to(device)

    return uids_t, user_feat, hist_sid, hist_len


@torch.no_grad()
def policy_generate_slate_actions(
    offline_utils,
    model,
    obs: Dict[str, Any],
    sidmap: SidMapping,
    trie_mask: torch.Tensor,
    trie_next: torch.Tensor,
    user_feat_cols: Sequence[str],
    max_hist_len: int,
    slate_size: int,
    beam_width: int,
    temperature: float,
    device: torch.device,
    itemid2candidx: Dict[int, int],
    sid_depth: int,
) -> torch.Tensor:
    """
    Return action indices: [B, slate_size] long (cpu)
    """
    uids_t, user_feat, hist_sid, hist_len = obs_to_model_inputs(
        obs=obs,
        sidmap=sidmap,
        user_feat_cols=user_feat_cols,
        max_hist_len=max_hist_len,
        device=device,
    )

    gen_sid = offline_utils.constrained_beam_search_sid(
        model=model,
        user_feat=user_feat,
        hist_sid=hist_sid,
        hist_len=hist_len,
        sid_depth=sid_depth,
        trie_mask=trie_mask,
        trie_next=trie_next,
        beam_width=beam_width,
        temperature=temperature,
    )  # torch.LongTensor

    if isinstance(gen_sid, tuple):
        gen_sid = gen_sid[0]

    gen_sid_cpu = gen_sid.detach().cpu().numpy()  # [B,W,L]
    B, W, L = gen_sid_cpu.shape

    actions = np.zeros((B, slate_size), dtype=np.int64)
    for b in range(B):
        chosen: List[int] = []
        for k in range(W):
            tup = tuple(int(x) for x in gen_sid_cpu[b, k, :].tolist())
            vid = sidmap.sid2video.get(tup, None)
            if vid is None:
                continue
            if vid in chosen:
                continue
            chosen.append(int(vid))
            if len(chosen) >= slate_size:
                break
        if len(chosen) < slate_size:
            cand_ids = list(itemid2candidx.keys())
            while len(chosen) < slate_size:
                chosen.append(int(random.choice(cand_ids)))
        for j, vid in enumerate(chosen[:slate_size]):
            actions[b, j] = int(itemid2candidx.get(int(vid), random.randrange(len(itemid2candidx))))
    return torch.from_numpy(actions).long()


def collect_rollouts_to_df(
    offline_utils,
    env,
    model_snapshot,
    sidmap: SidMapping,
    trie_mask: torch.Tensor,
    trie_next: torch.Tensor,
    reward_w: Dict[str, float],
    user_feat_cols: Sequence[str],
    max_hist_len_model: int,
    collect_steps: int,
    slate_size: int,
    beam_width: int,
    temperature: float,
    device: torch.device,
    reset_every: int = 200,
    sid_depth : int = 4,
) -> pd.DataFrame:
    """
    Run env for collect_steps steps, return per-item rows DataFrame.
    """
    cand_iids = getattr(env, "candidate_iids", None)
    if cand_iids is None:
        raise AttributeError("env has no candidate_iids; cannot map item_id->candidate index.")

    if isinstance(cand_iids, torch.Tensor):
        cand_iids_list = cand_iids.detach().cpu().long().tolist()
    else:
        cand_iids_list = list(map(int, cand_iids))
    itemid2candidx = {int(iid): int(i) for i, iid in enumerate(cand_iids_list)}

    response_types = getattr(env, "response_types", None)
    if response_types is None:
        response_types = ["is_click", "long_view", "is_like", "is_comment", "is_forward", "is_follow", "is_hate"]
    response_types = list(response_types)
    fb_cols = _infer_feedback_cols(response_types)

    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]

    episode_id = 0
    t_global = 0

    rows: List[Dict[str, Any]] = []

    model_snapshot.eval()

    while t_global < collect_steps:
        if reset_every > 0 and (t_global % reset_every == 0) and t_global > 0:
            obs = env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]
            episode_id += 1

        action_idx = policy_generate_slate_actions(
            offline_utils=offline_utils,
            model=model_snapshot,
            obs=obs,
            sidmap=sidmap,
            trie_mask=trie_mask,
            trie_next=trie_next,
            user_feat_cols=user_feat_cols,
            max_hist_len=max_hist_len_model,
            slate_size=slate_size,
            beam_width=max(beam_width, slate_size),
            temperature=temperature,
            device=device,
            itemid2candidx=itemid2candidx,
            sid_depth = sid_depth,
        )  # [B, slate] CPU

        step_dict = {"action": action_idx.to(getattr(env, "device", device))}
        ret = env.step(step_dict)
        if isinstance(ret, tuple) and len(ret) == 3:
            next_obs, response_dict, done = ret
        elif isinstance(ret, tuple) and len(ret) == 4:
            next_obs, response_dict, done, _info = ret
        else:
            raise RuntimeError(f"Unexpected env.step return: {type(ret)} / {getattr(ret, '__len__', lambda: 'NA')()}")

        immediate = response_dict.get("immediate_response", None)
        if immediate is None:
            raise KeyError("response_dict has no 'immediate_response' key.")
        if not isinstance(immediate, torch.Tensor):
            immediate = torch.as_tensor(immediate)

        user_id = obs["user_profile"]["user_id"]
        if isinstance(user_id, torch.Tensor):
            user_id = user_id.detach().cpu().long().numpy()
        else:
            user_id = np.asarray(user_id, dtype=np.int64)

        if isinstance(cand_iids, torch.Tensor):
            cand_tensor_cpu = cand_iids.detach().cpu().long()
        else:
            cand_tensor_cpu = torch.as_tensor(cand_iids_list, dtype=torch.long)

        action_idx_cpu = action_idx.detach().cpu().long()
        slate_item_ids = cand_tensor_cpu[action_idx_cpu]  # [B, slate]
        slate_item_ids_np = slate_item_ids.numpy()

        immediate_cpu = immediate.detach().cpu().numpy().astype(np.int64)  # [B,slate,R]
        B = immediate_cpu.shape[0]
        for b in range(B):
            for j in range(slate_size):
                row = {
                    "user_id": int(user_id[b]),
                    "video_id": int(slate_item_ids_np[b, j]),
                    "time_ms": int(t_global),
                    "session_id": int(episode_id),
                    "position": int(j),
                }
                for k, name in enumerate(fb_cols):
                    if k < immediate_cpu.shape[2]:
                        row[name] = int(immediate_cpu[b, j, k])
                rows.append(row)

        obs = next_obs
        t_global += 1

        if isinstance(done, torch.Tensor):
            done_any = bool(done.any().item())
        else:
            done_any = bool(np.any(done))
        if done_any:
            obs = env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]
            episode_id += 1

    df = pd.DataFrame(rows)
    return df



def infinite_loader(dataloader: torch.utils.data.DataLoader):
    while True:
        for batch in dataloader:
            yield batch


def train_steps_rrpo(
    offline_utils,
    model: torch.nn.Module,
    hrpo_table,
    trie_mask: torch.Tensor,
    trie_next: torch.Tensor,
    log_paths: List[str],
    sid_mapping_path: str,
    user_feat_path: str,
    device: torch.device,
    *,
    batch_size: int,
    num_workers: int,
    train_steps: int,
    lr: float,
    wd: float,
    beam_width: int,
    temperature: float,
    clip_eps: float,
    kl_coef: float,
    sft_coef: float,
    sync_old_every: int,
    label_col: str,
    max_hist_len: int,
    sid_depth: int,
    gamma: float,
) -> Dict[str, float]:
    """
    Build dataset from `log_paths`, then run `train_steps` GRPO/RRPO updates.
    Returns averaged metrics.
    """
    ds = offline_utils.KRPureValueDatasetWithUID(
        log_paths=log_paths,
        sid_mapping_path=sid_mapping_path,
        user_feat_path=user_feat_path,
        label_col=label_col,
        gamma=gamma,
        max_hist_len=max_hist_len,
        sid_depth=sid_depth,
    )
    if len(ds) == 0:
        return {"train_loss": 0.0, "kl": 0.0, "pg_loss": 0.0, "vf_loss": 0.0}

    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    it = infinite_loader(dl)

    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    model_old = deepcopy(model).eval()
    for p in model_old.parameters():
        p.requires_grad_(False)

    ref_model = None
    if kl_coef > 0:
        ref_model = deepcopy(model).eval()
        for p in ref_model.parameters():
            p.requires_grad_(False)

    model.train()
    agg = defaultdict(float)
    n = 0
    step_fn = getattr(offline_utils, "rrpo_step_hrpo_token", None)
    if step_fn is None:
        raise AttributeError("offline trainer must provide rrpo_step_hrpo_token")

    for step in range(1, train_steps + 1):
        batch = next(it)
        batch = offline_utils.move_batch_to_device(batch, device=device) if hasattr(offline_utils, "move_batch_to_device") else batch

        optim.zero_grad(set_to_none=True)
        loss, stats = step_fn(
            model=model,
            old_model=model_old,
            ref_model=ref_model,
            hrpo=[hrpo_table],
            batch=batch,
            device=device,
            trie_mask=trie_mask,
            trie_next=trie_next,
            group_size=beam_width,
            clip_eps=clip_eps,
            kl_coef=kl_coef,
            sft_coef=sft_coef,
            reward_scale=30.0,
            hrpo_weights=[1.0],
            global_step=step,
        )
        loss.backward()
        optim.step()

        if isinstance(stats, dict):
            for k, v in stats.items():
                try:
                    agg[k] += float(v)
                except Exception:
                    pass
        n += 1

        if sync_old_every > 0 and (step % sync_old_every == 0):
            model_old = deepcopy(model).eval()
            for p in model_old.parameters():
                p.requires_grad_(False)

    out = {k: (v / max(n, 1)) for k, v in agg.items()}
    return out



def build_parser(KREnvCls) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("HRPO Online Continuation in Simulator")

    p.add_argument("--init_ckpt", type=str, required=True, help="Initial policy checkpoint (offline HRPO).")
    p.add_argument("--sid_mapping_path", type=str, required=True, help="video_id -> sid_1..sid_L mapping CSV.")
    p.add_argument("--user_feat_path", type=str, required=True, help="User feature CSV used by offline training.")
    p.add_argument("--base_log_paths", type=str, default="", help="Comma-separated offline log CSV paths for training base.")
    p.add_argument("--hrpo_table_path", type=str, default="", help="Initial HRPO table pickle. If empty, start fresh.")
    p.add_argument("--output_dir", type=str, required=True, help="Output directory for online checkpoints/logs.")

    p.add_argument("--Q", type=int, default=10, help="Number of online rounds.")
    p.add_argument("--collect_steps", type=int, default=2000, help="Env steps per round.")
    p.add_argument("--train_steps", type=int, default=1000, help="Gradient steps per round.")
    p.add_argument("--reset_every", type=int, default=200, help="Force env.reset every N steps during collect (0 disables).")

    p.add_argument("--beam_width", type=int, default=24, help="Beam width for constrained decoding (>= slate_size recommended).")
    p.add_argument("--temperature", type=float, default=1.0, help="Decoding temperature for exploration.")

    p.add_argument("--batch_size", type=int, default=128, help="Train batch size.")
    p.add_argument("--num_workers", type=int, default=4, help="Dataloader workers.")
    p.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")
    p.add_argument("--wd", type=float, default=0.0, help="Weight decay.")
    p.add_argument("--clip_eps", type=float, default=0.2, help="PPO clip epsilon.")
    p.add_argument("--kl_coef", type=float, default=0.0, help="KL penalty coefficient.")
    p.add_argument("--sft_coef", type=float, default=0.0, help="Optional SFT loss coefficient (0 disables).")
    p.add_argument("--sync_old_every", type=int, default=200, help="Sync snapshot policy every N gradient steps during train.")

    p.add_argument("--model_size", type=str, default=None,
                help="mini/base/large. If set, overrides hid_dim/nhead.")
    p.add_argument("--hid_dim", type=int, default=None,
                    help="override hidden dim (e.g., 128).")
    p.add_argument("--nhead", type=int, default=None,
                    help="override nhead (e.g., 4).")


    p.add_argument("--label_col", type=str, default="is_click", help="Supervision label column for dataset sampling.")
    p.add_argument("--gamma", type=float, default=0.99, help="Discount for RTG if used by dataset.")
    p.add_argument("--max_hist_len_model", type=int, default=50, help="Max history length for model input/encoding.")
    p.add_argument("--max_hist_len_train", type=int, default=50, help="Max history length used in training dataset.")

    p.add_argument("--reward_weights", type=str, default="is_click:1", help="Reward weights for HRPO table update, JSON or k:v,...")

    p.add_argument(
        "--user_feat_cols",
        type=str,
        default="user_active_degree,is_live_streamer,is_video_author,follow_user_num_range,fans_user_num_range,"
                "friend_user_num_range,register_days_range,onehot_feat0,onehot_feat1,onehot_feat6,onehot_feat9,"
                "onehot_feat10,onehot_feat11",
        help="Comma-separated user feature columns order (without uf_ prefix).",
    )

    p.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    p.add_argument("--seed", type=int, default=2026)

    try:
        KREnvCls.parse_model_args(p)  # type: ignore
    except Exception:
        pass

    return p


def load_checkpoint_to_model(model: torch.nn.Module, ckpt_path: str, map_location: str = "cpu") -> None:
    ckpt = torch.load(ckpt_path, map_location=map_location)
    if isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[warn] missing keys: {missing[:10]}{'...' if len(missing)>10 else ''}")
    if unexpected:
        print(f"[warn] unexpected keys: {unexpected[:10]}{'...' if len(unexpected)>10 else ''}")


def main() -> None:
    KREnvCls = _try_import_env()
    offline_utils = _try_import_offline_utils()

    parser = build_parser(KREnvCls)
    args = parser.parse_args()

    seed_everything(args.seed)
    ensure_dir(args.output_dir)

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")

    reward_w = parse_reward_weights(args.reward_weights)

    sidmap = SidMapping.load(args.sid_mapping_path, video_id_col="video_id")
    sid_depth = sidmap.sid_depth

    try:
        trie_mask, trie_next = offline_utils.build_trie_from_sid_mapping(
            sid_mapping_path=args.sid_mapping_path,
            sid_depth=sid_depth,
        )
    except Exception:
        trie_mask, trie_next = build_trie_from_sid_mapping_fallback(
            sid_mapping_path=args.sid_mapping_path,
            sid_depth=sid_depth,
        )
    trie_mask = trie_mask.to(device)
    trie_next = trie_next.to(device)

    env = KREnvCls(args)
    if hasattr(env, "device"):
        env.device = device  # best-effort
    if hasattr(env, "to"):
        try:
            env.to(device)
        except Exception:
            pass

    if hasattr(offline_utils, "build_model_from_args"):
        model = offline_utils.build_model_from_args(
            args=args,
            sid_depth=sid_depth,
            num_classes=trie_mask.shape[-1],
            max_hist_len=args.max_hist_len_model,
        ).to(device)
    else:
        try:
            from model.onerec_value import OneRecWithValue  # type: ignore
        except Exception:
            from onerec_value import OneRecWithValue  # type: ignore

        user_feat_dim = None
        try:
            ufd = getattr(env, "observation_space", {}).get("user_feature_dims", {})
            cols = [c.strip() for c in args.user_feat_cols.split(",") if c.strip()]
            user_feat_dim = int(sum(int(ufd.get(c, 0)) for c in cols))
        except Exception:
            user_feat_dim = 0

        hid_dim = getattr(args, "hid_dim", 256)
        nhead = getattr(args, "nhead", 8)
        num_decoder_block = getattr(args, "num_decoder_block", 4)

        hid_dim = args.hid_dim
        nhead = args.nhead
        
        if hid_dim is None or nhead is None:
            if args.model_size is not None:
                hid_dim2, nhead2 = model_dim_from_size(args.model_size)
                hid_dim = hid_dim if hid_dim is not None else hid_dim2
                nhead  = nhead  if nhead  is not None else nhead2
            else:
                hid_dim = hid_dim if hid_dim is not None else 128
                nhead  = nhead  if nhead  is not None else 4

        
        model = OneRecWithValue(
            user_feat_dim=user_feat_dim,
            num_classes=int(trie_mask.shape[-1]),
            sid_depth=sid_depth,
            max_hist_len=args.max_hist_len_model,
            hid_dim=hid_dim,
            nhead=nhead,
            num_decoder_block=num_decoder_block,
        ).to(device)

    load_checkpoint_to_model(model, args.init_ckpt, map_location="cpu")
    model.eval()

    table_cls = getattr(offline_utils, "HRPOTable", None)
    if table_cls is None:
        raise AttributeError("offline trainer must provide HRPOTable")

    hrpo_table_path = args.hrpo_table_path
    if hrpo_table_path and os.path.isfile(hrpo_table_path):
        hrpo_table = table_cls.load(hrpo_table_path)
        print(f"[info] loaded HRPO table: {hrpo_table_path}")
    else:
        hrpo_table = table_cls(
            uid2ctx={},
            prefix_sum={},
            prefix_count={},
            ctx_sum={},
            ctx_count={},
            global_mean=0.0,
            min_prefix_count=getattr(args, "min_prefix_count", 1),
            smoothing_alpha=getattr(args, "smoothing_alpha", 0.0),
        )
        print("[info] created empty HRPO table")

    base_logs = [p for p in (args.base_log_paths.split(",") if args.base_log_paths else []) if p.strip()]
    online_logs: List[str] = []

    policy_snapshot = deepcopy(model).to(device).eval()
    for p in policy_snapshot.parameters():
        p.requires_grad_(False)

    user_feat_cols = [c.strip() for c in args.user_feat_cols.split(",") if c.strip()]

    for q in range(1, args.Q + 1):
        print(f"\n========== [Round {q}/{args.Q}] ==========")

        df_new = collect_rollouts_to_df(
            offline_utils=offline_utils,
            env=env,
            model_snapshot=policy_snapshot,
            sidmap=sidmap,
            trie_mask=trie_mask,
            trie_next=trie_next,
            reward_w=reward_w,
            user_feat_cols=user_feat_cols,
            max_hist_len_model=args.max_hist_len_model,
            collect_steps=args.collect_steps,
            slate_size=args.slate_size,
            beam_width=args.beam_width,
            temperature=args.temperature,
            device=device,
            reset_every=args.reset_every,
            sid_depth = sid_depth,
        )

        log_path = os.path.join(args.output_dir, f"online_round_{q:03d}.csv")
        df_new.to_csv(log_path, index=False)
        online_logs.append(log_path)
        print(f"[collect] saved {len(df_new)} rows -> {log_path}")

        hrpo_incremental_update_from_df(
            hrpo_table=hrpo_table,
            df_new=df_new,
            sidmap=sidmap,
            reward_w=reward_w,
            user_id_col="user_id",
            video_id_col="video_id",
        )
        hrpo_path = os.path.join(args.output_dir, f"hrpo_table_round_{q:03d}.pkl")
        hrpo_table.save(hrpo_path)
        print(f"[table] updated & saved -> {hrpo_path} (global_mean={getattr(hrpo_table, 'global_mean', 0.0):.6f})")

        train_logs = base_logs + online_logs  # simplest: train on all seen so far
        train_metrics = train_steps_rrpo(
            offline_utils=offline_utils,
            model=model,
            hrpo_table=hrpo_table,
            trie_mask=trie_mask,
            trie_next=trie_next,
            log_paths=train_logs,
            sid_mapping_path=args.sid_mapping_path,
            user_feat_path=args.user_feat_path,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            train_steps=args.train_steps,
            lr=args.lr,
            wd=args.wd,
            beam_width=args.beam_width,
            temperature=args.temperature,
            clip_eps=args.clip_eps,
            kl_coef=args.kl_coef,
            sft_coef=args.sft_coef,
            sync_old_every=args.sync_old_every,
            label_col=args.label_col,
            max_hist_len=args.max_hist_len_train,
            sid_depth=sid_depth,
            gamma=args.gamma,
        )
        print("[train] metrics:", {k: round(v, 6) for k, v in train_metrics.items()})

        ckpt_path = os.path.join(args.output_dir, f"policy_round_{q:03d}.pt")
        torch.save({"model": model.state_dict(), "round": q, "args": vars(args), "hrpo_table": hrpo_path}, ckpt_path)
        print(f"[ckpt] saved -> {ckpt_path}")

        policy_snapshot = deepcopy(model).to(device).eval()
        for p in policy_snapshot.parameters():
            p.requires_grad_(False)
        print("[sync] snapshot policy updated.")

    print("\n[done] Online continuation finished.")


if __name__ == "__main__":
    main()
