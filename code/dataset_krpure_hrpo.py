# -*- coding: utf-8 -*-
"""
KRPure dataset that returns dense per-token HRPO reward labels.

It mirrors the basic interface of your KRPureValueDataset:
  __getitem__ -> (target_sid [L], user_feat [F], hist_sid [H,L], hist_len, reward_label [L], ltv_label [1])

Where:
  reward_label[t] is the HRPO residual at depth t (1..L):
      residual_t = E[r | ctx, prefix[:t]] - E[r | ctx, prefix[:t-1]]

ltv_label uses discounted sum of *scalar* reward on the logged trajectory (optional supervision).
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass
class HRPOTable:
    sid_cols: List[str]
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
        return HRPOTable(
            sid_cols=list(d["sid_cols"]),
            sid_depth=int(d["sid_depth"]),
            uid2ctx={int(k): int(v) for k, v in d["uid2ctx"].items()},
            ctx_count={int(k): int(v) for k, v in d["ctx_count"].items()},
            ctx_sum={int(k): float(v) for k, v in d["ctx_sum"].items()},
            prefix_count={(int(k[0]), tuple(k[1])): int(v) for k, v in d["prefix_count"].items()},
            prefix_sum={(int(k[0]), tuple(k[1])): float(v) for k, v in d["prefix_sum"].items()},
            global_mean=float(d["global_mean"]),
            smoothing_alpha=float(d["smoothing_alpha"]),
            min_prefix_count=int(d["min_prefix_count"]),
        )

    def _ctx_mean(self, ctx: int) -> float:
        """Empirical-Bayes ctx mean smoothed toward global_mean."""
        cnt = self.ctx_count.get(ctx, 0)
        if cnt <= 0:
            return self.global_mean
        a = self.smoothing_alpha
        return (self.ctx_sum.get(ctx, 0.0) + a * self.global_mean) / (cnt + a)

    def _shrink_mean(self, sum_r: float, cnt: int, prior: float) -> float:
        """Shrink a sample mean toward a prior mean using pseudo-count a."""
        a = self.smoothing_alpha
        return (sum_r + a * prior) / (cnt + a)

    def mean_reward(self, ctx: int, prefix: Tuple[int, ...]) -> float:
        """
        Backoff: (ctx,prefix)->(ctx,shorter)->(ctx,()) -> global_mean

        IMPORTANT behavior for HRPO stability:
          - We do NOT want deep layers to become "almost always zero" due to a hard min-count threshold.
          - Therefore, if a prefix has ANY support (cnt>0), we still use it, but shrink toward ctx-mean.
          - Only when cnt==0 do we backoff to a shorter prefix.

        If you really want hard filtering, set min_prefix_count>1; then prefixes with 0<cnt<min will
        be treated like missing and will backoff.
        """
        ctx_prior = self._ctx_mean(ctx)

        p = prefix
        while True:
            if len(p) == 0:
                return ctx_prior
            key = (ctx, p)
            cnt = self.prefix_count.get(key, 0)
            if cnt > 0 and cnt >= self.min_prefix_count:
                return self._shrink_mean(self.prefix_sum.get(key, 0.0), cnt, prior=ctx_prior)
            if cnt > 0 and self.min_prefix_count <= 1:
                return self._shrink_mean(self.prefix_sum.get(key, 0.0), cnt, prior=ctx_prior)
            p = p[:-1]

    def residuals_for_path(self, ctx: int, path: Tuple[int, ...]) -> np.ndarray:
        assert len(path) == self.sid_depth
        res = np.zeros((self.sid_depth,), dtype=np.float32)
        prev = self.mean_reward(ctx, tuple())
        pref = tuple()
        for i, tok in enumerate(path):
            pref = pref + (int(tok),)
            cur = self.mean_reward(ctx, pref)
            res[i] = cur - prev
            prev = cur
        return res


class KRPureHRPODataset(Dataset):
    def __init__(
        self,
        log_paths: List[str],
        sid_mapping_path: str,
        user_feat_path: str,
        hrpo_table_path: str,
        sid_cols: Optional[List[str]] = None,
        reward_weights: Optional[Dict[str, float]] = None,
        gamma: float = 0.99,
        min_hist_len: int = 1,
        max_hist_len: int = 50,
        positive_only: bool = True,
        positive_threshold: float = 0.0,
        time_col: str = "time_ms",
        user_id_col: str = "user_id",
        video_id_col: str = "video_id",
    ):
        super().__init__()
        self.gamma = gamma
        self.min_hist_len = min_hist_len
        self.max_hist_len = max_hist_len
        self.positive_only = positive_only
        self.positive_threshold = float(positive_threshold)

        self.hrpo = HRPOTable.load(hrpo_table_path)

        df_sid = pd.read_csv(sid_mapping_path)
        if sid_cols is None:
            sid_cols = list(self.hrpo.sid_cols)
        self.sid_cols = sid_cols
        for c in [video_id_col] + self.sid_cols:
            if c not in df_sid.columns:
                raise ValueError(f"sid mapping missing col: {c}")
        df_sid = df_sid[[video_id_col] + self.sid_cols].copy()

        df_user = pd.read_csv(user_feat_path)
        if "user_id" not in df_user.columns:
            raise ValueError("user feature file missing 'user_id'")
        selected_user_features = [
            "user_active_degree",
            "is_live_streamer",
            "is_video_author",
            "follow_user_num_range",
            "fans_user_num_range",
            "friend_user_num_range",
            "register_days_range",
        ] + [f"onehot_feat{fid}" for fid in [0, 1, 6, 9, 10, 11]]

        feat_arrays = []
        df_user_userid = df_user["user_id"].astype(int).values
        for col in selected_user_features:
            if col not in df_user.columns:
                raise ValueError(f"user feature file missing '{col}'")
            raw_vals = df_user[col].astype(str).fillna("nan").values
            uniq = sorted(list(set(raw_vals)))
            vocab = {v: i for i, v in enumerate(uniq)}
            dim = len(uniq)
            idx = np.array([vocab[v] for v in raw_vals], dtype=np.int32)
            one_hot = np.zeros((len(raw_vals), dim), dtype="float32")
            one_hot[np.arange(len(raw_vals)), idx] = 1.0
            feat_arrays.append(one_hot)
        user_feat_mat = np.concatenate(feat_arrays, axis=1).astype("float32")
        user_id2idx = {int(u): i for i, u in enumerate(df_user_userid)}

        dfs = [pd.read_csv(p) for p in log_paths]
        df_log = pd.concat(dfs, axis=0, ignore_index=True)

        df_log = df_log.merge(df_sid, on=video_id_col, how="inner")
        df_log[user_id_col] = df_log[user_id_col].astype(int)

        if reward_weights is None:
            reward_weights = {"is_click": 1.0}
        self.reward_weights = reward_weights

        for k in reward_weights.keys():
            if k not in df_log.columns:
                raise ValueError(f"log missing behavior column: {k}")

        if time_col in df_log.columns:
            df_log = df_log.sort_values(by=[user_id_col, time_col], kind="mergesort").reset_index(drop=True)
        else:
            pass

        df_log = df_log[df_log[user_id_col].isin(user_id2idx.keys())].reset_index(drop=True)

        target_list = []
        user_feat_list = []
        hist_list = []
        hist_len_list = []
        hrpo_reward_list = []
        ltv_list = []
        sample_uid_list = []

        rw = np.zeros((len(df_log),), dtype=np.float32)
        for k, w in reward_weights.items():
            rw += df_log[k].astype(np.float32).values * float(w)
        df_log["_scalar_r"] = rw

        user_groups = df_log.groupby(user_id_col, sort=False).indices  # {uid: np.array(row_idx)}
        sid_mat = df_log[self.sid_cols].astype(np.int64).values
        uid_arr = df_log[user_id_col].astype(np.int64).values
        r_arr = df_log["_scalar_r"].astype(np.float32).values

        for uid, idxs in user_groups.items():
            idxs = np.asarray(idxs, dtype=np.int64)
            if len(idxs) <= self.min_hist_len:
                continue

            rr = r_arr[idxs]
            ltv = np.zeros_like(rr, dtype=np.float32)
            g = 1.0
            acc = 0.0
            for i in range(len(rr)-1, -1, -1):
                acc = rr[i] + self.gamma * acc
                ltv[i] = acc

            history: List[np.ndarray] = []
            ctx = int(self.hrpo.uid2ctx.get(int(uid), 0))
            uidx = user_id2idx[int(uid)]
            uf = user_feat_mat[uidx]

            for t in range(len(idxs)):
                cur_row = idxs[t]
                target_sid = sid_mat[cur_row].astype(np.int64)
                r_t = float(r_arr[cur_row])

                if (not self.positive_only) or (r_t > self.positive_threshold):
                    if len(history) >= self.min_hist_len:
                        h = history[-self.max_hist_len:]
                        h_len = len(h)
                        h_pad = np.zeros((self.max_hist_len, target_sid.shape[0]), dtype=np.int64)
                        h_arr = np.stack(h, axis=0).astype(np.int64)
                        h_pad[-h_len:, :] = h_arr

                        hrpo_r = self.hrpo.residuals_for_path(ctx, tuple(target_sid.tolist()))

                        target_list.append(target_sid)
                        user_feat_list.append(uf)
                        hist_list.append(h_pad)
                        hist_len_list.append(h_len)
                        hrpo_reward_list.append(hrpo_r.astype(np.float32))
                        ltv_list.append(float(ltv[t]))
                        sample_uid_list.append(int(uid))

                history.append(target_sid)

        self.hist_sid = torch.from_numpy(np.stack(hist_list, axis=0))      # [N, H, L]
        self.hist_len = torch.from_numpy(np.array(hist_len_list, dtype=np.int16))
        self.target_sid = torch.from_numpy(np.stack(target_list, axis=0))  # [N, L]
        self.user_feat = torch.from_numpy(np.stack(user_feat_list, axis=0)) # [N, F]
        self.rewards = torch.from_numpy(np.stack(hrpo_reward_list, axis=0)) # [N, L]
        self.ltvs = torch.from_numpy(np.array(ltv_list, dtype=np.float32)).unsqueeze(1) # [N,1]
        self.sample_user_ids = torch.from_numpy(np.array(sample_uid_list, dtype=np.int32))

        self.ltvs = torch.clamp(self.ltvs, min=0.0, max=50.0) / 50.0

        print(f"[HRPO-Data] Loaded {len(self.target_sid)} samples.")
        print(f"[HRPO-Data] user_feat_dim={self.user_feat.shape[1]} hist_max_len={self.hist_sid.shape[1]} sid_depth={self.target_sid.shape[1]}")

    def __len__(self):
        return self.target_sid.shape[0]

    def __getitem__(self, idx):
        return (
            self.target_sid[idx],
            self.user_feat[idx],
            self.hist_sid[idx],
            self.hist_len[idx],
            self.rewards[idx],
            self.ltvs[idx],
        )
