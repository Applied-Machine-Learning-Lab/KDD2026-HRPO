# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional
import re


class KRPureValueDataset(Dataset):
    def __init__(
        self,
        log_paths: List[str],
        sid_mapping_path: str,
        user_feat_path: str,
        sid_cols: Optional[List[str]] = None,
        label_col: str = "is_click",
        gt_spec: str = "",  # e.g. "is_click:1,long_view:2,is_like:5" (weight=0 -> ignore)
        gt_gate: str = "",  # optional, e.g. "is_click" -> only rows with gate==1 can be GT
        hist_from_gt: int = 0,  # 1: history only keeps GT events; 0: keep all exposures
        gamma: float = 0.99,  # 
        min_hist_len: int = 1,
        max_hist_len: int = 50,
    ):
        super().__init__()
        self.gamma = gamma
        self.min_hist_len = min_hist_len
        self.max_hist_len = max_hist_len

        df_sid = pd.read_csv(sid_mapping_path)
        if sid_cols is None or len(sid_cols) == 0:
            sid_cols = [c for c in df_sid.columns if c.startswith("sid_")]
            sid_cols = sorted(sid_cols)
        if len(sid_cols) == 0:
            raise ValueError("SID mappingfileno sid_* ")
        self.sid_cols = sid_cols
        self.sid_depth = len(sid_cols)

        df_user = pd.read_csv(user_feat_path)
        if "user_id" not in df_user.columns:
            raise ValueError("user featurefile 'user_id' ")

        df_user_userid = df_user["user_id"].values

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
        print("[Data] Encoding user features...")
        for col in selected_user_features:
            if col in df_user.columns:
                raw_vals = df_user[col].astype(str).values
                unique_vals = sorted(list(set(raw_vals)))
                vocab = {v: i for i, v in enumerate(unique_vals)}
                dim = len(unique_vals)
                val_indices = np.array([vocab[v] for v in raw_vals])
                one_hot = np.zeros((len(raw_vals), dim), dtype="float32")
                one_hot[np.arange(len(raw_vals)), val_indices] = 1.0
                feat_arrays.append(one_hot)
            else:
                print(f"[Warning] Missing column {col}, skipping.")

        if len(feat_arrays) > 0:
            user_feat_arr = np.concatenate(feat_arrays, axis=1)
        else:
            raise ValueError("No user features selected!")

        user_feat_dim = user_feat_arr.shape[1]
        user_id2idx = {uid: i for i, uid in enumerate(df_user_userid)}

        self.user_feat_dim = user_feat_dim

        self.label_col = label_col
        self.gt_spec = (gt_spec or "").strip()
        self.gt_gate = (gt_gate or "").strip()
        self.hist_from_gt = int(hist_from_gt)

        self.gt_weights: Optional[Dict[str, float]] = None
        if self.gt_spec:
            self.gt_weights = self._parse_gt_spec(self.gt_spec)
            if len(self.gt_weights) == 0:
                raise ValueError(f"gt_spec parsed empty. got: {self.gt_spec!r}")
        self._user_feat_arr = user_feat_arr
        self._user_id2idx = user_id2idx

        print(f"[Data] User feature processing done. Total dim: {user_feat_dim}")

        df_logs = []
        for p in log_paths:
            df = pd.read_csv(p)
            df_logs.append(df)
        df_log = pd.concat(df_logs, axis=0, ignore_index=True)

        print("[Data] Merging SID mapping...")
        df_log = df_log.merge(
            df_sid[["video_id"] + self.sid_cols], on="video_id", how="inner"
        )

        needed_cols = ["user_id", "video_id"] + self.sid_cols

        if self.gt_weights is None:
            needed_cols.append(label_col)
        else:
            needed_cols += list(self.gt_weights.keys())
            if self.gt_gate and (self.gt_gate not in needed_cols):
                needed_cols.append(self.gt_gate)

        if "time_ms" in df_log.columns:
            needed_cols.append("time_ms")
        elif "date" in df_log.columns and "hourmin" in df_log.columns:
            needed_cols += ["date", "hourmin"]

        df_log = df_log[needed_cols]
        df_log = df_log[df_log["user_id"].isin(user_id2idx.keys())]

        if self.gt_weights is None:
            df_log["_gt_score"] = df_log[label_col].astype("float32")
        else:
            score = np.zeros((len(df_log),), dtype="float32")
            for col, w in self.gt_weights.items():
                if col not in df_log.columns:
                    raise ValueError(f"gt_spec column missing in log: {col}")
                score += float(w) * (df_log[col].astype("float32").values > 0).astype("float32")
            if self.gt_gate:
                if self.gt_gate not in df_log.columns:
                    raise ValueError(f"gt_gate column missing in log: {self.gt_gate}")
                gate = (df_log[self.gt_gate].astype("float32").values > 0).astype("float32")
                score = score * gate
            df_log["_gt_score"] = score

        if "time_ms" in df_log.columns:
            df_log = df_log.sort_values(["user_id", "time_ms"])
        else:
            df_log = df_log.sort_values(["user_id", "date", "hourmin"])

        hist_list = []
        hist_len_list = []
        target_list = []
        user_feat_list = []
        reward_list = []  # click
        ltv_list = []  # LTV (gamma )

        sample_uid_list = []  # * sample user_id(foruser)

        print("[Data] Building sequence samples with Value Labels (Reward & LTV)...")

        for uid, g in df_log.groupby("user_id"):
            rewards = g["_gt_score"].values.astype(np.float32)
            ltvs = np.zeros_like(rewards)

            running_val = 0.0
            for t in range(len(rewards) - 1, -1, -1):
                running_val = rewards[t] + self.gamma * running_val
                ltvs[t] = running_val

            u_idx = user_id2idx.get(uid, None)
            if u_idx is None:
                continue
            u_vec = self._user_feat_arr[u_idx]
            sids = g[self.sid_cols].values.astype("int16")

            history = []

            for t in range(len(g)):
                sid_vec = sids[t]
                r_t = rewards[t]
                ltv_t = ltvs[t]

                if r_t > 0 and len(history) >= min_hist_len:
                    h = history[-max_hist_len:]
                    h_len = len(h)
                    h_pad = np.zeros((max_hist_len, self.sid_depth), dtype="int16")
                    h_arr = np.stack(h, axis=0)
                    h_pad[-h_len:, :] = h_arr

                    hist_list.append(h_pad)
                    hist_len_list.append(h_len)
                    target_list.append(sid_vec)
                    user_feat_list.append(u_vec)

                    reward_list.append(r_t)
                    ltv_list.append(ltv_t)
                    sample_uid_list.append(uid)  # * sample user

                if self.hist_from_gt:
                    if r_t > 0:
                        history.append(sid_vec)
                else:
                    history.append(sid_vec)

        if len(target_list) == 0:
            raise ValueError("No samples generated! Check label_col or data.")

        self.hist_sid = torch.from_numpy(np.stack(hist_list, axis=0))          # [N, T, L]
        self.hist_len = torch.from_numpy(np.array(hist_len_list, dtype="int16"))  # [N]
        self.target_sid = torch.from_numpy(np.stack(target_list, axis=0))      # [N, L]
        self.user_feat = torch.from_numpy(np.stack(user_feat_list, axis=0))    # [N, F]

        self.rewards = (
            torch.from_numpy(np.array(reward_list, dtype="float32")).unsqueeze(1)
        )  # [N,1]
        self.ltvs = (
            torch.from_numpy(np.array(ltv_list, dtype="float32")).unsqueeze(1)
        )  # [N,1]

        self.sample_user_ids = torch.from_numpy(
            np.array(sample_uid_list, dtype="int32")
        )  # [N]

        print("[Data] Processing LTV: Clipping to [0, 50] and Normalizing to [0, 1]...")
        self.ltvs = torch.clamp(self.ltvs, min=0.0, max=50.0)
        self.ltvs = self.ltvs / 50.0

        print(f"[Data] Loaded {len(self.target_sid)} samples with Value Labels.")

        self._print_stats("Short-term Reward", self.rewards.numpy())
        self._print_stats("Normalized LTV", self.ltvs.numpy())
        self._print_distribution("Normalized LTV", self.ltvs.numpy())

    def _print_stats(self, name, data):
        print(f"\n>>> Statistics for {name}:")
        print(f"    Min:    {np.min(data):.4f}")
        print(f"    Max:    {np.max(data):.4f}")
        print(f"    Mean:   {np.mean(data):.4f}")
        print(f"    Var:    {np.var(data):.4f}")
        print(f"    Median: {np.median(data):.4f}")
        print(f"    P25:    {np.percentile(data, 25):.4f}")
        print(f"    P50:    {np.percentile(data, 50):.4f}")
        print(f"    P75:    {np.percentile(data, 75):.4f}")
        print(f"    P95:    {np.percentile(data, 95):.4f}")
        print("--------------------------------------------")

    def _print_distribution(self, name, data):
        print(f"\n>>> Distribution for {name} (Interval 0.1):")
        total = len(data)
        bins = np.arange(0.0, 1.1, 0.1)
        hist, bin_edges = np.histogram(data, bins=bins)

        for i in range(len(hist)):
            start = bin_edges[i]
            end = bin_edges[i + 1]
            count = hist[i]
            ratio = count / total * 100
            print(f"    [{start:.1f} - {end:.1f}): {count} samples ({ratio:.2f}%)")
        print("--------------------------------------------")


    @staticmethod
    def _parse_gt_spec(gt_spec: str) -> Dict[str, float]:
        """Parse gt_spec string into {col: weight}.

        Examples:
            "is_click:1,long_view:2,is_like:5"
            "is_click=1 long_view=2"
        Weight == 0 will be ignored.
        """
        spec = (gt_spec or "").strip()
        if not spec:
            return {}
        parts = re.split(r"[;,\s]+", spec)
        out: Dict[str, float] = {}
        for p in parts:
            if not p:
                continue
            if ":" in p:
                k, v = p.split(":", 1)
            elif "=" in p:
                k, v = p.split("=", 1)
            else:
                raise ValueError(f"Bad gt_spec item {p!r}, expected 'col:weight' or 'col=weight'")
            k = k.strip()
            v = v.strip()
            if not k:
                continue
            w = float(v)
            if abs(w) <= 0.0:
                continue
            out[k] = w
        return out

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
