# -*- coding: utf-8 -*-
"""
dataset_env_ntp.py

EnvNTPDataset:
  generate_onerec_env_ntp_data.py clicksample, 
  history_iids mapping hist_sid ,  target_sid  label token, 
 user features, for OneRec  NTP training. 

CSV ( generate_onerec_env_ntp_data.py ): 
 user_id, history_iids, target_iid, target_sid

: 
 - user_id: user ID( user_features_Pure_fillna.csv )
 - history_iids: history ID, ,  "12,35,78"
 - target_iid: click ID
 - target_sid: click SID , ,  "3,15,9,27"

: 
 from dataset_env_ntp import EnvNTPDataset
 from torch.utils.data import DataLoader

 dataset = EnvNTPDataset(
 click_csv_path="onerec_env_ntp_click_samples.csv",
 sid_mapping_path="video_sid_mapping.csv",
 user_feat_path="user_features_Pure_fillna.csv",
 max_hist_len=50,
 sid_depth=4,
 )
 loader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=4)

 for target_sid, user_feat, hist_sid, hist_len in loader:
 ...
"""

import os
from typing import List, Dict, Tuple, Optional
from utils import get_onehot_vocab
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class EnvNTPDataset(Dataset):
    """
 EnvNTPDataset

 sample: 
 - target_sid: [sid_depth], click item  SID token 
 - user_feat: [user_feat_dim], user features
 - hist_sid: [max_hist_len, sid_depth], history SID ( 0 padding)
 - hist_len: , historylength(<= max_hist_len)
 """

    def __init__(
        self,
        click_csv_path: str,
        sid_mapping_path: str,
        user_feat_path: Optional[str] = None,
        max_hist_len: int = 50,
        sid_depth: int = 4,
        min_hist_len: int = 1,
    ):
        """
 Args:
 click_csv_path: generate_onerec_env_ntp_data.py output CSV filepath
 sid_mapping_path: video_sid_mapping.csv(item_id -> SID )
 user_feat_path: user_features_Pure_fillna.csv(user_id -> feature)
 max_hist_len: history length(,  0)
 sid_depth: SID length( SID codebook ,  4)
 min_hist_len: filterhistory length < min_hist_len sample
 """
        super().__init__()
        self.click_csv_path = click_csv_path
        self.sid_mapping_path = sid_mapping_path
        self.user_feat_path = user_feat_path
        self.max_hist_len = int(max_hist_len)
        self.sid_depth = int(sid_depth)
        self.min_hist_len = int(min_hist_len)

        if not os.path.exists(click_csv_path):
            raise FileNotFoundError(f"click_csv_path not found: {click_csv_path}")
        self.df = pd.read_csv(click_csv_path)

        required_cols = ["user_id", "history_iids", "target_iid", "target_sid"]
        for c in required_cols:
            if c not in self.df.columns:
                raise ValueError(f"Column '{c}' not found in {click_csv_path}")

        self.hist_iids_list: List[List[int]] = []
        hist_lens: List[int] = []

        for s in self.df["history_iids"].astype(str).tolist():
            if s.strip() == "":
                ids = []
            else:
                ids = [int(x) for x in s.split(",") if x.strip() != ""]
            self.hist_iids_list.append(ids)
            hist_lens.append(len(ids))

        self.hist_lens = np.array(hist_lens, dtype=np.int32)

        valid_mask = self.hist_lens >= self.min_hist_len
        if valid_mask.sum() == 0:
            raise ValueError(
                f"After filtering with min_hist_len={self.min_hist_len}, no samples remain."
            )
        self.df = self.df[valid_mask].reset_index(drop=True)
        self.hist_iids_list = [self.hist_iids_list[i] for i, v in enumerate(valid_mask) if v]
        self.hist_lens = self.hist_lens[valid_mask]

        print(
            f"[EnvNTPDataset] Loaded {len(self.df)} samples from {click_csv_path}, "
            f"min_hist_len={self.min_hist_len}, max_hist_len={self.max_hist_len}"
        )

        self._build_iid2sid()

        self.user_feat_dim = 0
        self.user_feat_map: Dict[int, np.ndarray] = {}
        if user_feat_path is not None:
            self._build_user_feat_map()
        else:
            print("[EnvNTPDataset] user_feat_path is None, user_feat_dim=0,  0 feature. ")

    def _build_iid2sid(self):
        if not os.path.exists(self.sid_mapping_path):
            raise FileNotFoundError(f"sid_mapping_path not found: {self.sid_mapping_path}")

        sid_df = pd.read_csv(self.sid_mapping_path)
        item_col_candidates = ["video_id", "item_id", "iid"]
        item_col = None
        for c in item_col_candidates:
            if c in sid_df.columns:
                item_col = c
                break
        if item_col is None:
            item_col = sid_df.columns[0]
            print(f"[EnvNTPDataset] item id column not found, fallback to first column: {item_col}")

        sid_cols = [c for c in sid_df.columns if c != item_col]
        if len(sid_cols) < self.sid_depth:
            raise ValueError(
                f"sid_mapping_path={self.sid_mapping_path}  SID  {len(sid_cols)} "
                f" sid_depth={self.sid_depth}"
            )
        sid_cols = sid_cols[: self.sid_depth]

        max_item_id = int(sid_df[item_col].max())
        self.iid2sid = np.zeros((max_item_id + 1, self.sid_depth), dtype=np.int64)

        for _, row in sid_df.iterrows():
            iid = int(row[item_col])
            if iid < 0:
                continue
            if iid >= self.iid2sid.shape[0]:
                continue
            self.iid2sid[iid] = row[sid_cols].values.astype(np.int64)

        print(
            f"[EnvNTPDataset] Built iid2sid mapping: "
            f"max_iid={max_item_id}, sid_depth={self.sid_depth}"
        )

    def _build_user_feat_map(self):
        """
  onerec_value_v1_32 user features,  user_feat_dim = 74, 
  user_proj.weight  ckpt ,  skip. 
 """
        if not os.path.exists(self.user_feat_path):
            raise FileNotFoundError(f"user_feat_path not found: {self.user_feat_path}")

        uf_df = pd.read_csv(self.user_feat_path).fillna("unknown")

        uid_col = "user_id" if "user_id" in uf_df.columns else uf_df.columns[0]

        used_user_features = [
            "user_active_degree",
            "is_live_streamer",
            "is_video_author",
            "follow_user_num_range",
            "fans_user_num_range",
            "friend_user_num_range",
            "register_days_range",
            "onehot_feat0",
            "onehot_feat1",
            "onehot_feat6",
            "onehot_feat9",
            "onehot_feat10",
            "onehot_feat11",
        ]

        selected_user_features = [f for f in used_user_features if f in uf_df.columns]
        if len(selected_user_features) != len(used_user_features):
            missing = set(used_user_features) - set(selected_user_features)
            print(f"[EnvNTPDataset][Warn] feature {self.user_feat_path} , : {missing}")

        if len(selected_user_features) == 0:
            raise ValueError(
                f"[EnvNTPDataset] No user features found in {self.user_feat_path}! "
                f"expected={used_user_features}"
            )

        print(
            f"[EnvNTPDataset] selected_user_features (aligned with onerec_value_v1_32) = "
            f"{selected_user_features}"
        )

        user_vocab = get_onehot_vocab(uf_df, selected_user_features)

        self.user_feat_map = {}
        self.user_feat_dim = 0

        for _, row in uf_df.iterrows():
            uid = int(row[uid_col])

            feat_vecs = []
            for f in selected_user_features:
                raw_val = row[f]
                vec = user_vocab[f].get(raw_val)

                if vec is None:
                    dim = len(next(iter(user_vocab[f].values())))
                    vec = np.zeros((dim,), dtype=np.float32)
                elif not isinstance(vec, np.ndarray):
                    vec = np.array([vec], dtype=np.float32)
                else:
                    vec = vec.astype(np.float32)

                feat_vecs.append(vec)

            if feat_vecs:
                vec_cat = np.concatenate(feat_vecs, axis=-1).astype(np.float32)
            else:
                vec_cat = np.zeros((0,), dtype=np.float32)

            self.user_feat_map[uid] = vec_cat
            if self.user_feat_dim == 0:
                self.user_feat_dim = vec_cat.shape[-1]

        print(
            f"[EnvNTPDataset] Built user_feat_map (aligned with ckpt): "
            f"{len(self.user_feat_map)} users, user_feat_dim={self.user_feat_dim}"
        )



    def __len__(self) -> int:
        return len(self.df)

    def _iid_seq_to_sid_seq(self, iid_list: List[int]) -> Tuple[np.ndarray, int]:
        """
 history iid mapping [max_hist_len, sid_depth]  SID , 
  0 padding, align. 
 : (hist_sid, hist_len)
 """
        L = len(iid_list)
        if L == 0:
            hist_len = 0
            hist_sid = np.zeros((self.max_hist_len, self.sid_depth), dtype=np.int64)
            return hist_sid, hist_len

        hist_len = min(L, self.max_hist_len)
        iid_list_use = iid_list[-hist_len:]

        hist_sid = np.zeros((self.max_hist_len, self.sid_depth), dtype=np.int64)

        start_idx = self.max_hist_len - hist_len
        for k, iid in enumerate(iid_list_use):
            if iid <= 0 or iid >= self.iid2sid.shape[0]:
                sid_vec = np.zeros((self.sid_depth,), dtype=np.int64)
            else:
                sid_vec = self.iid2sid[iid]
            hist_sid[start_idx + k] = sid_vec

        return hist_sid, hist_len

    def _parse_target_sid(self, sid_str: str) -> np.ndarray:
        """
  target_sid  [sid_depth]  int64 . 
  "3,15,9,27" -> [3,15,9,27]. 
 iflength sid_depth,  0; if. 
 """
        if sid_str.strip() == "":
            vals: List[int] = []
        else:
            vals = [int(x) for x in sid_str.split(",") if x.strip() != ""]

        if len(vals) >= self.sid_depth:
            arr = np.array(vals[: self.sid_depth], dtype=np.int64)
        else:
            arr = np.zeros((self.sid_depth,), dtype=np.int64)
            arr[: len(vals)] = np.array(vals, dtype=np.int64)
        return arr

    def __getitem__(self, idx: int):
        """
 :
 target_sid: LongTensor [sid_depth]
 user_feat: FloatTensor [user_feat_dim]
 hist_sid: LongTensor [max_hist_len, sid_depth]
 hist_len: LongTensor [] ()
 """
        row = self.df.iloc[idx]

        uid = int(row["user_id"])

        hist_iids = self.hist_iids_list[idx]
        hist_sid_np, hist_len = self._iid_seq_to_sid_seq(hist_iids)

        target_sid_str = str(row["target_sid"])
        target_sid_np = self._parse_target_sid(target_sid_str)

        if self.user_feat_dim > 0 and uid in self.user_feat_map:
            user_feat_np = self.user_feat_map[uid]
        else:
            user_feat_np = np.zeros((self.user_feat_dim,), dtype=np.float32)

        target_sid = torch.from_numpy(target_sid_np).long()
        user_feat = torch.from_numpy(user_feat_np).float()
        hist_sid = torch.from_numpy(hist_sid_np).long()
        hist_len = torch.tensor(hist_len, dtype=torch.long)

        return target_sid, user_feat, hist_sid, hist_len
