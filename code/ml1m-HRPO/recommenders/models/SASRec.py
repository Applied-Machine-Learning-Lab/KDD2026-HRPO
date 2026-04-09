import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


class PointWiseFeedForward(nn.Module):
    def __init__(self, hidden_units: int, dropout_rate: float) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2)
        outputs += inputs
        return outputs


class SASRecBackbone(nn.Module):
    def __init__(
        self,
        item_num: int,
        maxlen: int,
        hidden_units: int,
        num_blocks: int,
        num_heads: int,
        dropout_rate: float,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.item_num = int(item_num)
        self.maxlen = int(maxlen)
        self.hidden_units = int(hidden_units)
        self.dev = device

        self.item_emb = nn.Embedding(self.item_num + 1, self.hidden_units, padding_idx=0)
        self.pos_emb = nn.Embedding(self.maxlen, self.hidden_units)
        self.emb_dropout = nn.Dropout(dropout_rate)

        self.attention_layernorms = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        self.forward_layernorms = nn.ModuleList()
        self.forward_layers = nn.ModuleList()

        for _ in range(num_blocks):
            self.attention_layernorms.append(nn.LayerNorm(self.hidden_units, eps=1e-8))
            self.attention_layers.append(
                nn.MultiheadAttention(
                    embed_dim=self.hidden_units,
                    num_heads=num_heads,
                    dropout=dropout_rate,
                    batch_first=True,
                )
            )
            self.forward_layernorms.append(nn.LayerNorm(self.hidden_units, eps=1e-8))
            self.forward_layers.append(PointWiseFeedForward(self.hidden_units, dropout_rate))

        self.last_layernorm = nn.LayerNorm(self.hidden_units, eps=1e-8)

    def log2feats(self, log_seqs: torch.Tensor) -> torch.Tensor:
        seqs = self.item_emb(log_seqs) * (self.hidden_units**0.5)
        poss = torch.arange(log_seqs.shape[1], device=self.dev).unsqueeze(0).repeat(log_seqs.shape[0], 1)
        seqs = seqs + self.pos_emb(poss)
        seqs = self.emb_dropout(seqs)

        timeline_mask = log_seqs.eq(0)
        seqs = seqs.masked_fill(timeline_mask.unsqueeze(-1), 0.0)

        tl = seqs.shape[1]
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

        for i in range(len(self.attention_layers)):
            q = self.attention_layernorms[i](seqs)
            mha_out, _ = self.attention_layers[i](
                q,
                seqs,
                seqs,
                attn_mask=attention_mask,
                key_padding_mask=timeline_mask,
                need_weights=False,
            )
            seqs = q + mha_out
            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs = seqs.masked_fill(timeline_mask.unsqueeze(-1), 0.0)

        return self.last_layernorm(seqs)

    def score_all(self, log_seqs: torch.Tensor) -> torch.Tensor:
        feats = self.log2feats(log_seqs)
        final_feat = feats[:, -1, :]
        # index 0 in output corresponds to item id 1
        all_items = self.item_emb.weight[1:]
        return torch.matmul(final_feat, all_items.t())


class SASRec(nn.Module):
    """
    SASRec model for Agent4Rec simulation.
    It loads a pre-trained checkpoint and serves static user->item scores.
    """

    def __init__(self, args, data):
        super().__init__()
        self.args = args
        self.data = data
        self.device = torch.device(args.cuda)

        self.popularity = np.zeros(self.data.n_items, dtype=np.float32)
        for item, users in self.data.train_item_list.items():
            self.popularity[int(item)] = float(len(users))

        model_path = getattr(args, "model_path", "Saved")
        ckpt_path = self._resolve_checkpoint_path(dataset=args.dataset, model_path=model_path)
        self.backbone = None
        self.score_cache = None

        if ckpt_path is None:
            print("[SASRec] no checkpoint found, fallback to popularity scores.")
            return

        print(f"[SASRec] loading checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        cfg = ckpt.get("config", {})

        item_num = int(cfg.get("item_num", self.data.n_items))
        maxlen = int(cfg.get("maxlen", 50))
        hidden_units = int(cfg.get("hidden_units", 50))
        num_blocks = int(cfg.get("num_blocks", 2))
        num_heads = int(cfg.get("num_heads", 1))
        dropout_rate = float(cfg.get("dropout_rate", 0.2))

        self.backbone = SASRecBackbone(
            item_num=item_num,
            maxlen=maxlen,
            hidden_units=hidden_units,
            num_blocks=num_blocks,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            device=self.device,
        )
        self.backbone.load_state_dict(ckpt["model_state_dict"], strict=True)
        self.backbone.eval()

        min_rating = float(os.getenv("SASREC_MIN_RATING", "4.0"))
        cf_data_subdir = str(getattr(args, "cf_data_subdir", "") or "").strip()
        source_override = os.getenv("SASREC_SEQUENCE_SOURCE", "").strip().lower()
        if source_override in {"cf", "cf_data", "agent"}:
            prefer_cf_sequences = True
        elif source_override in {"raw", "ratings"}:
            prefer_cf_sequences = False
        else:
            prefer_cf_sequences = bool(cf_data_subdir)

        source_label = "cf_data" if prefer_cf_sequences else "raw_ratings"
        print(f"[SASRec] sequence source: {source_label}")
        seq_by_user = self._load_user_train_sequences(
            args.dataset,
            min_rating=min_rating,
            prefer_cf_sequences=prefer_cf_sequences,
        )
        self.score_cache = self._build_score_cache(seq_by_user=seq_by_user, maxlen=maxlen, item_num=item_num)

    def _resolve_checkpoint_path(self, dataset: str, model_path: str):
        env_path = os.getenv("SASREC_CKPT", "").strip()
        if env_path:
            p = Path(env_path)
            if p.exists():
                return p

        # Prefer the folder specified by the simulation runner (e.g. Saved_min4).
        model_path = str(model_path or "Saved")
        candidates = [
            Path(f"recommenders/weights/{dataset}/SASRec/{model_path}/epoch=best.loo.standard.pth"),
            Path(f"recommenders/weights/{dataset}/SASRec/{model_path}/epoch=best.loo.pth"),
            Path(f"recommenders/weights/{dataset}/SASRec/Saved_fullce_fullsel_h128/epoch=best.loo.standard.pth"),
            Path(f"recommenders/weights/{dataset}/SASRec/Saved_fullce_fullsel/epoch=best.loo.standard.pth"),
            Path(f"recommenders/weights/{dataset}/SASRec/Saved/epoch=best.loo.standard.pth"),
            Path(f"recommenders/weights/{dataset}/SASRec/Saved/epoch=best.loo.pth"),
        ]
        for p in candidates:
            if p.exists():
                return p
        return None

    def _load_user_train_sequences(
        self,
        dataset: str,
        min_rating: float,
        prefer_cf_sequences: bool,
    ) -> Dict[int, List[int]]:
        if prefer_cf_sequences:
            return self._load_cf_train_sequences()

        raw_dir = Path(f"datasets/{dataset}/raw_data")
        user_map_path = raw_dir / "user_id_map.pkl"
        movie_map_path = raw_dir / "movie_id_map.pkl"
        ratings_path = raw_dir / "ratings.dat"

        # MovieLens-style raw_data loader (has ratings + timestamp).
        if user_map_path.exists() and movie_map_path.exists() and ratings_path.exists():
            with user_map_path.open("rb") as f:
                user_id_map = pickle_load(f)
            with movie_map_path.open("rb") as f:
                movie_id_map = pickle_load(f)

            ratings = pd.read_csv(
                ratings_path,
                sep="::",
                engine="python",
                header=None,
                names=["user_id", "movie_id", "rating", "timestamp"],
            )
            ratings = ratings[ratings["rating"] >= min_rating]
            ratings = ratings[ratings["user_id"].isin(user_id_map.keys())]
            ratings = ratings[ratings["movie_id"].isin(movie_id_map.keys())].copy()

            ratings["uid"] = ratings["user_id"].map(user_id_map).astype(np.int32)
            ratings["iid"] = ratings["movie_id"].map(movie_id_map).astype(np.int32) + 1
            ratings = ratings.sort_values(["uid", "timestamp"])

            seq_by_user = {u: [] for u in range(self.data.n_users)}
            for uid, iid in ratings[["uid", "iid"]].itertuples(index=False):
                if 0 <= int(uid) < self.data.n_users:
                    seq_by_user[int(uid)].append(int(iid))

            # align with training setting: user_train = seq[:-2]
            out = {}
            for u, seq in seq_by_user.items():
                if len(seq) < 3:
                    out[u] = []
                else:
                    out[u] = seq[:-2]
            return out

        # Fallback: Agent4Rec-format datasets (e.g., Amazon All_Beauty) already
        # have time-ordered training sequences in cf_data. Here `train_user_list`
        # corresponds to the "user_train" split, so we do NOT cut off tail items.
        return self._load_cf_train_sequences()

    def _load_cf_train_sequences(self) -> Dict[int, List[int]]:
        out = {u: [] for u in range(self.data.n_users)}
        for u in range(self.data.n_users):
            hist = self.data.train_user_list.get(u, [])
            if not hist:
                continue
            # Reserve 0 for padding to match training (item_id = data_id + 1).
            out[u] = [int(i) + 1 for i in hist]
        return out

    def _build_score_cache(self, seq_by_user: Dict[int, List[int]], maxlen: int, item_num: int):
        if self.backbone is None:
            return None

        seq_mat = np.zeros((self.data.n_users, maxlen), dtype=np.int64)
        for u in range(self.data.n_users):
            seq = seq_by_user.get(u, [])
            if not seq:
                continue
            seq = seq[-maxlen:]
            seq_mat[u, -len(seq) :] = np.asarray(seq, dtype=np.int64)

        self.backbone = self.backbone.to(self.device)
        cache = np.zeros((self.data.n_users, self.data.n_items), dtype=np.float32)
        bs = 256
        with torch.no_grad():
            for st in range(0, self.data.n_users, bs):
                ed = min(st + bs, self.data.n_users)
                seq_t = torch.from_numpy(seq_mat[st:ed]).to(self.device)
                scores = self.backbone.score_all(seq_t).cpu().numpy()  # [B, item_num]
                # item id mapping: score col j <-> data item id j (because training used iid=data_id+1)
                use_cols = min(scores.shape[1], self.data.n_items)
                cache[st:ed, :use_cols] = scores[:, :use_cols]
                if use_cols < self.data.n_items:
                    cache[st:ed, use_cols:] = -1e9
        return cache

    def cuda(self, device):
        self.device = torch.device(device)
        if self.backbone is not None:
            self.backbone = self.backbone.to(self.device)
        return self

    def predict(self, users, items=None):
        if items is None:
            items = list(range(self.data.n_items))
        items = np.asarray(items, dtype=np.int64)

        if self.score_cache is not None:
            arr = self.score_cache[np.asarray(users, dtype=np.int64)][:, items]
            return arr.astype(np.float32)

        # Fallback: popularity
        return np.tile(self.popularity[items], (len(users), 1)).astype(np.float32)


def pickle_load(file_obj):
    import pickle

    return pickle.load(file_obj)
