import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

try:
    # Runtime path in Agent4Rec simulation.
    from models._sequence_heuristic import SequenceHeuristicBase
except Exception:
    # Package import fallback.
    from recommenders.models._sequence_heuristic import SequenceHeuristicBase


class OneRecHeuristic(SequenceHeuristicBase):
    """Heuristic fallback when no trained OneRec checkpoint is available."""

    def __init__(self, args, data):
        super().__init__(args, data, max_seq_len=35)
        stat_path = f"datasets/{args.dataset}/simulation/user_statistic.csv"
        stat_df = pd.read_csv(stat_path)
        if "user_id" in stat_df.columns:
            stat_df = stat_df.set_index("user_id")
        self.user_stat = stat_df

    def _trait(self, user_id):
        if user_id not in self.user_stat.index:
            return 2.0, 2.0, 2.0
        row = self.user_stat.loc[user_id]
        return float(row["activity"]), float(row["diversity"]), float(row["conformity"])

    def _score_user(self, user_id):
        score = np.zeros(self.n_items, dtype=np.float32)
        history = self.data.train_user_list.get(user_id, [])
        activity, diversity, conformity = self._trait(user_id)

        if not history:
            mix = 0.45 + 0.20 * (diversity - 1.0) / 2.0
            return mix * self.novelty + (1.0 - mix) * self.popularity

        seq = [int(x) for x in history[-self.max_seq_len :]]
        short_term = seq[-6:]
        long_term = seq[:-6]

        for i, item in enumerate(reversed(short_term)):
            w = 0.82**i
            self._row_add(score, self.transition, item, w)
            self._row_add(score, self.cooccurrence, item, 0.20 * w)

        if long_term:
            uniq, counts = np.unique(np.asarray(long_term, dtype=np.int32), return_counts=True)
            weights = counts.astype(np.float32) / max(float(counts.sum()), 1.0)
            for item, w in zip(uniq.tolist(), weights.tolist()):
                self._row_add(score, self.cooccurrence, int(item), 0.55 * w)

        explore = 0.08 + 0.08 * (diversity - 1.0) / 2.0
        conform = 0.05 + 0.06 * (conformity - 1.0) / 2.0
        active = 0.05 + 0.05 * (activity - 1.0) / 2.0
        score += explore * self.novelty + conform * self.popularity + active * score
        return score


class OneRecBackbone(nn.Module):
    """OneRec backbone for offline-trained checkpoint inference."""

    def __init__(
        self,
        item_num: int,
        hidden_units: int,
        num_layers: int,
        num_heads: int,
        dropout_rate: float,
        maxlen: int,
        user_ctx_dim: int = 3,
    ) -> None:
        super().__init__()
        self.item_num = int(item_num)
        self.hidden_units = int(hidden_units)
        self.maxlen = int(maxlen)
        self.user_ctx_dim = int(user_ctx_dim)

        self.item_emb = nn.Embedding(self.item_num + 1, self.hidden_units, padding_idx=0)
        self.pos_emb = nn.Embedding(self.maxlen, self.hidden_units)
        self.emb_dropout = nn.Dropout(float(dropout_rate))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_units,
            nhead=int(num_heads),
            dim_feedforward=self.hidden_units * 4,
            dropout=float(dropout_rate),
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=int(num_layers))
        self.ln = nn.LayerNorm(self.hidden_units)
        self.user_proj = nn.Sequential(
            nn.Linear(self.user_ctx_dim, self.hidden_units),
            nn.ReLU(),
            nn.Dropout(float(dropout_rate)),
            nn.Linear(self.hidden_units, self.hidden_units),
            nn.ReLU(),
        )

    def log2feats(self, log_seqs: torch.Tensor, user_ctx: torch.Tensor) -> torch.Tensor:
        bsz, slen = log_seqs.shape
        pos = torch.arange(slen, device=log_seqs.device).unsqueeze(0).expand(bsz, -1)

        x = self.item_emb(log_seqs) + self.pos_emb(pos)
        x = self.emb_dropout(x)
        key_padding = log_seqs.eq(0)
        z = self.encoder(x, src_key_padding_mask=key_padding)
        z = self.ln(z)
        z = z + self.user_proj(user_ctx).unsqueeze(1)
        z = z.masked_fill(key_padding.unsqueeze(-1), 0.0)
        return z

    def score_all(self, log_seqs: torch.Tensor, user_ctx: torch.Tensor) -> torch.Tensor:
        feats = self.log2feats(log_seqs, user_ctx)
        lengths = log_seqs.ne(0).sum(dim=1).clamp(min=1)
        idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, feats.size(-1))
        final_feat = feats.gather(1, idx).squeeze(1)
        all_items = self.item_emb.weight[1:]
        return torch.matmul(final_feat, all_items.t())


class OneRec(nn.Module):
    """
    OneRec model for Agent4Rec simulation.

    Priority:
    1) load offline-trained checkpoint from recommenders/weights/<dataset>/OneRec/<model_path>
    2) fallback to heuristic OneRec if checkpoint is missing.
    """

    def __init__(self, args, data):
        super().__init__()
        self.args = args
        self.data = data
        self.device = torch.device(args.cuda)

        self.heuristic = None
        self.backbone = None
        self.score_cache = None

        model_path = getattr(args, "model_path", "Saved")
        ckpt_path = self._resolve_checkpoint_path(dataset=args.dataset, model_path=model_path)
        if ckpt_path is None:
            print("[OneRec] no checkpoint found, fallback to heuristic scores.")
            self.heuristic = OneRecHeuristic(args, data)
            return

        print(f"[OneRec] loading checkpoint: {ckpt_path}")
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu")
            cfg = ckpt.get("config", {})

            item_num = int(cfg.get("item_num", self.data.n_items))
            maxlen = int(cfg.get("maxlen", 50))
            hidden_units = int(cfg.get("hidden_units", 64))
            num_layers = int(cfg.get("num_layers", 2))
            num_heads = int(cfg.get("num_heads", 4))
            dropout_rate = float(cfg.get("dropout_rate", 0.2))
            user_ctx_dim = int(cfg.get("user_ctx_dim", 3))

            self.backbone = OneRecBackbone(
                item_num=item_num,
                hidden_units=hidden_units,
                num_layers=num_layers,
                num_heads=num_heads,
                dropout_rate=dropout_rate,
                maxlen=maxlen,
                user_ctx_dim=user_ctx_dim,
            )
            self.backbone.load_state_dict(ckpt["model_state_dict"], strict=True)
            self.backbone.eval()

            user_ctx = self._load_user_context(args.dataset, self.data.n_users, user_ctx_dim=user_ctx_dim)
            seq_by_user = self._load_user_train_sequences()
            self.score_cache = self._build_score_cache(
                seq_by_user=seq_by_user,
                user_ctx=user_ctx,
                maxlen=maxlen,
                item_num=item_num,
            )
        except Exception as e:
            print(f"[OneRec] load failed ({e}), fallback to heuristic scores.")
            self.backbone = None
            self.score_cache = None
            self.heuristic = OneRecHeuristic(args, data)

    def _resolve_checkpoint_path(self, dataset: str, model_path: str):
        env_path = os.getenv("ONEREC_CKPT", "").strip()
        if env_path:
            p = Path(env_path)
            if p.exists():
                return p

        model_path = str(model_path or "Saved")
        candidates = [
            Path(f"recommenders/weights/{dataset}/OneRec/{model_path}/epoch=best.loo.standard.pth"),
            Path(f"recommenders/weights/{dataset}/OneRec/{model_path}/epoch=best.loo.pth"),
            Path(f"recommenders/weights/{dataset}/OneRec/Saved/epoch=best.loo.standard.pth"),
            Path(f"recommenders/weights/{dataset}/OneRec/Saved/epoch=best.loo.pth"),
        ]
        for p in candidates:
            if p.exists():
                return p
        return None

    def _choose_ctx_columns(self, df: pd.DataFrame) -> List[str]:
        preferred = ["activity", "diversity", "conformity"]
        if all(c in df.columns for c in preferred):
            return preferred
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if len(numeric_cols) >= 3:
            return numeric_cols[:3]
        return preferred

    def _load_user_context(self, dataset: str, user_num: int, user_ctx_dim: int) -> np.ndarray:
        out = np.full((int(user_num), int(user_ctx_dim)), 0.5, dtype=np.float32)
        stat_path = Path(f"datasets/{dataset}/simulation/user_statistic.csv")
        if not stat_path.exists():
            return out

        df = pd.read_csv(stat_path)
        if "user_id" in df.columns:
            df = df.set_index("user_id")
        elif df.columns[0].lower().startswith("unnamed"):
            df = df.set_index(df.columns[0])
        try:
            df.index = df.index.astype(int)
        except Exception:
            pass

        cols = self._choose_ctx_columns(df)
        for uid in range(int(user_num)):
            if uid not in df.index:
                continue
            row = df.loc[uid]
            vals = []
            for c in cols:
                if c in row.index:
                    vals.append(float(row[c]))
                else:
                    vals.append(2.0)
            vals = vals[:user_ctx_dim]
            if len(vals) < user_ctx_dim:
                vals += [2.0] * (user_ctx_dim - len(vals))
            out[uid] = np.asarray(vals, dtype=np.float32)

        mins = out.min(axis=0, keepdims=True)
        maxs = out.max(axis=0, keepdims=True)
        denom = np.maximum(maxs - mins, 1e-6)
        out = (out - mins) / denom
        return out.astype(np.float32)

    def _load_user_train_sequences(self) -> Dict[int, List[int]]:
        out = {u: [] for u in range(self.data.n_users)}
        for u in range(self.data.n_users):
            hist = self.data.train_user_list.get(u, [])
            if not hist:
                continue
            out[u] = [int(i) + 1 for i in hist]
        return out

    def _build_score_cache(
        self,
        seq_by_user: Dict[int, List[int]],
        user_ctx: np.ndarray,
        maxlen: int,
        item_num: int,
    ):
        if self.backbone is None:
            return None

        seq_mat = np.zeros((self.data.n_users, int(maxlen)), dtype=np.int64)
        for u in range(self.data.n_users):
            seq = seq_by_user.get(u, [])
            if not seq:
                continue
            seq = seq[-int(maxlen) :]
            seq_mat[u, : len(seq)] = np.asarray(seq, dtype=np.int64)

        self.backbone = self.backbone.to(self.device)
        cache = np.zeros((self.data.n_users, self.data.n_items), dtype=np.float32)
        bs = 256
        with torch.no_grad():
            for st in range(0, self.data.n_users, bs):
                ed = min(st + bs, self.data.n_users)
                seq_t = torch.from_numpy(seq_mat[st:ed]).to(self.device)
                ctx_t = torch.from_numpy(user_ctx[st:ed]).to(self.device)
                scores = self.backbone.score_all(seq_t, ctx_t).cpu().numpy()
                use_cols = min(scores.shape[1], self.data.n_items, int(item_num))
                cache[st:ed, :use_cols] = scores[:, :use_cols]
                if use_cols < self.data.n_items:
                    cache[st:ed, use_cols:] = -1e9
        return cache

    def cuda(self, device):
        self.device = torch.device(device)
        if self.backbone is not None:
            self.backbone = self.backbone.to(self.device)
        if self.heuristic is not None:
            self.heuristic = self.heuristic.to(self.device)
        return self

    def predict(self, users, items=None):
        if items is None:
            items = list(range(self.data.n_items))
        items = np.asarray(items, dtype=np.int64)

        if self.score_cache is not None:
            arr = self.score_cache[np.asarray(users, dtype=np.int64)][:, items]
            return arr.astype(np.float32)

        if self.heuristic is not None:
            return self.heuristic.predict(users, items)

        return np.zeros((len(users), len(items)), dtype=np.float32)
