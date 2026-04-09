import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

try:
    # When `recommenders/` is added to sys.path (Agent4Rec simulation runtime).
    from models._sequence_heuristic import SequenceHeuristicBase
except Exception:
    # When importing as a package from repo root.
    from recommenders.models._sequence_heuristic import SequenceHeuristicBase


class GRU4RecHeuristic(SequenceHeuristicBase):
    """Legacy heuristic GRU4Rec-style scorer used in early Agent4Rec baselines."""

    def __init__(self, args, data):
        super().__init__(args, data, max_seq_len=25)
        self.pop_coef = 0.10
        self.decay = 0.72

    def _score_user(self, user_id):
        score = np.zeros(self.n_items, dtype=np.float32)
        history = self.data.train_user_list.get(user_id, [])
        if not history:
            return 0.7 * self.popularity + 0.3 * self.novelty

        seq = [int(x) for x in history[-self.max_seq_len :]]
        for i, item in enumerate(reversed(seq)):
            w = self.decay**i
            self._row_add(score, self.transition, item, w)

        score += self.pop_coef * self.popularity
        return score


class GRU4RecBackbone(nn.Module):
    def __init__(
        self,
        item_num: int,
        maxlen: int,
        hidden_units: int,
        num_layers: int,
        dropout_rate: float,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.item_num = int(item_num)
        self.maxlen = int(maxlen)
        self.hidden_units = int(hidden_units)
        self.num_layers = int(num_layers)
        self.dev = device

        self.item_emb = nn.Embedding(self.item_num + 1, self.hidden_units, padding_idx=0)
        self.emb_dropout = nn.Dropout(float(dropout_rate))
        self.gru = nn.GRU(
            input_size=self.hidden_units,
            hidden_size=self.hidden_units,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=float(dropout_rate) if self.num_layers > 1 else 0.0,
        )

    def log2feats(self, log_seqs: torch.Tensor) -> torch.Tensor:
        # log_seqs: [B, L] right-padded (pad=0 at tail).
        x = self.item_emb(log_seqs)
        x = self.emb_dropout(x)
        out, _ = self.gru(x)
        return out  # [B, L, H]

    def score_all(self, log_seqs: torch.Tensor) -> torch.Tensor:
        feats = self.log2feats(log_seqs)
        lengths = log_seqs.ne(0).sum(dim=1).clamp(min=1)
        idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, feats.size(-1))
        final_feat = feats.gather(1, idx).squeeze(1)  # [B, H]
        all_items = self.item_emb.weight[1:]  # [N, H], index j => item id j+1
        return torch.matmul(final_feat, all_items.t())


class GRU4Rec(nn.Module):
    """
    GRU4Rec model for Agent4Rec simulation.

    - If a trained checkpoint exists, load it and precompute a user->item score cache.
    - Otherwise, fall back to the legacy heuristic baseline (transition matrix).
    """

    def __init__(self, args, data):
        super().__init__()
        self.args = args
        self.data = data
        self.device = torch.device(args.cuda)

        # A light popularity fallback (also used by heuristic base).
        self.popularity = np.zeros(self.data.n_items, dtype=np.float32)
        for item, users in self.data.train_item_list.items():
            self.popularity[int(item)] = float(len(users))

        self.heuristic = None
        self.backbone = None
        self.score_cache = None

        model_path = getattr(args, "model_path", "Saved")
        ckpt_path = self._resolve_checkpoint_path(dataset=args.dataset, model_path=model_path)
        if ckpt_path is None:
            print("[GRU4Rec] no checkpoint found, fallback to heuristic scores.")
            self.heuristic = GRU4RecHeuristic(args, data)
            return

        print(f"[GRU4Rec] loading checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        cfg = ckpt.get("config", {})

        item_num = int(cfg.get("item_num", self.data.n_items))
        maxlen = int(cfg.get("maxlen", 50))
        hidden_units = int(cfg.get("hidden_units", 64))
        num_layers = int(cfg.get("num_layers", 1))
        dropout_rate = float(cfg.get("dropout_rate", 0.2))

        self.backbone = GRU4RecBackbone(
            item_num=item_num,
            maxlen=maxlen,
            hidden_units=hidden_units,
            num_layers=num_layers,
            dropout_rate=dropout_rate,
            device=self.device,
        )
        self.backbone.load_state_dict(ckpt["model_state_dict"], strict=True)
        self.backbone.eval()

        seq_by_user = self._load_user_train_sequences()
        self.score_cache = self._build_score_cache(seq_by_user=seq_by_user, maxlen=maxlen, item_num=item_num)

    def _resolve_checkpoint_path(self, dataset: str, model_path: str):
        env_path = os.getenv("GRU4REC_CKPT", "").strip()
        if env_path:
            p = Path(env_path)
            if p.exists():
                return p

        model_path = str(model_path or "Saved")
        candidates = [
            Path(f"recommenders/weights/{dataset}/GRU4Rec/{model_path}/epoch=best.loo.standard.pth"),
            Path(f"recommenders/weights/{dataset}/GRU4Rec/{model_path}/epoch=best.loo.pth"),
            Path(f"recommenders/weights/{dataset}/GRU4Rec/Saved/epoch=best.loo.standard.pth"),
            Path(f"recommenders/weights/{dataset}/GRU4Rec/Saved/epoch=best.loo.pth"),
        ]
        for p in candidates:
            if p.exists():
                return p
        return None

    def _load_user_train_sequences(self) -> Dict[int, List[int]]:
        # Agent4Rec-format datasets have time-ordered training sequences in cf_data.
        out = {u: [] for u in range(self.data.n_users)}
        for u in range(self.data.n_users):
            hist = self.data.train_user_list.get(u, [])
            if not hist:
                continue
            # Reserve 0 for padding: item ids in training are 1..N.
            out[u] = [int(i) + 1 for i in hist]
        return out

    def _build_score_cache(self, seq_by_user: Dict[int, List[int]], maxlen: int, item_num: int):
        if self.backbone is None:
            return None

        # Right-padded sequence matrix.
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
                scores = self.backbone.score_all(seq_t).cpu().numpy()  # [B, item_num]
                # score col j <-> data item id j because training uses iid=data_id+1.
                use_cols = min(scores.shape[1], self.data.n_items)
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

        # Should not happen, but fail open.
        return np.tile(self.popularity[items], (len(users), 1)).astype(np.float32)
