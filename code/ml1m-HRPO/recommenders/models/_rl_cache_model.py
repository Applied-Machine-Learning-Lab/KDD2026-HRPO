from __future__ import annotations

import os
from pathlib import Path

import numpy as np


class RLCachedModelBase:
    """
    RL score-cache wrapper for Agent4Rec simulation.

    The training script precomputes a dense [n_users, n_items] score cache.
    During simulation, this class only serves scores through `predict`.
    """

    def __init__(self, args, data, model_name: str):
        self.args = args
        self.data = data
        self.model_name = str(model_name).upper()
        self.device = None

        self.score_cache = None
        self.popularity = np.zeros(self.data.n_items, dtype=np.float32)
        for item, users in self.data.train_item_list.items():
            self.popularity[int(item)] = float(len(users))

        ckpt_path = self._resolve_score_cache_path(args.dataset, self.model_name)
        if ckpt_path is None:
            print(f"[{self.model_name}] no score cache found, fallback to popularity.")
            return

        score = np.load(ckpt_path).astype(np.float32)
        if score.ndim != 2:
            print(f"[{self.model_name}] invalid cache shape={score.shape}, fallback to popularity.")
            return

        # Align cache shape to current Data object.
        out = np.full((self.data.n_users, self.data.n_items), -1e9, dtype=np.float32)
        u = min(self.data.n_users, score.shape[0])
        i = min(self.data.n_items, score.shape[1])
        out[:u, :i] = score[:u, :i]
        self.score_cache = out
        print(f"[{self.model_name}] loaded score cache: {ckpt_path} shape={self.score_cache.shape}")

    def _resolve_score_cache_path(self, dataset: str, model_name: str):
        env_key = f"{model_name}_SCORE_CACHE"
        env_path = os.getenv(env_key, "").strip()
        if env_path:
            p = Path(env_path)
            if p.exists():
                return p

        candidates = [
            Path(f"recommenders/weights/{dataset}/{model_name}/Saved/score_cache.npy"),
            Path(f"recommenders/weights/{dataset}/{model_name}/Saved/score_cache_topk.npy"),
        ]
        for p in candidates:
            if p.exists():
                return p
        return None

    def cuda(self, device):
        self.device = device
        return self

    def predict(self, users, items=None):
        if items is None:
            items = np.arange(self.data.n_items, dtype=np.int64)
        else:
            items = np.asarray(items, dtype=np.int64)

        users = np.asarray(users, dtype=np.int64)

        if self.score_cache is not None:
            return self.score_cache[users][:, items].astype(np.float32)

        return np.tile(self.popularity[items], (len(users), 1)).astype(np.float32)
