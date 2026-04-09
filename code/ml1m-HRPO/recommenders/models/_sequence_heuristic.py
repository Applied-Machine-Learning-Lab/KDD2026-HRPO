import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn


class SequenceHeuristicBase(nn.Module):
    """Lightweight sequence recommenders for simulation-time ranking only."""

    def __init__(self, args, data, max_seq_len=30):
        super().__init__()
        self.args = args
        self.data = data
        self.max_seq_len = int(max_seq_len)
        self.n_items = int(data.n_items)

        self.popularity = np.zeros(self.n_items, dtype=np.float32)
        for item, users in data.train_item_list.items():
            self.popularity[int(item)] = float(len(users))
        pmax = float(self.popularity.max())
        if pmax > 0:
            self.popularity = self.popularity / pmax
        self.novelty = 1.0 - self.popularity

        self.transition = self._build_transition_matrix(data.train_user_list)
        self.cooccurrence = self._build_cooccurrence_matrix(data.train_user_list)

    def _build_transition_matrix(self, train_user_list):
        rows, cols, vals = [], [], []
        for _, hist in train_user_list.items():
            if hist is None or len(hist) < 2:
                continue
            for i in range(len(hist) - 1):
                a = int(hist[i])
                b = int(hist[i + 1])
                if 0 <= a < self.n_items and 0 <= b < self.n_items:
                    rows.append(a)
                    cols.append(b)
                    vals.append(1.0)
        if not rows:
            return sp.csr_matrix((self.n_items, self.n_items), dtype=np.float32)

        mat = sp.csr_matrix((vals, (rows, cols)), shape=(self.n_items, self.n_items), dtype=np.float32)
        row_sum = np.asarray(mat.sum(axis=1)).reshape(-1)
        inv = np.zeros_like(row_sum, dtype=np.float32)
        nz = row_sum > 0
        inv[nz] = 1.0 / row_sum[nz]
        return sp.diags(inv) @ mat

    def _build_cooccurrence_matrix(self, train_user_list, window=3):
        rows, cols, vals = [], [], []
        w = max(int(window), 1)
        for _, hist in train_user_list.items():
            if hist is None or len(hist) < 2:
                continue
            n = len(hist)
            for i in range(n):
                a = int(hist[i])
                if a < 0 or a >= self.n_items:
                    continue
                left = max(0, i - w)
                right = min(n, i + w + 1)
                for j in range(left, right):
                    if i == j:
                        continue
                    b = int(hist[j])
                    if 0 <= b < self.n_items:
                        rows.append(a)
                        cols.append(b)
                        vals.append(1.0)
        if not rows:
            return sp.csr_matrix((self.n_items, self.n_items), dtype=np.float32)

        mat = sp.csr_matrix((vals, (rows, cols)), shape=(self.n_items, self.n_items), dtype=np.float32)
        row_sum = np.asarray(mat.sum(axis=1)).reshape(-1)
        inv = np.zeros_like(row_sum, dtype=np.float32)
        nz = row_sum > 0
        inv[nz] = 1.0 / row_sum[nz]
        return sp.diags(inv) @ mat

    def _row_add(self, score, matrix, item_id, weight):
        if weight == 0.0:
            return
        if item_id < 0 or item_id >= self.n_items:
            return
        row = matrix.getrow(int(item_id))
        if row.nnz == 0:
            return
        score[row.indices] += float(weight) * row.data.astype(np.float32)

    def _score_user(self, user_id):
        raise NotImplementedError

    def predict(self, users, items=None):
        if items is None:
            items = np.arange(self.n_items, dtype=np.int32)
        else:
            items = np.asarray(items, dtype=np.int32)

        out = np.zeros((len(users), len(items)), dtype=np.float32)
        for i, uid in enumerate(users):
            full_score = self._score_user(int(uid))
            out[i] = full_score[items]
        return out
