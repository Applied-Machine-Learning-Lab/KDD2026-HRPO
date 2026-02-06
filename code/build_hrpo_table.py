# -*- coding: utf-8 -*-
"""
HRPO prefix-utility table builder.

Goal:
  1) Group users into discrete context buckets (ctx_id) from user features.
  2) On logs, estimate prefix returns:  E[target | ctx, SID-prefix].
  3) Convert prefix expectations to *residual per depth*:
        residual_t = E[target | ctx, prefix[:t]] - E[target | ctx, prefix[:t-1]]
     which becomes a dense reward label for each SID token position.

Target options
--------------
By default, target is the *single-step* (per-row) reward computed from `--reward_weights`.

If you enable `--use_rtg`, target becomes the *future return-to-go* (RTG) inside each (user,session):
    RTG_t = sum_{k=0..} (gamma^k * r_{t+k})
Optionally truncate to a finite horizon H:
    RTG_t^H = sum_{k=0..H-1} (gamma^k * r_{t+k})

This makes HRPO "look into the future" and can stabilize long-term alignment when only single-step
rewards are available in the logs.

Output (pickle)
---------------
{
  "sid_cols": [...],
  "sid_depth": int,
  "reward_weights": {behavior: weight, ...},
  "ctx_mode": "bucket" | "kmeans",
  "uid2ctx": {user_id(int): ctx_id(int), ...},

  "ctx_count": {ctx_id: count_rows, ...},
  "ctx_sum": {ctx_id: sum_target, ...},

  "prefix_count": {(ctx_id, prefix_tuple): count, ...},
  "prefix_sum": {(ctx_id, prefix_tuple): sum_target, ...},

  "global_mean": float,
  "smoothing_alpha": float,
  "min_prefix_count": int,

  "use_rtg": bool,
  "rtg_gamma": float,
  "rtg_horizon": int,
  "session_col": str,
  "order_col": str,
}

Notes
-----
* Global prefix stats (ctx_id = -1) are used as a backoff when (ctx,prefix) is sparse.
* The builder supports an optional "unbiased/randomized" window slicing within a single session CSV
  (KuaiRand-Pure 2022-04-22 .. 2022-05-08).

"""

from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def parse_reward_weights(s: str) -> Dict[str, float]:
    """
    Accept:
      - JSON string: '{"is_click": 0.0025, "is_like": 0.001}'
      - or "k=v,k=v"
    """
    s = s.strip()
    if s.startswith("{"):
        return {k: float(v) for k, v in json.loads(s).items()}
    out = {}
    for part in s.split(","):
        if not part.strip():
            continue
        k, v = part.split("=")
        out[k.strip()] = float(v)
    return out




def compute_ctx_user_counts(uid2ctx: Dict[int, int]) -> Dict[int, int]:
    """Count how many users fall into each ctx_id (bucket size)."""
    return dict(Counter(uid2ctx.values()))


def build_ctx_bucket_ids(
    df_user: pd.DataFrame,
    ctx_features: List[str],
    max_ctx: int = 512,
    min_ctx_users: int = 10,
) -> Tuple[Dict[int, int], Dict[Tuple[str, ...], int], Dict[int, int]]:
    """
    Discrete ctx = tuple of feature strings.
    Keep top (max_ctx-1) frequent ctx keys; everything else -> ctx_id=0 (OOV).
    Return:
      uid2ctx, key2ctx, ctx_user_counts
    """
    if "user_id" not in df_user.columns:
        raise ValueError("user_feat_path must contain 'user_id' column")

    vals = []
    for c in ctx_features:
        if c not in df_user.columns:
            raise ValueError(f"user_feat missing column: {c}")
        vals.append(df_user[c].astype(str).fillna("nan").values)
    keys = list(zip(*vals))  # List[Tuple[str,...]]
    uid = df_user["user_id"].astype(int).values

    key_counter = Counter(keys)

    kept = [(k, n) for k, n in key_counter.items() if n >= min_ctx_users]
    kept.sort(key=lambda x: x[1], reverse=True)
    kept = kept[: max(0, max_ctx - 1)]

    key2ctx = {k: (i + 1) for i, (k, _) in enumerate(kept)}  # 1..K
    uid2ctx = {}
    ctx_user_counts = Counter()
    for u, k in zip(uid, keys):
        cid = key2ctx.get(k, 0)
        uid2ctx[int(u)] = int(cid)
        ctx_user_counts[int(cid)] += 1
    return uid2ctx, key2ctx, dict(ctx_user_counts)


def build_ctx_kmeans_ids(
    df_user: pd.DataFrame,
    ctx_features: List[str],
    n_clusters: int = 128,
    random_state: int = 2025,
) -> Dict[int, int]:
    """
    Optional alternative: KMeans on one-hot encoded categorical features.
    Requires sklearn.
    """
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.cluster import MiniBatchKMeans

    if "user_id" not in df_user.columns:
        raise ValueError("user_feat_path must contain 'user_id' column")

    X = df_user[ctx_features].astype(str).fillna("nan")
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    Xoh = enc.fit_transform(X)

    km = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        batch_size=4096,
        n_init="auto",
        max_iter=200,
    )
    labels = km.fit_predict(Xoh)
    uid = df_user["user_id"].astype(int).values
    return {int(u): int(l) for u, l in zip(uid, labels)}

def _infer_token_base(df_sid: pd.DataFrame, cols: List[str]) -> int:
    """Infer token base (vocab size) from sid mapping columns."""
    mx = 0
    for c in cols:
        if c not in df_sid.columns:
            continue
        v = pd.to_numeric(df_sid[c], errors="coerce").fillna(0).astype(int).max()
        mx = max(mx, int(v))
    return max(2, mx + 1)


def _pad_or_trim_user_feat_np(x: np.ndarray, target_dim: int) -> np.ndarray:
    if x.ndim != 2:
        raise ValueError(f"user_feat must be 2D, got {x.shape}")
    B, F = x.shape
    if F == target_dim:
        return x
    if F > target_dim:
        return x[:, :target_dim]
    out = np.zeros((B, target_dim), dtype=x.dtype)
    out[:, :F] = x
    return out


def build_ctx_taste_hist_ids(
    log_paths: List[str],
    df_sid: pd.DataFrame,
    df_user: pd.DataFrame,
    user_id_col: str,
    video_id_col: str,
    sid_cols: List[str],
    taste_sid_cols: List[str],
    taste_event_col: str,
    taste_event_min: float,
    embed_method: str,
    embed_dim: int,
    n_clusters: int,
    max_ctx: int,
    chunksize: int = 200000,
    random_state: int = 2025,
    unbiased_date_col: str = "",
    apply_unbiased_fn=None,
) -> Dict[int, int]:
    """
    Taste ctx v1: build per-user histogram over SID prefixes (e.g., sid_1,sid_2),
    optionally embed/topic-model, then cluster -> ctx_id.

    Users with no taste events will be assigned ctx_id=0 (OOV).
    Cluster ids are shifted by +1 to avoid collision with OOV=0.
    """
    from sklearn.cluster import MiniBatchKMeans

    if user_id_col not in df_user.columns:
        raise ValueError(f"user_feat_path must contain '{user_id_col}' column")
    user_ids = df_user[user_id_col].astype(int).values
    uid2row = {int(u): i for i, u in enumerate(user_ids.tolist())}
    n_users = len(user_ids)

    for c in taste_sid_cols:
        if c not in sid_cols:
            raise ValueError(f"--taste_sid_cols must be subset of --sid_cols. got {taste_sid_cols}, sid_cols={sid_cols}")

    base = _infer_token_base(df_sid, taste_sid_cols)
    L = len(taste_sid_cols)
    V = int(base ** L)
    if V > 200000:
        raise ValueError(
            f"taste histogram dim too large: base^{L}={V}. "
            f"Reduce --taste_sid_cols (e.g. sid_1,sid_2) or ensure tokens are small."
        )

    counts = np.zeros((n_users, V), dtype=np.float32)

    read_cols = {user_id_col, video_id_col}
    if taste_event_col:
        read_cols.add(taste_event_col)
    if unbiased_date_col:
        read_cols.add(unbiased_date_col)

    for p in log_paths:
        print(f"[CTX][taste_hist] scan log: {p}")
        for chunk in pd.read_csv(p, chunksize=chunksize, usecols=lambda c: c in read_cols):
            if apply_unbiased_fn is not None:
                chunk = apply_unbiased_fn(chunk)
            if len(chunk) == 0:
                continue

            chunk = chunk.merge(df_sid[[video_id_col] + taste_sid_cols], on=video_id_col, how="inner")
            if len(chunk) == 0:
                continue

            if taste_event_col:
                if taste_event_col not in chunk.columns:
                    continue
                ev = pd.to_numeric(chunk[taste_event_col], errors="coerce").fillna(0.0).values.astype(np.float32)
                chunk = chunk[ev >= float(taste_event_min)]
                if len(chunk) == 0:
                    continue

            chunk[user_id_col] = chunk[user_id_col].astype(int)
            row = chunk[user_id_col].map(uid2row)
            chunk = chunk[~row.isna()].copy()
            if len(chunk) == 0:
                continue
            row = row.dropna().astype(int).values

            tok = chunk[taste_sid_cols].apply(pd.to_numeric, errors="coerce").fillna(0).astype(int).values
            cat = np.zeros(len(tok), dtype=np.int64)
            for j in range(L):
                cat = cat * base + tok[:, j].astype(np.int64)

            df_tmp = pd.DataFrame({"row": row, "cat": cat})
            g = df_tmp.groupby(["row", "cat"], sort=False).size().reset_index(name="cnt")
            counts[g["row"].values, g["cat"].values] += g["cnt"].values.astype(np.float32)

    row_sum = counts.sum(axis=1)
    ok = row_sum > 0
    n_ok = int(ok.sum())
    print(f"[CTX][taste_hist] users with taste events: {n_ok}/{n_users}, V={V}, base={base}, L={L}")

    if n_ok == 0:
        return {int(u): 0 for u in user_ids.tolist()}

    X = counts[ok]
    if embed_method == "raw":
        emb = X
    elif embed_method == "svd":
        from sklearn.decomposition import TruncatedSVD
        n_comp = min(int(embed_dim), max(2, X.shape[1] - 1))
        svd = TruncatedSVD(n_components=n_comp, random_state=random_state)
        emb = svd.fit_transform(X)
    elif embed_method == "nmf":
        from sklearn.decomposition import NMF
        n_comp = min(int(embed_dim), max(2, X.shape[1] // 4))
        nmf = NMF(n_components=n_comp, init="nndsvda", max_iter=200, random_state=random_state)
        emb = nmf.fit_transform(X)
    elif embed_method == "lda":
        from sklearn.decomposition import LatentDirichletAllocation
        lda = LatentDirichletAllocation(n_components=int(embed_dim), random_state=random_state, learning_method="batch", max_iter=20)
        emb = lda.fit_transform(X)
    else:
        raise ValueError(f"Unknown --taste_embed_method={embed_method}")

    K = int(n_clusters)
    if max_ctx > 1:
        K = min(K, int(max_ctx) - 1)
    K = min(K, n_ok)  # cannot exceed samples
    if K <= 1:
        uid2ctx = {int(u): 1 for u in user_ids.tolist()}
        for u, has in zip(user_ids.tolist(), ok.tolist()):
            if not has:
                uid2ctx[int(u)] = 0
        return uid2ctx

    km = MiniBatchKMeans(n_clusters=K, random_state=random_state, batch_size=4096, n_init="auto", max_iter=200)
    labels = km.fit_predict(emb)

    uid2ctx: Dict[int, int] = {}
    ok_users = user_ids[ok]
    for u, l in zip(ok_users.tolist(), labels.tolist()):
        uid2ctx[int(u)] = int(l) + 1
    for u, has in zip(user_ids.tolist(), ok.tolist()):
        if not has:
            uid2ctx[int(u)] = 0

    print(f"[CTX][taste_hist] built ctx_id in [0..{K}] (0=OOV)")
    return uid2ctx


def _infer_onerec_hparams_from_state(state: Dict[str, "torch.Tensor"]) -> Dict[str, int]:
    import re
    hp: Dict[str, int] = {}
    if "sid_embedding.weight" in state:
        hp["num_classes"] = int(state["sid_embedding.weight"].shape[0])
        hp["hid_dim"] = int(state["sid_embedding.weight"].shape[1])
    if "sid_pos_embedding.weight" in state:
        hp["sid_depth"] = int(state["sid_pos_embedding.weight"].shape[0])
    if "hist_pos_embedding.weight" in state:
        hp["max_hist_len"] = int(state["hist_pos_embedding.weight"].shape[0])
    if "user_proj.weight" in state:
        hp["user_feat_dim"] = int(state["user_proj.weight"].shape[1])

    def _infer_layers(prefix: str) -> int:
        pat = re.compile(re.escape(prefix) + r"\.net\.layers\.(\d+)\.")
        mx = -1
        for k in state.keys():
            m = pat.search(k)
            if m:
                mx = max(mx, int(m.group(1)))
        return max(1, mx + 1)

    hp["num_decoder_block"] = _infer_layers("decoder")
    hp["hist_num_layers"] = _infer_layers("hist_encoder")
    return hp


def build_ctx_taste_onerec_ids(
    log_paths: List[str],
    df_sid: pd.DataFrame,
    df_user: pd.DataFrame,
    user_id_col: str,
    video_id_col: str,
    sid_cols: List[str],
    taste_event_col: str,
    taste_event_min: float,
    n_clusters: int,
    max_ctx: int,
    onerec_ckpt: str,
    onerec_code_root: str = "",
    onerec_device: str = "cpu",
    onerec_batch_size: int = 1024,
    use_user_feat: bool = False,
    concat_user_vec: bool = False,
    taste_max_hist_len: int = 50,
    chunksize: int = 200000,
    random_state: int = 2025,
    unbiased_date_col: str = "",
    apply_unbiased_fn=None,
) -> Dict[int, int]:
    """
    Taste ctx v2: use a trained OneRec encoder to extract user taste vectors from their (clicked) history,
    then cluster -> ctx_id.

    Users with no taste events will be assigned ctx_id=0 (OOV).
    Cluster ids are shifted by +1.
    """
    import torch
    from sklearn.cluster import MiniBatchKMeans
    import sys
    from collections import deque

    if not onerec_ckpt:
        raise ValueError("--ctx_mode=taste_onerec requires --onerec_ckpt")

    if onerec_code_root:
        sys.path.insert(0, onerec_code_root)

    try:
        from model.onerec import OneRecSIDWithContext
    except Exception as e:
        raise ImportError(
            "Failed to import model.onerec.OneRecSIDWithContext. "
            "Set --onerec_code_root to your repo root that contains 'model/' package."
        ) from e

    ckpt = torch.load(onerec_ckpt, map_location="cpu")
    state = ckpt.get("model", ckpt)
    hp = _infer_onerec_hparams_from_state(state)

    if len(sid_cols) != int(hp.get("sid_depth", len(sid_cols))):
        raise ValueError(f"[CTX][taste_onerec] sid_depth mismatch: sid_cols={len(sid_cols)} but ckpt sid_depth={hp.get('sid_depth')}")

    nhead = None
    if isinstance(ckpt, dict) and isinstance(ckpt.get("args", None), dict):
        nhead = int(ckpt["args"].get("nhead", 0) or 0)
    if not nhead or hp["hid_dim"] % nhead != 0:
        for cand in [8, 4, 2, 1]:
            if hp["hid_dim"] % cand == 0:
                nhead = cand
                break

    model = OneRecSIDWithContext(
        num_decoder_block=hp["num_decoder_block"],
        hid_dim=hp["hid_dim"],
        nhead=int(nhead),
        sid_depth=hp["sid_depth"],
        num_classes=hp["num_classes"],
        user_feat_dim=hp["user_feat_dim"],
        max_hist_len=int(hp["max_hist_len"]),
        hist_num_layers=hp["hist_num_layers"],
        dropout_ratio=0.0,
    )
    miss, unexp = model.load_state_dict(state, strict=False)
    print(
        f"[CTX][taste_onerec] load ckpt: missing={len(miss)} unexpected={len(unexp)} "
        f"hid_dim={hp['hid_dim']} num_classes={hp['num_classes']} sid_depth={hp['sid_depth']}"
    )

    device = torch.device(onerec_device)
    model = model.to(device).eval()

    if user_id_col not in df_user.columns:
        raise ValueError(f"user_feat_path must contain '{user_id_col}' column")
    user_ids = df_user[user_id_col].astype(int).values
    uid2row = {int(u): i for i, u in enumerate(user_ids.tolist())}
    n_users = len(user_ids)

    H_model = int(hp["max_hist_len"])
    H_use = min(int(taste_max_hist_len), H_model)
    deqs = [deque(maxlen=H_use) for _ in range(n_users)]

    read_cols = {user_id_col, video_id_col}
    if taste_event_col:
        read_cols.add(taste_event_col)
    if unbiased_date_col:
        read_cols.add(unbiased_date_col)

    for p in log_paths:
        print(f"[CTX][taste_onerec] scan log: {p}")
        for chunk in pd.read_csv(p, chunksize=chunksize, usecols=lambda c: c in read_cols):
            if apply_unbiased_fn is not None:
                chunk = apply_unbiased_fn(chunk)
            if len(chunk) == 0:
                continue

            chunk = chunk.merge(df_sid[[video_id_col] + sid_cols], on=video_id_col, how="inner")
            if len(chunk) == 0:
                continue

            if taste_event_col:
                if taste_event_col not in chunk.columns:
                    continue
                ev = pd.to_numeric(chunk[taste_event_col], errors="coerce").fillna(0.0).values.astype(np.float32)
                chunk = chunk[ev >= float(taste_event_min)]
                if len(chunk) == 0:
                    continue

            chunk[user_id_col] = chunk[user_id_col].astype(int)
            row = chunk[user_id_col].map(uid2row)
            chunk = chunk[~row.isna()].copy()
            if len(chunk) == 0:
                continue
            chunk["_row"] = row.dropna().astype(int).values

            for r, g in chunk.groupby("_row", sort=False):
                sids = g[sid_cols].apply(pd.to_numeric, errors="coerce").fillna(0).astype(int).values
                dq = deqs[int(r)]
                for sid in sids:
                    dq.append(tuple(int(x) for x in sid.tolist()))

    H = H_use
    L = len(sid_cols)
    hist = np.zeros((n_users, H, L), dtype=np.int64)
    hist_len = np.zeros((n_users,), dtype=np.int64)
    for i, dq in enumerate(deqs):
        seq = list(dq)
        ln = len(seq)
        hist_len[i] = ln
        if ln > 0:
            hist[i, :ln, :] = np.asarray(seq, dtype=np.int64)

    ok = hist_len > 0
    n_ok = int(ok.sum())
    print(f"[CTX][taste_onerec] users with taste events: {n_ok}/{n_users}, H={H}, L={L}")

    if n_ok == 0:
        return {int(u): 0 for u in user_ids.tolist()}

    user_feat = None
    if use_user_feat:
        feat_cols = [c for c in df_user.columns if c != user_id_col]
        uf = df_user[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).values.astype(np.float32)
        uf = _pad_or_trim_user_feat_np(uf, int(hp["user_feat_dim"]))
        user_feat = uf

    emb_dim = int(hp["hid_dim"]) * (2 if concat_user_vec else 1)
    emb = np.zeros((n_users, emb_dim), dtype=np.float32)

    with torch.no_grad():
        bs = int(onerec_batch_size)
        for st in range(0, n_users, bs):
            ed = min(n_users, st + bs)
            h = torch.from_numpy(hist[st:ed]).to(device=device, dtype=torch.long)
            hl = torch.from_numpy(hist_len[st:ed]).to(device=device, dtype=torch.long)
            uctx = None
            uvec = None
            if use_user_feat and (user_feat is not None):
                uf = torch.from_numpy(user_feat[st:ed]).to(device=device, dtype=torch.float32)
                uvec = model.user_proj(uf)
                uctx = uvec
            enc_out, mask = model.encode_history_seq(h, hl, user_ctx=uctx)
            valid = (~mask).float().unsqueeze(-1)
            pooled = (enc_out * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
            if concat_user_vec and (uvec is not None):
                pooled = torch.cat([pooled, uvec], dim=1)
            emb[st:ed] = pooled.detach().cpu().numpy().astype(np.float32)

    X = emb[ok]
    K = int(n_clusters)
    if max_ctx > 1:
        K = min(K, int(max_ctx) - 1)
    K = min(K, n_ok)
    if K <= 1:
        uid2ctx = {int(u): 1 for u in user_ids.tolist()}
        for u, has in zip(user_ids.tolist(), ok.tolist()):
            if not has:
                uid2ctx[int(u)] = 0
        return uid2ctx

    km = MiniBatchKMeans(n_clusters=K, random_state=random_state, batch_size=4096, n_init="auto", max_iter=200)
    labels = km.fit_predict(X)

    uid2ctx: Dict[int, int] = {}
    ok_users = user_ids[ok]
    for u, l in zip(ok_users.tolist(), labels.tolist()):
        uid2ctx[int(u)] = int(l) + 1
    for u, has in zip(user_ids.tolist(), ok.tolist()):
        if not has:
            uid2ctx[int(u)] = 0

    print(f"[CTX][taste_onerec] built ctx_id in [0..{K}] (0=OOV)")
    return uid2ctx


def compute_rtg_sorted(
    user_ids: np.ndarray,
    session_ids: np.ndarray,
    rewards: np.ndarray,
    gamma: float = 1.0,
    horizon: int = 0,
) -> np.ndarray:
    """
    Compute RTG in O(N) given rows are sorted by (user_id, session_id, time/position) ascending.

    If horizon <= 0: infinite-horizon-to-session-end RTG.
    If horizon > 0 : truncate to that many future steps.
    """
    user_ids = user_ids.astype(np.int64, copy=False)
    session_ids = session_ids.astype(np.int64, copy=False)
    r = rewards.astype(np.float64, copy=False)

    n = len(r)
    full = np.zeros(n, dtype=np.float64)

    next_val = 0.0
    prev_u = None
    prev_s = None
    for i in range(n - 1, -1, -1):
        u = int(user_ids[i])
        s = int(session_ids[i])
        if i == n - 1 or u != prev_u or s != prev_s:
            next_val = 0.0
        full[i] = float(r[i]) + float(gamma) * float(next_val)
        next_val = full[i]
        prev_u, prev_s = u, s

    if horizon is None or int(horizon) <= 0:
        return full

    H = int(horizon)
    gamma_h = float(gamma) ** H
    shift = np.zeros(n, dtype=np.float64)

    start = 0
    while start < n:
        u = int(user_ids[start])
        s = int(session_ids[start])
        end = start + 1
        while end < n and int(user_ids[end]) == u and int(session_ids[end]) == s:
            end += 1
        seg_len = end - start
        if seg_len > H:
            shift[start : end - H] = full[start + H : end]
        start = end

    return full - gamma_h * shift


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log_paths", nargs="+", required=True, help="One or more CSV logs.")
    ap.add_argument("--sid_mapping_path", type=str, required=True)
    ap.add_argument("--user_feat_path", type=str, required=True)

    ap.add_argument("--sid_cols", type=str, default="sid_1,sid_2,sid_3,sid_4")
    ap.add_argument("--user_id_col", type=str, default="user_id")
    ap.add_argument("--video_id_col", type=str, default="video_id")

    ap.add_argument("--session_col", type=str, default="session", help="Session id column used for RTG.")
    ap.add_argument("--time_col", type=str, default="time_ms", help="Preferred ordering column (e.g., time_ms).")
    ap.add_argument("--position_col", type=str, default="position", help="Fallback ordering column if time_col missing.")

    ap.add_argument("--date_col", type=str, default="date", help="Date column for unbiased slicing (yyyymmdd).")
    ap.add_argument("--use_unbiased", action="store_true",
                    help="If set, only use rows whose date_col is in [unbiased_start, unbiased_end] (inclusive).")
    ap.add_argument("--unbiased_start", type=int, default=20220422)
    ap.add_argument("--unbiased_end", type=int, default=20220508)

    ap.add_argument("--reward_weights", type=str, default='{"is_click":1.0}',
                    help='JSON like {"is_click":0.0025,"is_like":0.001} or k=v,k=v')

    ap.add_argument(
        "--ctx_features",
        type=str,
        default="user_active_degree,fans_user_num_range,register_days_range,is_video_author,is_live_streamer",
        help="Comma-separated user feature columns used to define ctx buckets / clustering.",
    )
    ap.add_argument("--ctx_mode", type=str, choices=["bucket", "kmeans", "taste_hist", "taste_onerec"], default="bucket")
    ap.add_argument("--max_ctx", type=int, default=512)
    ap.add_argument("--min_ctx_users", type=int, default=1)
    ap.add_argument("--kmeans_k", type=int, default=128)
    ap.add_argument(
        "--taste_sid_cols",
        type=str,
        default="sid_1,sid_2",
        help="Comma-separated SID columns used to build taste histogram (must be subset of --sid_cols).",
    )
    ap.add_argument(
        "--taste_event_col",
        type=str,
        default="is_click",
        help="Only rows with taste_event_col >= taste_event_min will be used to build taste ctx. Set empty to disable filter.",
    )
    ap.add_argument("--taste_event_min", type=float, default=1.0)
    ap.add_argument("--taste_embed_method", type=str, default="svd", choices=["raw", "svd", "nmf", "lda"])
    ap.add_argument("--taste_dim", type=int, default=32, help="Embedding/topic dim before clustering (taste_hist).")
    ap.add_argument("--taste_max_hist_len", type=int, default=50, help="Max history length per user (taste_onerec).")

    ap.add_argument("--onerec_ckpt", type=str, default="", help="Path to a trained OneRec/OneRecWithValue checkpoint .pt")
    ap.add_argument("--onerec_code_root", type=str, default="", help="Optional: repo root to add into sys.path for importing model.*")
    ap.add_argument("--onerec_device", type=str, default="cpu")
    ap.add_argument("--onerec_batch_size", type=int, default=1024)
    ap.add_argument("--onerec_use_user_feat", action="store_true", help="Fuse projected user_feat into encoder when extracting taste vectors.")
    ap.add_argument("--onerec_concat_user_vec", action="store_true", help="If set, cluster on concat([pooled_hist_vec, user_vec]).")

    ap.add_argument("--chunksize", type=int, default=200000, help="CSV chunksize for streaming build (non-RTG).")

    ap.add_argument("--min_prefix_count", type=int, default=1)
    ap.add_argument("--smoothing_alpha", type=float, default=100.0)

    ap.add_argument("--use_rtg", action="store_true",
                    help="If set, aggregate using per-row return-to-go within (user,session).")
    ap.add_argument("--rtg_gamma", type=float, default=1.0, help="Discount factor gamma for RTG.")
    ap.add_argument("--rtg_horizon", type=int, default=0, help="If >0, truncate RTG to this horizon length.")

    ap.add_argument("--out_path", type=str, required=True)
    ap.add_argument(
        "--bucket_stats_out_path",
        type=str,
        default="",
        help="Optional: also dump a lightweight pkl with only ctx bucket percentile stats (users/logs/event_sum).",
    )

    ap.add_argument(
        "--no_cohorting",
        action="store_true",
        help=(
            "Disable cohort conditioning by forcing all users into a single ctx_id=0. "
            "This yields a global (non-cohort) prefix table while keeping all other logic unchanged."
        ),
    )
    args = ap.parse_args()

    sid_cols = [s.strip() for s in args.sid_cols.split(",") if s.strip()]
    ctx_features = [s.strip() for s in args.ctx_features.split(",") if s.strip()]
    reward_w = parse_reward_weights(args.reward_weights)

    use_unbiased = bool(getattr(args, "use_unbiased", False))
    date_col = getattr(args, "date_col", "date")
    unbiased_start = int(getattr(args, "unbiased_start", 20220422))
    unbiased_end = int(getattr(args, "unbiased_end", 20220508))
    if use_unbiased:
        print(f"[HRPO] USE_UNBIASED=1 slicing {date_col} in [{unbiased_start}, {unbiased_end}] (inclusive)")

    use_rtg = bool(getattr(args, "use_rtg", False))
    if use_rtg:
        print(f"[HRPO] USE_RTG=1 -> target=RTG (gamma={args.rtg_gamma}, horizon={args.rtg_horizon})")

    print("[HRPO] loading sid mapping:", args.sid_mapping_path)
    df_sid = pd.read_csv(args.sid_mapping_path)
    need_cols = [args.video_id_col] + sid_cols
    for c in need_cols:
        if c not in df_sid.columns:
            raise ValueError(f"sid_mapping missing col: {c}")
    df_sid = df_sid[need_cols].copy()

    print("[HRPO] loading user features:", args.user_feat_path)
    df_user = pd.read_csv(args.user_feat_path)
    def _apply_unbiased_slice_ctx(df: pd.DataFrame) -> pd.DataFrame:
        if not use_unbiased:
            return df
        if date_col not in df.columns:
            return df
        d = pd.to_numeric(df[date_col], errors="coerce").fillna(0).astype(int)
        return df[(d >= int(unbiased_start)) & (d <= int(unbiased_end))]

    if args.ctx_mode == "bucket":
        uid2ctx, _key2ctx, _ctx_user_counts = build_ctx_bucket_ids(
            df_user=df_user,
            ctx_features=ctx_features,
            max_ctx=args.max_ctx,
            min_ctx_users=args.min_ctx_users,
        )
        ctx_mode_meta = {"ctx_mode": "bucket", "max_ctx": args.max_ctx, "min_ctx_users": args.min_ctx_users}

    elif args.ctx_mode == "kmeans":
        uid2ctx = build_ctx_kmeans_ids(
            df_user=df_user,
            ctx_features=ctx_features,
            n_clusters=args.kmeans_k,
        )
        ctx_mode_meta = {"ctx_mode": "kmeans", "kmeans_k": args.kmeans_k}

    elif args.ctx_mode == "taste_hist":
        taste_sid_cols = [s.strip() for s in args.taste_sid_cols.split(",") if s.strip()]
        uid2ctx = build_ctx_taste_hist_ids(
            log_paths=args.log_paths,
            df_sid=df_sid,
            df_user=df_user,
            user_id_col=args.user_id_col,
            video_id_col=args.video_id_col,
            sid_cols=sid_cols,
            taste_sid_cols=taste_sid_cols,
            taste_event_col=str(args.taste_event_col or "").strip(),
            taste_event_min=float(args.taste_event_min),
            embed_method=str(args.taste_embed_method),
            embed_dim=int(args.taste_dim),
            n_clusters=int(args.kmeans_k),
            max_ctx=int(args.max_ctx),
            chunksize=int(args.chunksize),
            random_state=2025,
            unbiased_date_col=(date_col if use_unbiased else ""),
            apply_unbiased_fn=_apply_unbiased_slice_ctx,
        )
        ctx_mode_meta = {
            "ctx_mode": "taste_hist",
            "kmeans_k": int(args.kmeans_k),
            "taste_sid_cols": taste_sid_cols,
            "taste_event_col": str(args.taste_event_col),
            "taste_event_min": float(args.taste_event_min),
            "taste_embed_method": str(args.taste_embed_method),
            "taste_dim": int(args.taste_dim),
        }

    elif args.ctx_mode == "taste_onerec":
        uid2ctx = build_ctx_taste_onerec_ids(
            log_paths=args.log_paths,
            df_sid=df_sid,
            df_user=df_user,
            user_id_col=args.user_id_col,
            video_id_col=args.video_id_col,
            sid_cols=sid_cols,
            taste_event_col=str(args.taste_event_col or "").strip(),
            taste_event_min=float(args.taste_event_min),
            n_clusters=int(args.kmeans_k),
            max_ctx=int(args.max_ctx),
            onerec_ckpt=str(args.onerec_ckpt),
            onerec_code_root=str(args.onerec_code_root),
            onerec_device=str(args.onerec_device),
            onerec_batch_size=int(args.onerec_batch_size),
            use_user_feat=bool(args.onerec_use_user_feat),
            concat_user_vec=bool(args.onerec_concat_user_vec),
            taste_max_hist_len=int(args.taste_max_hist_len),
            chunksize=int(args.chunksize),
            random_state=2025,
            unbiased_date_col=(date_col if use_unbiased else ""),
            apply_unbiased_fn=_apply_unbiased_slice_ctx,
        )
        ctx_mode_meta = {
            "ctx_mode": "taste_onerec",
            "kmeans_k": int(args.kmeans_k),
            "taste_event_col": str(args.taste_event_col),
            "taste_event_min": float(args.taste_event_min),
            "taste_max_hist_len": int(args.taste_max_hist_len),
            "onerec_ckpt": str(args.onerec_ckpt),
            "onerec_device": str(args.onerec_device),
            "onerec_use_user_feat": bool(args.onerec_use_user_feat),
            "onerec_concat_user_vec": bool(args.onerec_concat_user_vec),
        }
    else:
        raise ValueError(f"Unknown --ctx_mode={args.ctx_mode}")

    if bool(getattr(args, "no_cohorting", False)):
        uid_col = args.user_id_col if (args.user_id_col in df_user.columns) else "user_id"
        if uid_col not in df_user.columns:
            raise ValueError(f"[HRPO][no_cohorting] user_feat missing uid column: '{uid_col}'")
        uids = df_user[uid_col].astype(int).values
        uid2ctx = {int(u): 0 for u in uids.tolist()}
        ctx_mode_meta = {**ctx_mode_meta, "no_cohorting": True}
        print("[HRPO][CFG] no_cohorting=1 -> force ctx_id=0 for all users (global prefix table)")

    print(f"[HRPO] ctx_mode={args.ctx_mode}, ctx_count(unique)~{len(set(uid2ctx.values()))}")

    ctx_user_counts = compute_ctx_user_counts(uid2ctx)
    _sizes = np.asarray(list(ctx_user_counts.values()), dtype=np.float64)
    _pcts = np.arange(0, 101, 1, dtype=np.int64)  # 0%..100%
    if _sizes.size == 0:
        ctx_user_count_percentiles = {int(p): 0 for p in _pcts.tolist()}
    else:
        try:
            _q = np.percentile(_sizes, _pcts, method="linear")
        except TypeError:
            _q = np.percentile(_sizes, _pcts, interpolation="linear")
        ctx_user_count_percentiles = {int(p): int(round(v)) for p, v in zip(_pcts.tolist(), _q.tolist())}

    print(
        "[HRPO] ctx bucket user-count percentiles (p0/p25/p50/p75/p90/p100): "
        f"{ctx_user_count_percentiles[0]}/"
        f"{ctx_user_count_percentiles[25]}/"
        f"{ctx_user_count_percentiles[50]}/"
        f"{ctx_user_count_percentiles[75]}/"
        f"{ctx_user_count_percentiles[90]}/"
        f"{ctx_user_count_percentiles[100]}"
    )


    prefix_count: Dict[Tuple[int, Tuple[int, ...]], int] = defaultdict(int)
    prefix_sum: Dict[Tuple[int, Tuple[int, ...]], float] = defaultdict(float)
    ctx_count: Dict[int, int] = defaultdict(int)
    ctx_sum: Dict[int, float] = defaultdict(float)

    total_count = 0
    total_sum = 0.0
    sid_depth = len(sid_cols)

    base_cols = {args.user_id_col, args.video_id_col}
    for k in reward_w.keys():
        base_cols.add(k)
    if use_unbiased:
        base_cols.add(date_col)
    if use_rtg:
        base_cols.add(args.session_col)
        if args.time_col:
            base_cols.add(args.time_col)
        if args.position_col:
            base_cols.add(args.position_col)

    def _apply_unbiased_slice(df: pd.DataFrame) -> pd.DataFrame:
        if not use_unbiased:
            return df
        if date_col not in df.columns:
            raise ValueError(f"use_unbiased requires date_col='{date_col}' in log; available cols: {list(df.columns)[:30]}")
        d = pd.to_numeric(df[date_col], errors="coerce")
        m = (d >= unbiased_start) & (d <= unbiased_end)
        if not bool(m.all()):
            df = df.loc[m].copy()
        return df

    def _compute_scalar_reward(df: pd.DataFrame) -> np.ndarray:
        r = np.zeros(len(df), dtype=np.float64)
        for k, w in reward_w.items():
            if k not in df.columns:
                continue
            r += pd.to_numeric(df[k], errors="coerce").fillna(0.0).values.astype(np.float64) * float(w)
        return r

    def _accumulate_from_df(df: pd.DataFrame) -> None:
        nonlocal total_count, total_sum

        if len(df) == 0:
            return

        total_count += int(len(df))
        total_sum += float(df["_r"].sum())

        g_ctx = df.groupby("ctx_id")["_r"].agg(["count", "sum"])
        for ctx_id, row in g_ctx.iterrows():
            ctx_id = int(ctx_id)
            ctx_count[ctx_id] += int(row["count"])
            ctx_sum[ctx_id] += float(row["sum"])

        for t in range(1, sid_depth + 1):
            gcols = ["ctx_id"] + sid_cols[:t]
            g = df.groupby(gcols)["_r"].agg(["count", "sum"])
            for idx, row in g.iterrows():
                if not isinstance(idx, tuple):
                    idx = (idx,)
                ctx = int(idx[0])
                pref = tuple(int(x) for x in idx[1:])
                key = (ctx, pref)
                prefix_count[key] += int(row["count"])
                prefix_sum[key] += float(row["sum"])

            g_g = df.groupby(sid_cols[:t])["_r"].agg(["count", "sum"])
            for idx, row in g_g.iterrows():
                if not isinstance(idx, tuple):
                    idx = (idx,)
                pref = tuple(int(x) for x in idx)
                key = (-1, pref)
                prefix_count[key] += int(row["count"])
                prefix_sum[key] += float(row["sum"])

    print("[HRPO] building from logs:", args.log_paths)

    if use_rtg:
        for p in args.log_paths:
            print("[HRPO][RTG] loading:", p)
            try:
                df = pd.read_csv(p, usecols=lambda c: c in base_cols)
            except TypeError:
                df = pd.read_csv(p, usecols=list(base_cols))
            df = _apply_unbiased_slice(df)
            if len(df) == 0:
                continue

            df = df.merge(df_sid, on=args.video_id_col, how="inner")
            if len(df) == 0:
                continue

            missing = [c for c in (base_cols | set(sid_cols)) if c not in df.columns]
            if missing:
                raise ValueError(f"log missing cols: {missing}")

            df[args.user_id_col] = df[args.user_id_col].astype(int)
            df["ctx_id"] = df[args.user_id_col].map(uid2ctx).fillna(0).astype(int)

            r0 = _compute_scalar_reward(df)
            df["_r0"] = r0

            if args.session_col not in df.columns:
                raise ValueError(f"--use_rtg requires session_col='{args.session_col}' in log.")
            order_col = None
            if args.time_col and args.time_col in df.columns:
                order_col = args.time_col
            elif args.position_col and args.position_col in df.columns:
                order_col = args.position_col
            else:
                order_col = None

            if order_col is not None:
                df = df.sort_values([args.user_id_col, args.session_col, order_col], kind="mergesort")
            else:
                df = df.sort_values([args.user_id_col, args.session_col], kind="mergesort")

            rtg = compute_rtg_sorted(
                user_ids=df[args.user_id_col].values,
                session_ids=df[args.session_col].values,
                rewards=df["_r0"].values,
                gamma=float(args.rtg_gamma),
                horizon=int(args.rtg_horizon),
            )
            df["_r"] = rtg

            _accumulate_from_df(df)

    else:
        for p in args.log_paths:
            print("[HRPO][STREAM] ->", p)
            for chunk in pd.read_csv(p, chunksize=args.chunksize, usecols=lambda c: c in base_cols):
                chunk = _apply_unbiased_slice(chunk)
                if len(chunk) == 0:
                    continue

                chunk = chunk.merge(df_sid, on=args.video_id_col, how="inner")
                if len(chunk) == 0:
                    continue

                needed = list({args.user_id_col, args.video_id_col} | set(reward_w.keys()))
                missing = [c for c in needed if c not in chunk.columns]
                if missing:
                    raise ValueError(f"log missing cols: {missing}")

                chunk[args.user_id_col] = chunk[args.user_id_col].astype(int)
                chunk["ctx_id"] = chunk[args.user_id_col].map(uid2ctx).fillna(0).astype(int)

                chunk["_r"] = _compute_scalar_reward(chunk)

                _accumulate_from_df(chunk)

    global_mean = total_sum / max(1, total_count)
    print(f"[HRPO] total rows={total_count} global_mean={global_mean:.6g}")
    print(f"[HRPO] num(prefix keys)={len(prefix_count)} num(ctx)={len(ctx_count)}")

    all_ctx_ids = sorted(ctx_user_counts.keys())
    _log_sizes = np.asarray([ctx_count.get(int(c), 0) for c in all_ctx_ids], dtype=np.float64)
    _event_sizes = np.asarray([ctx_sum.get(int(c), 0.0) for c in all_ctx_ids], dtype=np.float64)

    _pcts2 = np.arange(0, 101, 1, dtype=np.int64)  # 0%..100%
    if _log_sizes.size == 0:
        ctx_log_count_percentiles = {int(p): 0 for p in _pcts2.tolist()}
        ctx_event_sum_percentiles = {int(p): 0.0 for p in _pcts2.tolist()}
    else:
        try:
            _q_log = np.percentile(_log_sizes, _pcts2, method="linear")
            _q_evt = np.percentile(_event_sizes, _pcts2, method="linear")
        except TypeError:
            _q_log = np.percentile(_log_sizes, _pcts2, interpolation="linear")
            _q_evt = np.percentile(_event_sizes, _pcts2, interpolation="linear")

        ctx_log_count_percentiles = {int(p): int(round(v)) for p, v in zip(_pcts2.tolist(), _q_log.tolist())}
        ctx_event_sum_percentiles = {int(p): float(v) for p, v in zip(_pcts2.tolist(), _q_evt.tolist())}

    ctx_bucket_profile_percentiles = {
        int(p): {
            "num_users": int(ctx_user_count_percentiles.get(int(p), 0)),
            "num_logs": int(ctx_log_count_percentiles.get(int(p), 0)),
            "event_sum": float(ctx_event_sum_percentiles.get(int(p), 0.0)),
        }
        for p in _pcts2.tolist()
    }

    print(
        "[HRPO] ctx bucket log-count percentiles (p0/p25/p50/p75/p90/p100): "
        f"{ctx_log_count_percentiles[0]}/"
        f"{ctx_log_count_percentiles[25]}/"
        f"{ctx_log_count_percentiles[50]}/"
        f"{ctx_log_count_percentiles[75]}/"
        f"{ctx_log_count_percentiles[90]}/"
        f"{ctx_log_count_percentiles[100]}"
    )
    print(
        "[HRPO] ctx bucket event-sum percentiles (p0/p25/p50/p75/p90/p100): "
        f"{ctx_event_sum_percentiles[0]:.3g}/"
        f"{ctx_event_sum_percentiles[25]:.3g}/"
        f"{ctx_event_sum_percentiles[50]:.3g}/"
        f"{ctx_event_sum_percentiles[75]:.3g}/"
        f"{ctx_event_sum_percentiles[90]:.3g}/"
        f"{ctx_event_sum_percentiles[100]:.3g}"
    )

    payload = {
        "sid_cols": sid_cols,
        "sid_depth": sid_depth,
        "reward_weights": reward_w,
        **ctx_mode_meta,
        "uid2ctx": {int(k): int(v) for k, v in uid2ctx.items()},
        "ctx_user_counts": {int(k): int(v) for k, v in ctx_user_counts.items()},
        "ctx_user_count_percentiles": {int(k): int(v) for k, v in ctx_user_count_percentiles.items()},
        "ctx_log_count_percentiles": {int(k): int(v) for k, v in ctx_log_count_percentiles.items()},
        "ctx_event_sum_percentiles": {int(k): float(v) for k, v in ctx_event_sum_percentiles.items()},
        "ctx_bucket_profile_percentiles": ctx_bucket_profile_percentiles,
        "ctx_count": dict(ctx_count),
        "ctx_sum": dict(ctx_sum),
        "prefix_count": dict(prefix_count),
        "prefix_sum": dict(prefix_sum),
        "global_mean": float(global_mean),
        "smoothing_alpha": float(args.smoothing_alpha),
        "min_prefix_count": int(args.min_prefix_count),
        "use_rtg": bool(use_rtg),
        "rtg_gamma": float(args.rtg_gamma),
        "rtg_horizon": int(args.rtg_horizon),
        "session_col": str(args.session_col),
        "order_col": str(args.time_col if (use_rtg and args.time_col) else args.position_col),
    }

    print("[HRPO] saving:", args.out_path)
    with open(args.out_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    if str(getattr(args, "bucket_stats_out_path", "")).strip():
        stats_payload = {
            "reward_weights": reward_w,
            **ctx_mode_meta,
            "ctx_user_count_percentiles": {int(k): int(v) for k, v in ctx_user_count_percentiles.items()},
            "ctx_log_count_percentiles": {int(k): int(v) for k, v in ctx_log_count_percentiles.items()},
            "ctx_event_sum_percentiles": {int(k): float(v) for k, v in ctx_event_sum_percentiles.items()},
            "ctx_bucket_profile_percentiles": ctx_bucket_profile_percentiles,
        }
        out2 = str(args.bucket_stats_out_path)
        print("[HRPO] saving bucket stats:", out2)
        with open(out2, "wb") as f:
            pickle.dump(stats_payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("[HRPO] done.")


if __name__ == "__main__":
    main()
