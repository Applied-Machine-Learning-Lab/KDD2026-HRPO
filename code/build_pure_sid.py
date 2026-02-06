#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Build multi-layer Semantic IDs (SID) for KuaiRand videos with residual KMeans.

Example:
  python build_kuairand_sid.py \
    --video-basic dataset/kuairand/kuairand-Pure/data/video_features_basic_pure.csv \
    --video-stat dataset/kuairand/kuairand-Pure/data/video_features_statistic_pure.csv \
    --output-dir dataset/kuairand/kuairand-Pure/sid/ \
    --n-layers 4 --codebook-size 256 --max-tag 500
"""

import os
import json
import argparse
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder, StandardScaler, MultiLabelBinarizer
from sklearn.cluster import KMeans

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def ensure_dir(path: str):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def log(msg: str):
    print(f"[SID] {msg}")


def load_video_features(basic_path: str, stat_path: str | None = None) -> pd.DataFrame:
    """Load and merge basic/statistic video features on video_id."""
    log(f"Loading basic features from {basic_path}")
    df_basic = pd.read_csv(basic_path)

    if "tag" in df_basic.columns:
        df_basic["tag"] = df_basic["tag"].fillna("0")
    if "music_type" in df_basic.columns:
        df_basic["music_type"] = df_basic["music_type"].fillna(0)
    if "visible_status" in df_basic.columns:
        df_basic["visible_status"] = df_basic["visible_status"].fillna(0)

    df_basic = df_basic.fillna(0)

    if stat_path is not None and os.path.isfile(stat_path):
        log(f"Loading statistic features from {stat_path}")
        df_stat = pd.read_csv(stat_path).fillna(0)

        stat_cols = [c for c in df_stat.columns if c != "video_id"]
        df = df_basic.merge(df_stat[["video_id"] + stat_cols],
                            on="video_id", how="left")
        df = df.fillna(0)
        log(f"Merged basic + stat features, shape={df.shape}")
    else:
        log("No statistic feature file provided, only using basic features.")
        df = df_basic

    return df
def parse_tags(tag_str: str):
    if pd.isna(tag_str):
        return []
    s = str(tag_str)
    if not s or s == "0":
        return []
    out = []
    for t in s.split(","):
        t = t.strip()
        if not t or t == "0":
            continue
        try:
            out.append(int(t))
        except ValueError:
            continue
    return out

def build_feature_matrix(df: pd.DataFrame, max_tag: int = 500):
    """Build the final dense feature matrix used by SID clustering."""
    log("Building feature matrix...")
    video_ids = df["video_id"].values

    basic_num_cols = []
    for c in ["video_duration", "server_width", "server_height"]:
        if c in df.columns:
            basic_num_cols.append(c)

    stat_num_cols = [
        c for c in df.columns
        if c not in ["video_id", "author_id", "video_type", "upload_dt",
                     "upload_type", "visible_status", "music_id",
                     "music_type", "tag"]
        and df[c].dtype != "object"
    ]

    num_cols = basic_num_cols + stat_num_cols
    log(f"Numeric columns ({len(num_cols)}): {num_cols}")

    X_num = None
    if num_cols:
        num_values = df[num_cols].astype(float).values
        stat_idx = [num_cols.index(c) for c in stat_num_cols]
        num_values[:, stat_idx] = np.log1p(num_values[:, stat_idx])

        scaler = StandardScaler()
        X_num = scaler.fit_transform(num_values).astype("float32")
        log(f"Numeric feature matrix shape: {X_num.shape}")

    cat_cols = [c for c in
                ["video_type", "upload_type", "visible_status", "music_type"]
                if c in df.columns]

    X_cat = None
    if cat_cols:
        log(f"Categorical columns ({len(cat_cols)}): {cat_cols}")
        ohe = OneHotEncoder(handle_unknown="ignore")
        cat_values = df[cat_cols].astype(str).values
        X_cat = ohe.fit_transform(cat_values)
        if hasattr(X_cat, "toarray"):
            X_cat = X_cat.toarray()
        X_cat = X_cat.astype("float32")
        log(f"Categorical feature matrix shape: {X_cat.shape}")

    X_tag = None
    if "tag" in df.columns:
        log("Processing tag multi-labels...")
        tag_lists = [parse_tags(v) for v in df["tag"].values]

        from collections import Counter
        all_tags = []
        for ts in tag_lists:
            all_tags.extend(ts)
        counter = Counter(all_tags)
        most_common = counter.most_common(max_tag)
        allowed_tags = sorted([t for t, _ in most_common])
        allowed_set = set(allowed_tags)
        filtered = [[t for t in ts if t in allowed_set] for ts in tag_lists]

        mlb = MultiLabelBinarizer(classes=allowed_tags)
        X_tag = mlb.fit_transform(filtered).astype("float32")
        log(f"Tag feature matrix shape: {X_tag.shape}")

    feats = []
    for block in (X_num, X_cat, X_tag):
        if block is not None and block.size > 0:
            feats.append(block)

    if not feats:
        raise ValueError("No features found for SID.")

    X = np.concatenate(feats, axis=1)
    log(f"Final feature matrix shape: {X.shape}")
    return video_ids, X


def rq_kmeans(
    X: np.ndarray,
    n_layers: int = 4,
    codebook_size: int = 32,
    max_iter: int = 50,
    n_init: int = 10,
    random_state: int = 42,
):
    """
    Residual quantization KMeans.

    Args:
      X: [N, D] feature matrix.
    Returns:
      codes: [N, n_layers] cluster indices.
      codebooks: list of [K, D] codebooks.
      recon_error: final reconstruction MSE.
    """
    N, D = X.shape
    residual = X.copy().astype("float32")
    codes = np.zeros((N, n_layers), dtype=np.int32)
    codebooks = []

    for layer in range(n_layers):
        log(f"Training layer {layer+1}/{n_layers} with K={codebook_size}...")
        kmeans = KMeans(
            n_clusters=codebook_size,
            max_iter=max_iter,
            n_init=n_init,
            random_state=random_state + layer,
            verbose=0,
        )
        kmeans.fit(residual)
        centers = kmeans.cluster_centers_.astype("float32")
        labels = kmeans.labels_.astype(np.int32)

        codebooks.append(centers)
        codes[:, layer] = labels

        residual -= centers[labels]

        layer_err = np.mean(np.sum(residual ** 2, axis=1))
        log(f"  Layer {layer+1} mean squared residual error: {layer_err:.6f}")

    recon_error = np.mean(np.sum(residual ** 2, axis=1))
    log(f"Final reconstruction MSE after {n_layers} layers: {recon_error:.6f}")

    return codes, codebooks, recon_error

def main():
    parser = argparse.ArgumentParser(
        description="Build multi-layer Semantic IDs (RQ-KMeans) for KuaiRand videos."
    )
    parser.add_argument(
        "--video-basic",
        type=str,
        default=str(PROJECT_ROOT / "dataset/kuairand/kuairand-Pure/data/video_features_basic_pure.csv"),
        help="Path to video_features_basic_pure.csv",
    )
    parser.add_argument(
        "--video-stat",
        type=str,
        default=str(PROJECT_ROOT / "dataset/kuairand/kuairand-Pure/data/video_features_statistic_pure.csv"),
        help="Path to video_features_statistic_pure.csv (optional)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="dataset/kuairand/kuairand-Pure/sid/3_64",
        help="Directory to save SID mapping and codebooks.",
    )
    parser.add_argument(
        "--n-layers",
        type=int,
        default=3,
        help="Number of residual quantization layers.",
    )
    parser.add_argument(
        "--codebook-size",
        type=int,
        default=64,
        help="Number of clusters per layer (codebook size).",
    )
    parser.add_argument(
        "--max-tag",
        type=int,
        default=500,
        help="Keep top-N most frequent tags for multi-hot encoding.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=50,
        help="Max iterations for each KMeans.",
    )
    parser.add_argument(
        "--n-init",
        type=int,
        default=10,
        help="Number of random initializations for each KMeans.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for KMeans.",
    )
    parser.add_argument(
        "--log-session",
        type=str,
        default=str(PROJECT_ROOT / "dataset/kuairand/kuairand-Pure/data/log_session_4_08_to_5_08_Pure.csv"),
        help="Path to log_session.csv; if provided, only videos whose video_id appear in this file will be used to build SID.",
    )

    args = parser.parse_args()

    ensure_dir(args.output_dir)

    stat_path = args.video_stat if os.path.isfile(args.video_stat) else None
    df_video = load_video_features(args.video_basic, stat_path)

    if args.log_session is not None and os.path.isfile(args.log_session):
        log(f"Filtering videos using log_session file: {args.log_session}")
        df_log = pd.read_csv(args.log_session, usecols=["video_id"])

        df_log["video_id"] = df_log["video_id"].astype(df_video["video_id"].dtype)

        session_vids = set(df_log["video_id"].unique())
        all_vids = set(df_video["video_id"].unique())
        inter_vids = all_vids & session_vids
        missing_in_feat = len(session_vids - all_vids)

        log(f"Videos in feature table: {len(all_vids)}")
        log(f"Videos in log_session: {len(session_vids)}")
        log(f"Intersection size: {len(inter_vids)}")
        if missing_in_feat > 0:
            log(f"Warning: {missing_in_feat} videos appear in log_session but have no video features.")

        before = len(df_video)
        df_video = df_video[df_video["video_id"].isin(inter_vids)].reset_index(drop=True)
        log(f"Filtered videos by log_session: {before} -> {len(df_video)} rows kept.")
    else:
        log("No log_session file provided, using all videos in feature table.")

    video_ids, X = build_feature_matrix(df_video, max_tag=args.max_tag)

    codes, codebooks, recon_error = rq_kmeans(
        X,
        n_layers=args.n_layers,
        codebook_size=args.codebook_size,
        max_iter=args.max_iter,
        n_init=args.n_init,
        random_state=args.seed,
    )

    sid_cols = [f"sid_{i+1}" for i in range(args.n_layers)]
    df_sid = pd.DataFrame(codes, columns=sid_cols)
    df_sid.insert(0, "video_id", video_ids)

    mapping_path = os.path.join(args.output_dir, "video_sid_mapping.csv")
    df_sid.to_csv(mapping_path, index=False)
    log(f"Saved SID mapping to: {mapping_path}")

    for i, cb in enumerate(codebooks):
        cb_path = os.path.join(args.output_dir, f"codebook_layer{i+1}.npy")
        np.save(cb_path, cb)
        log(f"Saved codebook for layer {i+1} to: {cb_path}")

    config = {
        "n_layers": args.n_layers,
        "codebook_size": args.codebook_size,
        "max_tag": args.max_tag,
        "max_iter": args.max_iter,
        "n_init": args.n_init,
        "seed": args.seed,
        "reconstruction_mse": float(recon_error),
        "num_videos": int(len(video_ids)),
        "feature_dim": int(X.shape[1]),
    }
    config_path = os.path.join(args.output_dir, "sid_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    log(f"Saved config to: {config_path}")

    log("Done.")


if __name__ == "__main__":
    main()
