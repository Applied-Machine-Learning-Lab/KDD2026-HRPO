# -*- coding: utf-8 -*-
"""
eval_tiger_sid_env.py

Evaluate a TIGER-SID policy in KuaiSim.

Pipeline:
1) Encode user history items into SID token sequences.
2) Decode candidate SID sequences with TIGER beam search.
3) Map decoded SID sequences back to item IDs.
4) Step the simulator and report reward/behavior metrics.
"""

import os
import argparse
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import T5Config, T5ForConditionalGeneration

from env.KREnvironment_WholeSession_GPU import KREnvironment_WholeSession_GPU
from model.reward import get_immediate_reward
import utils  #  HAC / OneRec random seed

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TIGER(nn.Module):
    """
    T5-based generative recommender (Tiger) for SID tokens.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        t5cfg = T5Config(
            num_layers=config["num_layers"],
            num_decoder_layers=config["num_decoder_layers"],
            d_model=config["d_model"],
            d_ff=config["d_ff"],
            num_heads=config["num_heads"],
            d_kv=config["d_kv"],
            dropout_rate=config["dropout_rate"],
            vocab_size=config["vocab_size"],
            pad_token_id=config["pad_token_id"],
            eos_token_id=config["eos_token_id"],
            decoder_start_token_id=config["pad_token_id"],
            feed_forward_proj=config["feed_forward_proj"],
        )
        self.model = T5ForConditionalGeneration(t5cfg)

    @property
    def n_parameters(self) -> str:
        num_params = lambda ps: sum(p.numel() for p in ps if p.requires_grad)
        total_params = num_params(self.parameters())
        emb_params = num_params(self.model.get_input_embeddings().parameters())
        return (
            f"#Embedding parameters: {emb_params}\n"
            f"#Non-embedding parameters: {total_params - emb_params}\n"
            f"#Total trainable parameters: {total_params}\n"
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
    ):
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        return out.loss, out.logits

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        **kwargs,
    ):
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )


def load_tiger_model(args, device: torch.device):
    """
    Load TIGER from checkpoint and infer SID vocabulary from mapping.
    """
    print(f"[Model] Loading TIGER checkpoint: {args.tiger_ckpt}")
    state_dict = torch.load(args.tiger_ckpt, map_location=device)

    df_sid = pd.read_csv(args.sid_mapping_path)
    sid_cols = [c for c in df_sid.columns if c.startswith("sid_")]
    if not sid_cols:
        raise ValueError(f"No sid_* columns found in {args.sid_mapping_path}")
    sid_depth = len(sid_cols)
    codes_raw = df_sid[sid_cols].values.astype(int)
    codebook_size = int(codes_raw.max() + 1)
    vocab_size = codebook_size + 1   # 0 for PAD/EOS, 1..codebook_size  SID token

    config = {
        "num_layers": args.num_layers,
        "num_decoder_layers": args.num_decoder_layers,
        "d_model": args.d_model,
        "d_ff": args.d_ff,
        "num_heads": args.num_heads,
        "d_kv": args.d_kv,
        "dropout_rate": args.dropout_rate,
        "feed_forward_proj": args.feed_forward_proj,
        "vocab_size": vocab_size,
        "pad_token_id": 0,
        "eos_token_id": 0,
    }

    model = TIGER(config).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    num_params = sum(p.numel() for p in model.parameters())
    print(f"[Model] Total parameters: {num_params:,d}")
    print(model.n_parameters)
    print(f"[Model] Loaded TIGER-SID with vocab_size={vocab_size}, sid_depth={sid_depth}")
    return model, sid_depth, codebook_size


def build_iid2sid_tokens(reader, mapping_path: str, sid_depth: int, device: torch.device):
    """
    Build item_id -> SID-token lookup for Tiger decoding.
    Raw SID code uses 0..K-1, while model token uses SID+1 and 0 is PAD.
    """
    df = pd.read_csv(mapping_path)
    sid_cols = [f"sid_{i+1}" for i in range(sid_depth)]
    if not all(c in df.columns for c in sid_cols):
        sid_cols = [f"sid_{i}" for i in range(sid_depth)]
    if not all(c in df.columns for c in sid_cols):
        raise ValueError(f"SID columns not found in mapping: {sid_cols}")

    df = df[["video_id"] + sid_cols].drop_duplicates("video_id").set_index("video_id")

    n_items = len(reader.items)
    iid2sid_raw = np.zeros((n_items + 1, sid_depth), dtype="int64")
    sid2iid_map_tok = {}

    for vid, iid in reader.item_id_vocab.items():
        if vid in df.index:
            vals = df.loc[vid, sid_cols].astype(int).values  # raw SID in [0, K-1]
            iid2sid_raw[iid] = vals

    iid2sid_tok = np.zeros_like(iid2sid_raw, dtype="int64")
    iid2sid_tok[1:] = iid2sid_raw[1:] + 1

    for iid in range(1, n_items + 1):
        codes_tok = tuple(iid2sid_tok[iid].tolist())
        if any(c > 0 for c in codes_tok):
            sid2iid_map_tok[codes_tok] = iid

    iid2sid_tok_t = torch.from_numpy(iid2sid_tok).to(device)
    return iid2sid_tok_t, sid2iid_map_tok


class TigerSIDPolicy:
    def __init__(self, model: TIGER, iid2sid_tok: torch.Tensor, sid2iid_map_tok: Dict[Tuple[int, ...], int],
                 sid_depth: int, device: torch.device, slate_size: int, reader, args):
        self.model = model
        self.iid2sid_tok = iid2sid_tok   # [n_items+1, sid_depth], token = raw+1, 0=PAD
        self.sid2iid_map = sid2iid_map_tok
        self.sid_depth = sid_depth
        self.device = device
        self.slate_size = slate_size
        self.reader = reader

        self.beam_width = args.beam_width
        self.max_hist_items = args.max_hist_items

    def _build_history_tokens(self, hist_iids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert item history [B, H] into flattened Tiger SID-token inputs.
        Returns input_ids and attention_mask.
        """
        B, H_env = hist_iids.shape
        device = hist_iids.device
        max_hist_items = self.max_hist_items
        sid_depth = self.sid_depth

        hist_iids = hist_iids.clamp(min=0, max=self.iid2sid_tok.size(0) - 1)

        H = min(H_env, max_hist_items)
        if H_env > H:
            hist_inp = hist_iids[:, -H:]  # [B, H]
        else:
            pad_items = torch.zeros(B, max_hist_items - H_env, dtype=torch.long, device=device)
            hist_inp = torch.cat([pad_items, hist_iids], dim=1)  # [B, max_hist_items]
            H = max_hist_items

        hist_sids = self.iid2sid_tok[hist_inp]  # [B, H, D]
        B, H, D = hist_sids.shape
        assert D == sid_depth

        history = hist_sids.reshape(B, H * D)
        attention_mask = (history != 0).long()

        return history, attention_mask

    @torch.no_grad()
    def act(self, observation, candidate_info):
        """
        Inputs:
          observation["user_history"]["history"]: [B, H_env] item ids
          candidate_info["item_id"]: [N_cand] candidate item ids
        Output:
          [B, slate_size] local candidate indices for env.step.
        """
        device = self.device
        hist_iids = observation["user_history"]["history"].long().to(device)   # [B, H_env]
        cand_iids = candidate_info["item_id"].long().cpu().numpy()            # [N_cand]
        global_to_local = {gid: idx for idx, gid in enumerate(cand_iids)}

        B = hist_iids.size(0)

        input_ids, attention_mask = self._build_history_tokens(hist_iids)  # [B, L_in], [B, L_in]

        gen = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_beams=self.beam_width,
            num_return_sequences=self.beam_width,
            max_length=self.sid_depth + 1,  # +1 for decoder_start_token
            early_stopping=True,
            do_sample=False,
        )
        gen = gen[:, 1:1 + self.sid_depth]
        gen = gen.view(B, self.beam_width, self.sid_depth)  # [B, W, D]

        gen_np = gen.cpu().numpy()
        actions = []

        for i in range(B):
            user_hist_items = set(hist_iids[i].cpu().numpy().tolist())
            beams = gen_np[i]  # [W, D]

            item_score_pairs = []
            for w in range(self.beam_width):
                sid_seq = tuple(int(x) for x in beams[w].tolist())
                if sid_seq in self.sid2iid_map:
                    gid = self.sid2iid_map[sid_seq]   # environment item_id (iid)
                    if gid in global_to_local and gid not in user_hist_items:
                        local_idx = global_to_local[gid]
                        score = self.beam_width - w
                        item_score_pairs.append((local_idx, score))

            item_score_pairs.sort(key=lambda x: x[1], reverse=True)
            candidate_idxs = [x[0] for x in item_score_pairs]
            candidate_idxs = list(dict.fromkeys(candidate_idxs))  # dedupe

            if len(candidate_idxs) < self.slate_size:
                miss = self.slate_size - len(candidate_idxs)
                all_idx = np.arange(len(cand_iids))
                remain = np.setdiff1d(all_idx, candidate_idxs)
                if len(remain) > 0:
                    if len(remain) >= miss:
                        fill = np.random.choice(remain, miss, replace=False).tolist()
                    else:
                        fill = remain.tolist()
                    candidate_idxs.extend(fill)

            actions.append(candidate_idxs[:self.slate_size])

        return torch.tensor(actions, dtype=torch.long, device=device)


def parse_args():
    parser = argparse.ArgumentParser()
    parser = KREnvironment_WholeSession_GPU.parse_model_args(parser)

    parser.add_argument("--tiger_ckpt", type=str, required=True, help="TIGER-SID trained checkpoint")
    parser.add_argument("--sid_mapping_path", type=str,
                        default=str(PROJECT_ROOT / "code/dataset/kuairand/kuairand-Pure/sid/32_mask/video_sid_mapping.csv"),
                        help="Path to SID mapping CSV (sid_1, sid_2, ...).")

    parser.add_argument("--num_episodes", type=int, default=200)
    parser.add_argument("--max_steps_per_episode", type=int, default=20, help="Max steps per episode.")
    parser.add_argument("--eval_log_path", type=str,
                        default="../output/KuaiRand_Pure/eval/tiger_sid_eval.log")
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda:0")

    parser.add_argument("--model_size", type=str, default="medium",
                        choices=["mini", "medium", "large"],
                        help="Tiger model size preset.")
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_decoder_layers", type=int, default=4)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--d_ff", type=int, default=1024)
    parser.add_argument("--num_heads", type=int, default=6)
    parser.add_argument("--d_kv", type=int, default=64)
    parser.add_argument("--dropout_rate", type=float, default=0.1)
    parser.add_argument("--feed_forward_proj", type=str, default="relu")

    parser.add_argument("--max_hist_items", type=int, default=50,
                        help="Max history items fed into Tiger.")
    parser.add_argument("--beam_width", type=int, default=50,
                        help="Beam width for SID decoding.")

    parser.add_argument("--seed", type=int, default=2025,
                        help="Random seed for reproducible evaluation.")

    args = parser.parse_args()
    if args.model_size == "mini":
        args.num_layers = 3
        args.num_decoder_layers = 3
        args.d_model = 128
        args.d_ff = 512
        args.num_heads = 4
        args.d_kv = 16
    elif args.model_size == "medium":
        args.num_layers = 4
        args.num_decoder_layers = 4
        args.d_model = 128
        args.d_ff = 1024
        args.num_heads = 6
        args.d_kv = 64
    elif args.model_size == "large":
        args.num_layers = 6
        args.num_decoder_layers = 6
        args.d_model = 192
        args.d_ff = 1536
        args.num_heads = 8
        args.d_kv = 24

    return args


def run_eval(args):
    device = torch.device(args.device)
    print(f"[Info] Using device: {device}")

    env = KREnvironment_WholeSession_GPU(args)
    if hasattr(env, "set_seed"):
        env.set_seed(args.seed)
        print(f"[Seed] env.set_seed({args.seed}) called.")

    model, sid_depth, codebook_size = load_tiger_model(args, device)
    iid2sid_tok, sid2iid_map = build_iid2sid_tokens(env.reader, args.sid_mapping_path, sid_depth, device)

    policy = TigerSIDPolicy(
        model=model,
        iid2sid_tok=iid2sid_tok,
        sid2iid_map_tok=sid2iid_map,
        sid_depth=sid_depth,
        device=device,
        slate_size=args.slate_size,
        reader=env.reader,
        args=args,
    )

    print(f"[Eval] Tiger-SID baseline | Beam Width={args.beam_width} | Slate Size={args.slate_size}")
    print(f"[Eval] SID depth={sid_depth}, codebook_size={codebook_size}")

    observation = env.reset({"batch_size": args.episode_batch_size})
    
    B = int(args.episode_batch_size)
    cur_returns = torch.zeros(B, device=device)
    cur_lengths = torch.zeros(B, device=device)
    finished = 0
    all_ret, all_len = [], []
    
    beh_names = None
    if hasattr(env, "response_types"):
        try:
            beh_names = list(env.response_types)
        except Exception:
            beh_names = None
    if beh_names is None:
        try:
            beh_names = env.reader.get_statistics().get("feedback_type", None)
        except Exception:
            beh_names = None
    if beh_names is None and hasattr(env, "response_weights"):
        try:
            K_tmp = int(env.response_weights.shape[0])
            beh_names = [f"fb{i}" for i in range(K_tmp)]
        except Exception:
            beh_names = None
    if beh_names is None:
        beh_names = ["is_click", "long_view", "is_like", "is_comment", "is_forward", "is_follow", "is_hate"]
    
    K = len(beh_names)
    cur_beh_counts = torch.zeros(B, K, device=device)   # (B, K)
    cur_impr = torch.zeros(B, device=device)            # (B,)
    total_beh_counts = torch.zeros(K, device=device)    # (K,)
    total_impr = 0.0

    response_weights = env.response_weights

    try:
        while finished < args.num_episodes:
            try:
                cand = env.get_candidate_info(feed_dict=None)
            except Exception:
                cand = env.get_candidate_info()

            action = policy.act(observation, cand)

            next_obs, resp, _ = env.step({'action': action})

            im = resp.get("immediate_response", None)
            if im is None:
                raise RuntimeError("Missing immediate_response in environment output.")
            if isinstance(im, np.ndarray):
                im = torch.from_numpy(im)
            im = im.to(device).float()  # (B, slate, K)

            cur_impr += float(im.size(1))
            K_eff = min(K, int(im.size(2)))
            if K_eff < K:
                cur_beh_counts[:, :K_eff] += im[:, :, :K_eff].sum(dim=1)
            else:
                cur_beh_counts += im.sum(dim=1)


            rw_dict = {
                'immediate_response': resp['immediate_response'],
                'immediate_response_weight': response_weights,
            }
            step_r = get_immediate_reward(rw_dict)

            cur_returns += step_r
            cur_lengths += 1

            done = resp['done']
            if done.any():
                idxs = torch.nonzero(done).squeeze(-1)
                for idx in idxs:
                    if finished < args.num_episodes:
                        all_ret.append(cur_returns[idx].item())
                        all_len.append(cur_lengths[idx].item())
                        finished += 1

                        total_beh_counts += cur_beh_counts[idx]
                        total_impr += float(cur_impr[idx].item())
                        if finished % args.log_every == 0:
                            print(
                                f"Progress {finished}/{args.num_episodes} | "
                                f"Ret: {np.mean(all_ret):.4f} | Len: {np.mean(all_len):.2f}"
                            )
                cur_returns[done] = 0
                cur_lengths[done] = 0

                cur_beh_counts[done] = 0
                cur_impr[done] = 0
            observation = next_obs

    except KeyboardInterrupt:
        print("[Eval] Interrupted by user.")

    if all_ret:
        print("=" * 40)
        total_ret = float(np.mean(all_ret))
        depth = float(np.mean(all_len))
        avg_step_r = total_ret / depth if depth > 0 else 0.0

        print(f"Total Reward: {total_ret:.4f}")
        print(f"Depth: {depth:.2f}")
        print(f"Avg Step Reward: {avg_step_r:.4f}")

        coverage = 0.0
        ild = 0.0
        if hasattr(env, "get_report"):
            env_report = env.get_report()
            coverage = float(env_report.get("coverage", 0.0))
            ild = float(env_report.get("ILD", 0.0))

        print(f"Coverage: {coverage:.2f}")
        print(f"ILD: {ild:.4f}")

        print("Table-style metrics:")
        print(f"Depth: {depth:.2f}")
        print(f"Average reward: {avg_step_r:.4f}")
        print(f"Total reward: {total_ret:.4f}")
        print(f"Coverage: {coverage:.2f}")
        print(f"ILD: {ild:.4f}")

        print("Behavior rates (count / impressions):")
        if total_impr <= 0:
            print("  [WARN] total_impr=0")
        else:
            for k, name in enumerate(beh_names):
                cnt = float(total_beh_counts[k].item())
                rate = 100.0 * cnt / total_impr
                print(f"  {name}: {int(round(cnt))}/{int(round(total_impr))} ({rate:.4f}%)")

        print("=" * 40)


if __name__ == "__main__":
    args = parse_args()
    utils.set_random_seed(args.seed)
    run_eval(args)
