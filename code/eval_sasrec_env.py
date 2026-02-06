# -*- coding: utf-8 -*-
"""
eval_sasrec_env_v3.py

Goal: print/log outputs are FULLY aligned with eval_onerec_value_rerank.py
so you can compare baselines line-by-line.

Key alignment:
- Progress line:  Progress {finished}/{num_episodes} | Ret: ... | Len: ...
- Final summary block:
  ========================================
  Alpha: ...
  Total Reward: ...
  Depth: ...
  Avg Step Reward: ...
  Coverage: ...
  ILD: ...
  Table-style metrics:
  ...
  Behavior rates (count / impressions):
  ...
  ========================================

Note:
- SASRec does NOT use rerank_alpha / beam_width; we keep these args only for logging compatibility.
- Reward computation uses model.reward.get_immediate_reward(rw_dict) (repo API).
"""
from __future__ import annotations

import argparse
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from env.KREnvironment_WholeSession_GPU import KREnvironment_WholeSession_GPU
from model.reward import get_immediate_reward
from model.sasrec import SASRec, SASRecConfig
import utils  # keep same seed helper as onerec/hac eval scripts


def load_sasrec_model(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    cfg_dict = ckpt.get("cfg", None) if isinstance(ckpt, dict) else None

    if cfg_dict is None:
        item_w = state["item_emb.weight"]
        pos_w = state["pos_emb.weight"]
        n_items = int(item_w.shape[0] - 1)
        d_model = int(item_w.shape[1])
        max_len = int(pos_w.shape[0])
        cfg = SASRecConfig(n_items=n_items, max_len=max_len, d_model=d_model, n_heads=4, n_layers=2, dropout=0.2)
    else:
        cfg = SASRecConfig(**cfg_dict)

    model = SASRec(cfg).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, cfg


class SASRecPolicy:
    """
    Choose Top-K items by SASRec score from candidate pool.

    Action returned MUST be local indices into candidate pool (as required by KuaiSim env.step).
    """
    def __init__(self, model: SASRec, device: torch.device, slate_size: int, max_hist_len: int, args):
        self.model = model
        self.device = device
        self.slate_size = int(slate_size)
        self.max_hist_len = int(max_hist_len)
        self.args = args

        self._stat_users = 0
        self._stat_steps = 0
        self._stat_fallback_users = 0

    def _extract_hist(self, observation: Dict[str, Any]) -> torch.Tensor:
        hist = observation["user_history"]["history"]
        if isinstance(hist, np.ndarray):
            hist = torch.from_numpy(hist)
        return hist.to(self.device).long()

    def _prepare_seq_leftpad(self, hist: torch.Tensor) -> torch.Tensor:
        """
        Convert env history (any padding) -> SASRec input sequence:
          - ids in [1..n_items], PAD=0
          - LEFT-padded to length L (recent items at right end)
        """
        B, _H = hist.shape
        L = self.max_hist_len
        seq = torch.zeros((B, L), dtype=torch.long, device=self.device)
        for b in range(B):
            items = [int(x) for x in hist[b].tolist() if int(x) > 0]
            if len(items) > L:
                items = items[-L:]
            if len(items) > 0:
                seq[b, -len(items):] = torch.tensor(items, dtype=torch.long, device=self.device)
        return seq

    @torch.no_grad()
    def act(self, observation: Dict[str, Any], candidate_info: Dict[str, Any]) -> torch.Tensor:
        device = self.device
        hist = self._extract_hist(observation)  # (B,H)
        B = int(hist.size(0))

        cand = candidate_info["item_id"]
        if isinstance(cand, np.ndarray):
            cand = torch.from_numpy(cand)
        cand = cand.to(device).long().view(-1)  # (C,)
        C = int(cand.numel())

        seq = self._prepare_seq_leftpad(hist)  # (B,L)
        user_emb = self.model.encode(seq)      # (B,D)
        scores = self.model.score_candidates(user_emb, cand)  # (B,C)

        actions = torch.zeros((B, self.slate_size), dtype=torch.long, device=device)
        cand_list = cand.tolist()

        for b in range(B):
            self._stat_steps += 1
            used_pos = set()
            picked: List[int] = []

            if not getattr(self.args, "allow_repeat", False):
                hset = set([int(x) for x in hist[b].tolist() if int(x) > 0])
            else:
                hset = set()

            order = torch.argsort(scores[b], descending=True).tolist()
            for pos in order:
                gid = int(cand_list[pos])
                if gid <= 0:
                    continue
                if pos in used_pos:
                    continue
                if gid in hset:
                    continue
                used_pos.add(pos)
                picked.append(pos)
                if len(picked) >= self.slate_size:
                    break

            if len(picked) < self.slate_size:
                self._stat_fallback_users += 1
                for pos in range(C):
                    gid = int(cand_list[pos])
                    if gid <= 0:
                        continue
                    if pos in used_pos:
                        continue
                    if (not getattr(self.args, "allow_repeat", False)) and (gid in hset):
                        continue
                    used_pos.add(pos)
                    picked.append(pos)
                    if len(picked) >= self.slate_size:
                        break

            if len(picked) < self.slate_size:
                for pos in range(C):
                    if pos in used_pos:
                        continue
                    used_pos.add(pos)
                    picked.append(pos)
                    if len(picked) >= self.slate_size:
                        break

            actions[b] = torch.tensor(picked[: self.slate_size], dtype=torch.long, device=device)

        return actions

    def report_stats(self):
        if not getattr(self.args, "report_pick_stats", False):
            return
        users = max(1, int(self._stat_users))  # we don't track true users; keep aligned format
        steps = max(1, int(self._stat_steps))
        fb_rate = float(self._stat_fallback_users) / float(max(1, steps))
        print("=" * 40)
        print("[PickStats] steps=", steps, " users=", users)
        print(f"[PickStats] fallback_users={int(self._stat_fallback_users)}  fallback_rate={fb_rate:.4f}")
        print("[PickStats] no beam hits recorded (SASRec baseline)")
        print("=" * 40)


class RandomPolicy:
    """
    Random baseline: each step randomly sample slate_size candidates (avoid history if possible).
    Returns local indices.
    """
    def __init__(self, slate_size: int, device: torch.device, seed: int):
        self.slate_size = int(slate_size)
        self.device = device
        self.seed = int(seed)
        self._t = 0

    def act(self, observation, candidate_info):
        cand = candidate_info["item_id"]
        if torch.is_tensor(cand):
            cand = cand.detach().cpu().numpy()
        cand = np.asarray(cand)
        n_cand = int(len(cand))

        hist = observation["user_history"]["history"]
        if torch.is_tensor(hist):
            hist = hist.detach().cpu().numpy()
        hist = np.asarray(hist)

        B = int(hist.shape[0])
        all_idx = np.arange(n_cand, dtype=np.int64)
        actions = np.zeros((B, self.slate_size), dtype=np.int64)

        for i in range(B):
            self._t += 1
            rng = np.random.default_rng(self.seed + self._t + i)
            if getattr(observation, "allow_repeat", False):
                picked = rng.choice(all_idx, size=self.slate_size, replace=False if n_cand >= self.slate_size else True)
                actions[i] = picked
                continue

            hset = set(hist[i].tolist())
            mask = np.array([gid not in hset for gid in cand], dtype=bool)
            remain = all_idx[mask]
            pool = remain if remain.size >= self.slate_size else all_idx
            replace = pool.size < self.slate_size
            actions[i] = rng.choice(pool, size=self.slate_size, replace=replace)

        return torch.tensor(actions, dtype=torch.long, device=self.device)


def parse_args():
    parser = argparse.ArgumentParser()
    parser = KREnvironment_WholeSession_GPU.parse_model_args(parser)

    parser.add_argument("--sasrec_ckpt", type=str, required=True, help="SASRec checkpoint (best.pt/last.pt)")

    parser.add_argument("--num_episodes", type=int, default=200)
    parser.add_argument("--max_steps_per_episode", type=int, default=20, help=" HAC align")
    parser.add_argument("--eval_log_path", type=str, default="../output/KuaiRand_Pure/eval/sasrec.log")
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda:0")

    parser.add_argument("--beam_width", type=int, default=0, help="(unused for SASRec) printed for log alignment")
    parser.add_argument("--rerank_alpha", type=float, default=0.0, help="(unused for SASRec) printed as Alpha for alignment")
    parser.add_argument("--rerank_formula", type=str, default="mul", choices=["mul", "add"], help="(unused)")

    parser.add_argument("--seed", type=int, default=2025, help="random seed for reproducible eval")
    parser.add_argument("--max_hist_len", type=int, default=50)
    parser.add_argument("--report_pick_stats", action="store_true")
    parser.add_argument("--allow_repeat", action="store_true")
    parser.add_argument("--random_policy", action="store_true", help="enable random baseline (ignore SASRec)")

    return parser.parse_args()


def run_eval(args):
    device = torch.device(args.device)
    print(f"[Info] Using device: {device}")

    env = KREnvironment_WholeSession_GPU(args)
    if hasattr(env, "set_seed"):
        env.set_seed(args.seed)
        print(f"[Seed] env.set_seed({args.seed}) called.")

    model, cfg = load_sasrec_model(args.sasrec_ckpt, device)

    policy = SASRecPolicy(model=model, device=device, slate_size=args.slate_size, max_hist_len=args.max_hist_len, args=args)
    rand_policy = RandomPolicy(args.slate_size, device, seed=args.seed)

    if args.random_policy:
        print('[Eval] RandomPolicy ENABLED: ignore SASRec, sample random items each step.')

    print(
        f"[Eval] Rerank Alpha={args.rerank_alpha}, "
        f"Beam Width={args.beam_width}, Seed={args.seed}"
    )

    observation = env.reset({"batch_size": args.episode_batch_size})

    cur_returns = torch.zeros(args.episode_batch_size, device=device)
    cur_lengths = torch.zeros(args.episode_batch_size, device=device)
    finished = 0
    all_ret, all_len = [], []

    beh_names = None
    if hasattr(env, "response_types"):
        try:
            beh_names = list(env.response_types)
        except Exception:
            beh_names = None
    if beh_names is None:
        beh_names = ["is_click", "long_view", "is_like", "is_comment", "is_forward", "is_follow", "is_hate"]

    K = len(beh_names)
    B = int(args.episode_batch_size)
    cur_beh_counts = torch.zeros(B, K, device=device)
    cur_impr = torch.zeros(B, device=device)
    total_beh_counts = torch.zeros(K, device=device)
    total_impr = 0.0

    response_weights = env.response_weights

    try:
        while finished < args.num_episodes:
            cand = env.get_candidate_info(feed_dict=None) if hasattr(env, "get_candidate_info") else env.get_candidate_info()
            action = (rand_policy.act(observation, cand) if args.random_policy else policy.act(observation, cand))

            next_obs, resp, _ = env.step({"action": action})

            im = resp.get("immediate_response", None)
            if im is None:
                raise RuntimeError("resp no 'immediate_response' , ")
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
                "immediate_response": resp["immediate_response"],
                "immediate_response_weight": response_weights,
            }
            step_r = get_immediate_reward(rw_dict).to(device)
            cur_returns += step_r
            cur_lengths += 1

            done = resp["done"]
            if isinstance(done, np.ndarray):
                done = torch.from_numpy(done)
            done = done.to(device).bool()

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
                                f"Ret: {np.mean(all_ret):.4f} | "
                                f"Len: {np.mean(all_len):.2f}"
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
        print(f"Alpha: {args.rerank_alpha}")

        total_ret = float(np.mean(all_ret))
        depth = float(np.mean(all_len))
        avg_step_r = total_ret / depth if depth > 0 else 0.0

        print(f"Total Reward: {total_ret:.4f}")
        print(f"Depth: {depth:.2f}")
        print(f"Avg Step Reward: {avg_step_r:.4f}")

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

        if not args.random_policy and hasattr(policy, "report_stats"):
            policy.report_stats()

        print("=" * 40)


if __name__ == "__main__":
    args = parse_args()
    utils.set_random_seed(args.seed)
    run_eval(args)
