# -*- coding: utf-8 -*-
"""
eval_gru4rec_env_v1.py

Goal: print/log outputs are aligned with eval_onerec_value_rerank.py
so you can compare baselines line-by-line.

This file is adapted from eval_sasrec_env.py (same env loop, same metrics, same logging style).

Notes:
- GRU4Rec does NOT use rerank_alpha / beam_width; we keep these args only for logging compatibility.
- Reward computation uses get_immediate_reward(rw_dict) (repo API).
"""
from __future__ import annotations

import argparse
from typing import Any, Dict, List

import numpy as np
import torch

from env.KREnvironment_WholeSession_GPU import KREnvironment_WholeSession_GPU
from model.reward import get_immediate_reward
from model.gru4rec import GRU4Rec, GRU4RecConfig
import utils  # keep same seed helper as onerec/hac eval scripts


def load_gru4rec_model(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    cfg_dict = ckpt.get("cfg", None) if isinstance(ckpt, dict) else None

    if cfg_dict is None:
        item_w = state["item_emb.weight"]
        n_items = int(item_w.shape[0] - 1)
        d_model = int(item_w.shape[1])

        ks = []
        for k in range(0, 32):
            if f"gru.weight_ih_l{k}" in state:
                ks.append(k)
        n_layers = (max(ks) + 1) if ks else 1

        tie_weights = ("out.weight" not in state) and ("out.bias" not in state)

        cfg = GRU4RecConfig(
            n_items=n_items,
            max_len=50,
            d_model=d_model,
            n_layers=n_layers,
            dropout=0.2,
            tie_weights=tie_weights,
            final_act="elu",
        )
    else:
        cfg = GRU4RecConfig(**cfg_dict)

    model = GRU4Rec(cfg).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, cfg


class GRU4RecPolicy:
    """
    Choose Top-K items by GRU4Rec score from candidate pool.

    Action returned MUST be local indices into candidate pool (as required by KuaiSim env.step).
    """
    def __init__(self, model: GRU4Rec, device: torch.device, slate_size: int, max_hist_len: int, args):
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
        Convert env history (any padding) -> model input sequence:
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

            picked_t = torch.tensor(picked[: self.slate_size], dtype=torch.long, device=device)
            am = getattr(self.args, 'action_mode', 'index')
            if am == 'id':
                picked_ids = cand[picked_t] if cand.dim() == 1 else cand[b, picked_t]
                actions[b] = picked_ids.to(device)
            elif am == 'index1':
                actions[b] = (picked_t + 1)
            else:
                actions[b] = picked_t

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
        print("[PickStats] no beam hits recorded (GRU4Rec baseline)")
        print("=" * 40)



class RandomPolicy:
    """
    Random baseline: each step randomly sample slate_size candidates (avoid history if possible).

    Action can be encoded as:
    - index : 0-based local indices into candidate pool
    - index1: 1-based local indices into candidate pool (diagnose)
    - id    : global item_id (diagnose)
    """
    def __init__(self, slate_size: int, device: torch.device, seed: int, allow_repeat: bool = False, action_mode: str = "index"):
        self.slate_size = int(slate_size)
        self.device = device
        self.seed = int(seed)
        self.allow_repeat = bool(allow_repeat)
        self.action_mode = str(action_mode)
        self._t = 0

    def act(self, observation, candidate_info):
        cand = candidate_info["item_id"]
        if torch.is_tensor(cand):
            cand = cand.detach().cpu().numpy()
        cand = np.asarray(cand)

        hist = observation["user_history"]["history"]
        if torch.is_tensor(hist):
            hist = hist.detach().cpu().numpy()
        hist = np.asarray(hist)
        B = int(hist.shape[0])

        if cand.ndim == 1:
            cand_2d = np.broadcast_to(cand[None, :], (B, cand.shape[0]))
        else:
            cand_2d = cand
        C = int(cand_2d.shape[1])

        all_idx = np.arange(C, dtype=np.int64)
        actions = np.zeros((B, self.slate_size), dtype=np.int64)

        for i in range(B):
            self._t += 1
            rng = np.random.default_rng(self.seed + self._t + i)

            if self.allow_repeat:
                pool = all_idx
            else:
                hset = set([int(x) for x in hist[i].tolist() if int(x) > 0])
                mask = np.array([int(gid) not in hset for gid in cand_2d[i].tolist()], dtype=bool)
                remain = all_idx[mask]
                pool = remain if remain.size > 0 else all_idx

            replace = pool.size < self.slate_size
            picked_pos = rng.choice(pool, size=self.slate_size, replace=replace)

            if self.action_mode == "id":
                picked_ids = [int(cand_2d[i, p]) for p in picked_pos]
                actions[i] = np.array(picked_ids, dtype=np.int64)
            elif self.action_mode == "index1":
                actions[i] = picked_pos + 1
            else:
                actions[i] = picked_pos

        return torch.tensor(actions, dtype=torch.long, device=self.device)


def parse_args():
    parser = argparse.ArgumentParser()
    parser = KREnvironment_WholeSession_GPU.parse_model_args(parser)

    parser.add_argument("--gru4rec_ckpt", type=str, required=True, help="GRU4Rec checkpoint (best.pt/last.pt)")

    parser.add_argument("--num_episodes", type=int, default=200)
    parser.add_argument("--max_steps_per_episode", type=int, default=20, help=" HAC align")
    parser.add_argument("--eval_log_path", type=str, default="../output/KuaiRand_Pure/eval/gru4rec.log")
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda:0")

    parser.add_argument("--beam_width", type=int, default=0, help="(unused for GRU4Rec) printed for log alignment")
    parser.add_argument("--rerank_alpha", type=float, default=0.0, help="(unused for GRU4Rec) printed as Alpha for alignment")
    parser.add_argument("--rerank_formula", type=str, default="mul", choices=["mul", "add"], help="(unused)")

    parser.add_argument("--seed", type=int, default=2025, help="random seed for reproducible eval")
    parser.add_argument("--max_hist_len", type=int, default=50)
    parser.add_argument("--report_pick_stats", action="store_true")
    parser.add_argument("--allow_repeat", action="store_true")
    parser.add_argument("--random_policy", action="store_true", help="enable random baseline (ignore GRU4Rec)")

    parser.add_argument(
        "--action_mode", type=str, default="index",
        choices=["index", "index1", "id"],
        help="How to encode actions for env.step: "
             "index=0-based local indices into candidate pool (repo default); "
             "index1=1-based local indices (diagnose off-by-one); "
             "id=global item_id (diagnose env expecting ids)."
    )
    parser.add_argument(
        "--hist_mode", type=str, default="env",
        choices=["env", "click"],
        help="History fed into GRU4Rec: env=use env observation['user_history']['history']; "
             "click=maintain click-only history online (recommended if training used click-only)."
    )
    parser.add_argument(
        "--hist_reverse", action="store_true",
        help="Reverse the order of extracted history items before feeding into model (diagnose history order mismatch)."
    )

    return parser.parse_args()


def run_eval(args):
    device = torch.device(args.device)
    print(f"[Info] Using device: {device}")

    env = KREnvironment_WholeSession_GPU(args)
    if hasattr(env, "set_seed"):
        env.set_seed(args.seed)
        print(f"[Seed] env.set_seed({args.seed}) called.")

    model, _cfg = load_gru4rec_model(args.gru4rec_ckpt, device)

    policy = GRU4RecPolicy(model=model, device=device, slate_size=args.slate_size, max_hist_len=args.max_hist_len, args=args)
    rand_policy = RandomPolicy(args.slate_size, device, seed=args.seed, allow_repeat=args.allow_repeat, action_mode=args.action_mode)

    if args.random_policy:
        print("[Eval] RandomPolicy ENABLED: ignore GRU4Rec, sample random items each step.")

    print(
        f"[Eval] Rerank Alpha={args.rerank_alpha}, "
        f"Beam Width={args.beam_width}, Seed={args.seed}"
    )

    observation = env.reset({"batch_size": args.episode_batch_size})

    def _extract_items_from_hist(hist_row):
        items = [int(x) for x in hist_row.tolist() if int(x) > 0]
        if getattr(args, 'hist_reverse', False):
            items = list(reversed(items))
        if len(items) > args.max_hist_len:
            items = items[-args.max_hist_len:]
        return items

    click_hist = torch.zeros((args.episode_batch_size, args.max_hist_len), dtype=torch.long, device=device)
    try:
        h0 = observation['user_history']['history']
        if isinstance(h0, np.ndarray):
            h0 = torch.from_numpy(h0)
        h0 = h0.to(device).long()
        for b in range(h0.size(0)):
            items = _extract_items_from_hist(h0[b])
            if len(items) > 0:
                click_hist[b, -len(items):] = torch.tensor(items, dtype=torch.long, device=device)
    except Exception:
        pass


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
            obs_for_policy = observation
            if getattr(args, 'hist_mode', 'env') == 'click':
                obs_for_policy = dict(observation)
                uh = dict(observation.get('user_history', {}))
                uh['history'] = click_hist.detach().cpu().numpy() if isinstance(observation['user_history']['history'], np.ndarray) else click_hist
                obs_for_policy['user_history'] = uh
            action = (rand_policy.act(obs_for_policy, cand) if args.random_policy else policy.act(obs_for_policy, cand))

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


            if getattr(args, 'hist_mode', 'env') == 'click':
                try:
                    click_idx = beh_names.index('is_click') if ('beh_names' in locals() and 'is_click' in beh_names) else 0
                    clicks = im[:, :, click_idx]  # (B, slate)
                    cand_ids = cand['item_id']
                    if isinstance(cand_ids, np.ndarray):
                        cand_ids = torch.from_numpy(cand_ids)
                    cand_ids = cand_ids.to(device).long()
                    am = getattr(args, 'action_mode', 'index')
                    act = action
                    if isinstance(act, np.ndarray):
                        act = torch.from_numpy(act)
                    act = act.to(device).long()
                    if am == 'id':
                        rec_ids = act
                    elif am == 'index1':
                        rec_ids = cand_ids[act.clamp(min=1, max=cand_ids.numel()) - 1]
                    else:
                        rec_ids = cand_ids[act.clamp(min=0, max=cand_ids.numel()-1)]
                    B_now = int(click_hist.size(0))
                    for b in range(B_now):
                        for j in range(int(rec_ids.size(1))):
                            if float(clicks[b, j].item()) > 0.5:
                                gid = int(rec_ids[b, j].item())
                                if gid <= 0:
                                    continue
                                click_hist[b, :-1] = click_hist[b, 1:]
                                click_hist[b, -1] = gid
                except Exception:
                    pass

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

                if getattr(args, 'hist_mode', 'env') == 'click':
                    try:
                        h_new = next_obs['user_history']['history']
                        if isinstance(h_new, np.ndarray):
                            h_new = torch.from_numpy(h_new)
                        h_new = h_new.to(device).long()
                        for idx in idxs.tolist():
                            click_hist[idx].zero_()
                            items = _extract_items_from_hist(h_new[idx])
                            if len(items) > 0:
                                click_hist[idx, -len(items):] = torch.tensor(items, dtype=torch.long, device=device)
                    except Exception:
                        pass

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
