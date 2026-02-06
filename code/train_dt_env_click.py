# -*- coding: utf-8 -*-
"""
Train a DecisionTransformerRec policy online in KuaiSim.

This script:
- collects trajectories from the environment,
- stores click-token histories and rewards,
- builds RTG-conditioned training batches,
- optimizes next-item likelihood under DT.
"""

import os
import argparse
import time
import random
from pathlib import Path
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from env.KREnvironment_WholeSession_GPU import KREnvironment_WholeSession_GPU
from model.DecisionTransformerRec import DecisionTransformerRec
from model.reward import get_immediate_reward
import utils

PROJECT_ROOT = Path(__file__).resolve().parents[1]



class Trajectory:
    """One user trajectory with clicked item tokens and rewards."""
    def __init__(self, user_id, user_feats):
        self.user_id = int(user_id)
        self.user_feats = {k: np.array(v, copy=True) for k, v in user_feats.items()}
        self.items = []        # click item_id (;  click =  token)
        self.rewards = []      # click reward ()
        self.total_return = 0.0  #  episode (click reward )

    def add_step(self, item_id, reward):
        """Append one click token and its reward."""
        self.items.append(int(item_id))
        self.rewards.append(float(reward))
        self.total_return += float(reward)

    def length(self):
        return len(self.items)


class ReplayBuffer:
    """Episode replay buffer for DT training."""
    def __init__(self, max_episodes=10000):
        self.max_episodes = max_episodes
        self.episodes = []

    def __len__(self):
        return len(self.episodes)

    def add(self, traj: Trajectory):
        if traj.length() == 0:
            return
        self.episodes.append(traj)
        if len(self.episodes) > self.max_episodes:
            self.episodes.pop(0)

    def sample_episodes(self, batch_size):
        if len(self.episodes) == 0:
            return []
        batch_size = min(batch_size, len(self.episodes))
        return random.sample(self.episodes, batch_size)



class DTPolicy:
    def __init__(self, model, env, args):
        self.model = model
        self.env = env
        self.device = args.device
        self.slate_size = args.slate_size
        self.rtg_scale = args.rtg_scale
        self.max_timestep = args.max_timestep
        self.target_return = args.target_return
        self.eps_greedy = args.eps_greedy

        self.temperature = getattr(args, 'temperature', 0.0)
        self.user_feat_names = list(self.model.user_feat_emb.keys())

        self.target_return_min = getattr(args, "target_return_min", None)
        self.target_return_max = getattr(args, "target_return_max", None)
        if (self.target_return_min is not None) and (self.target_return_max is not None):
            assert self.target_return_max >= self.target_return_min,\
                "target_return_max  >= target_return_min"

        self.reset_state(env.episode_batch_size)
        self.forbid_repeat_item = getattr(args, "forbid_repeat_item", False)
        self.forbid_clicked_item = getattr(args, "forbid_clicked_item", False)


    def _sample_target_return(self, batch_size: int) -> torch.Tensor:
        """
        Sample target_return per episode.
        Uses [target_return_min, target_return_max] when provided,
        otherwise falls back to fixed target_return.
        """
        if (self.target_return_min is not None) and (self.target_return_max is not None):
            low = float(self.target_return_min)
            high = float(self.target_return_max)
            vals = torch.rand(batch_size, device=self.device) * (high - low) + low
        else:
            vals = torch.full(
                (batch_size,),
                float(self.target_return),
                dtype=torch.float32,
                device=self.device
            )
        return vals

    
    def reset_state(self, batch_size):
        self.cur_rtg = self._sample_target_return(batch_size)

        self.cur_timestep = torch.zeros(batch_size, dtype=torch.long, device=self.device)

        self.served_items = [set() for _ in range(batch_size)]
        self.clicked_items = [set() for _ in range(batch_size)]




    @torch.no_grad()
    def act(self, observation, candidate_info):
        """
        Build slate actions from current observation and candidate set.
        The model predicts next-item logits from right-aligned history tokens.
        """
        user_profile = observation['user_profile']
        user_history = observation['user_history']

        item_seq_raw = user_history['history'].long()         # [B, L]
        hist_len = user_history['history_length'].long()      # [B]
        B, L = item_seq_raw.shape
        device = self.device

        pos = torch.arange(L, device=device).unsqueeze(0)     # [1, L]
        time_seq = (self.cur_timestep.unsqueeze(1) + pos).clamp(max=self.max_timestep - 1)  # [B, L]

        item_seq = torch.zeros_like(item_seq_raw)
        attn_mask = torch.zeros((B, L), dtype=torch.float32, device=device)

        for b in range(B):
            hlen = int(hist_len[b].item())
            valid_pos = torch.nonzero(item_seq_raw[b] != 0, as_tuple=False).view(-1)
            if valid_pos.numel() == 0 or hlen <= 0:
                continue
            k = min(hlen, int(valid_pos.numel()), L - 1)
            if k <= 0:
                continue
            sel = valid_pos[-k:]
            dst_start = L - 1 - k
            dst_end = L - 1
            item_seq[b, dst_start:dst_end] = item_seq_raw[b, sel]
            attn_mask[b, dst_start:dst_end] = 1.0

        rtg_seq = (self.cur_rtg / self.rtg_scale).unsqueeze(1).expand(B, L)

        feed_dict = {
            'input_items': item_seq,
            'input_rtgs': rtg_seq,
            'input_timesteps': time_seq,
            'user_id': user_profile['user_id'].long(),
            'attention_mask': attn_mask
        }

        for feat_name in self.user_feat_names:
            key = f'uf_{feat_name}'
            if key in user_profile:
                feed_dict[key] = user_profile[key].float()

        out = self.model(feed_dict)
        logits = out['logits'][:, -1, :]          # [B, V]
        logits[:, 0] = -float('inf')              # padding id = 0 

        candidate_ids = candidate_info['item_id'].to(device)      # [N]
        cand_logits = logits.index_select(1, candidate_ids)       # [B, N]

        B2, N = cand_logits.shape
        assert B2 == B

        if self.forbid_repeat_item or self.forbid_clicked_item:
            for b in range(B):
                banned = set()
                if self.forbid_repeat_item:
                    banned |= self.served_items[b]
                if self.forbid_clicked_item:
                    banned |= self.clicked_items[b]
                if not banned:
                    continue

                banned_tensor = torch.tensor(list(banned), dtype=torch.long, device=device)
                mask_b = (candidate_ids.unsqueeze(0) == banned_tensor.unsqueeze(1)).any(dim=0)  # [N]
                if mask_b.all():
                    continue
                cand_logits[b, mask_b] = -float('inf')

        if getattr(self, "temperature", 0.0) and float(self.temperature) > 0.0:
            temp = float(self.temperature)
            probs = torch.softmax(cand_logits / temp, dim=1)  # [B, N]
            action_idx = torch.empty((B, self.slate_size), dtype=torch.long, device=device)

            for b in range(B):
                p = probs[b]
                if (not torch.isfinite(p).all()) or (p.sum() <= 0):
                    valid = torch.isfinite(cand_logits[b])  # not -inf
                    idxs = torch.nonzero(valid, as_tuple=False).view(-1)
                    if idxs.numel() == 0:
                        action_idx[b] = torch.randint(0, N, (self.slate_size,), device=device)
                    elif idxs.numel() >= self.slate_size:
                        perm = idxs[torch.randperm(idxs.numel(), device=device)[:self.slate_size]]
                        action_idx[b] = perm
                    else:
                        action_idx[b] = idxs[torch.randint(0, idxs.numel(), (self.slate_size,), device=device)]
                else:
                    nonzero = int((p > 0).sum().item())
                    replacement = nonzero < self.slate_size
                    action_idx[b] = torch.multinomial(p, num_samples=self.slate_size, replacement=replacement)

            return action_idx   # candidate index
        else:
            _, greedy_idx = torch.topk(cand_logits, k=self.slate_size, dim=1)  # [B, slate]
            return greedy_idx   # candidate index

    def update_after_step(self, step_reward, done, candidate_ids=None, action=None, click_mask=None):
        """
        Update RTG/timestep state after env.step and refresh dedupe sets.
        """
        if (self.forbid_repeat_item or self.forbid_clicked_item) and candidate_ids is not None and action is not None:
            B, slate_size = action.shape
            for b in range(B):
                chosen_ids = candidate_ids[action[b].long()].tolist()
                if self.forbid_repeat_item and chosen_ids:
                    self.served_items[b].update(int(x) for x in chosen_ids)

                if self.forbid_clicked_item and click_mask is not None:
                    clicked_slots = torch.nonzero(click_mask[b], as_tuple=False).view(-1)
                    if clicked_slots.numel() > 0:
                        clicked_ids = candidate_ids[action[b, clicked_slots].long()].tolist()
                        self.clicked_items[b].update(int(x) for x in clicked_ids)

        self.cur_rtg = self.cur_rtg - step_reward.to(self.device)
        self.cur_rtg = torch.clamp(self.cur_rtg, min=0.0)
        self.cur_timestep = self.cur_timestep + 1

        if done.any():
            idxs = torch.nonzero(done).squeeze(-1)

            if (self.target_return_min is not None) and (self.target_return_max is not None):
                new_rtg = self._sample_target_return(len(idxs))
                self.cur_rtg[idxs] = new_rtg
            else:
                self.cur_rtg[idxs] = float(self.target_return)

            self.cur_timestep[idxs] = 0
            for idx in idxs:
                i = int(idx.item())
                self.served_items[i].clear()
                self.clicked_items[i].clear()





def compute_returns(rewards, gamma=1.0):
    """
    Compute return-to-go from a reward list.
    """
    L = len(rewards)
    rtg = np.zeros(L, dtype=np.float32)
    G = 0.0
    for i in reversed(range(L)):
        G = rewards[i] + gamma * G
        rtg[i] = G
    return rtg


def build_batch_from_episodes(episodes, model, args, device):
    """
    Convert sampled trajectories into a DT training batch feed_dict.
    """
    if len(episodes) == 0:
        return None

    max_len = args.max_hist_seq_len
    B = len(episodes)

    item_seqs = torch.zeros((B, max_len), dtype=torch.long, device=device)
    rtg_seqs = torch.zeros((B, max_len), dtype=torch.float32, device=device)
    time_seqs = torch.zeros((B, max_len), dtype=torch.long, device=device)
    attn_mask = torch.zeros((B, max_len), dtype=torch.float32, device=device)
    user_ids = torch.zeros((B,), dtype=torch.long, device=device)

    user_feat_tensors = {}
    for feat_name, proj_layer in model.user_feat_emb.items():
        key = f'uf_{feat_name}'
        dim = proj_layer.in_features
        user_feat_tensors[key] = torch.zeros((B, dim), dtype=torch.float32, device=device)

    for i, traj in enumerate(episodes):
        items = traj.items
        rewards = traj.rewards
        L = min(len(items), max_len)

        items = items[-L:]
        rewards = rewards[-L:]

        rtg = compute_returns(rewards, gamma=1.0)   # [L]

        item_seqs[i, :L] = torch.tensor(items, dtype=torch.long, device=device)
        attn_mask[i, :L] = 1.0
        time_seqs[i, :L] = torch.arange(L, device=device)
        rtg_seqs[i, :L] = torch.tensor(rtg, dtype=torch.float32, device=device) / args.rtg_scale

        user_ids[i] = traj.user_id

        for key, tensor in user_feat_tensors.items():
            if key in traj.user_feats:
                feat_vec = torch.tensor(traj.user_feats[key], dtype=torch.float32, device=device)
                tensor[i] = feat_vec

    feed_dict = {
        'input_items': item_seqs,
        'input_rtgs': rtg_seqs,
        'input_timesteps': time_seqs,
        'attention_mask': attn_mask,
        'user_id': user_ids
    }
    feed_dict.update(user_feat_tensors)
    return feed_dict



def collect_episodes(env, policy, buffer, args, device):
    """
    Roll out policy in the simulator and append finished episodes to buffer.
    """
    observation = env.reset()
    batch_size = env.episode_batch_size
    policy.reset_state(batch_size)

    response_weights = env.response_weights

    traj_per_slot = []
    user_profile = observation['user_profile']
    user_feat_keys = [k for k in user_profile.keys() if k != 'user_id']

    for b in range(batch_size):
        uid = user_profile['user_id'][b].item()
        feats = {k: user_profile[k][b].detach().cpu().numpy() for k in user_feat_keys}
        traj_per_slot.append(Trajectory(uid, feats))

    served_items_cov = [set() for _ in range(batch_size)]

    collected_episodes = 0
    max_steps = args.collect_max_steps

    epoch_total_returns = []

    step_count = 0
    while collected_episodes < args.collect_episodes_per_epoch and step_count < max_steps:
        step_count += 1

        try:
            cand = env.get_candidate_info(feed_dict=None)
        except TypeError:
            cand = env.get_candidate_info()

        cand_ids = cand['item_id'].to(device)      # [N]

        action = policy.act(observation, cand)     # [B, slate_size]

        next_obs, resp, _ = env.step({'action': action})

        im_resp = torch.as_tensor(resp['immediate_response'], device=device)
        B, slate_size = action.shape

        if im_resp.dim() == 3:
            click_mask = (im_resp[..., 0] > 0)  # [B, slate_size]
        else:
            click_mask = torch.zeros(B, slate_size, dtype=torch.bool, device=device)

        rw_dict = {
            'immediate_response': resp['immediate_response'],
            'immediate_response_weight': response_weights
        }
        base_step_r = get_immediate_reward(rw_dict).to(device)   # [B]

        cov_coef = getattr(args, "coverage_penalty_coef", 0.0)
        penalty = torch.zeros(B, device=device)

        for b in range(batch_size):
            for j in range(slate_size):
                cand_idx = int(action[b, j].item())
                item_id = int(cand_ids[cand_idx].item())

                if cov_coef > 0.0 and (item_id in served_items_cov[b]):
                    penalty[b] += cov_coef

                served_items_cov[b].add(item_id)

        step_r = base_step_r - penalty

        im_resp = torch.as_tensor(resp['immediate_response'], device=device)
        B, slate_size = action.shape

        if im_resp.dim() == 3:
            click_mask = (im_resp[..., 0] > 0)  # [B, slate_size]
        else:
            click_mask = torch.zeros(B, slate_size, dtype=torch.bool, device=device)

        slot_rewards = torch.zeros(B, slate_size, device=device)

        for b in range(B):
            clicked_slots = torch.nonzero(click_mask[b], as_tuple=False).view(-1)
            if clicked_slots.numel() == 0:
                continue
            r_step = float(step_r[b].item())
            r_each = r_step / float(clicked_slots.numel())
            for j in clicked_slots:
                slot_rewards[b, int(j.item())] = r_each

        for b in range(batch_size):
            clicked_slots = torch.nonzero(click_mask[b], as_tuple=False).view(-1)
            if clicked_slots.numel() == 0:
                continue

            for j in clicked_slots:
                j = int(j.item())
                cand_idx = action[b, j]
                item_id = cand_ids[cand_idx].item()
                r_ij = slot_rewards[b, j].item()
                traj_per_slot[b].add_step(item_id, r_ij)



        policy.update_after_step(
            step_r,
            resp['done'],
            candidate_ids=cand_ids,
            action=action,
            click_mask=click_mask,
        )

        done = resp['done']    # [B] bool
        if done.any():
            idxs = torch.nonzero(done).squeeze(-1)
            for idx in idxs:
                idx = idx.item()
                traj = traj_per_slot[idx]

                if traj.length() > 0:
                    epoch_total_returns.append(traj.total_return)

                buffer.add(traj)
                collected_episodes += 1

                served_items_cov[idx] = set()

                if collected_episodes >= args.collect_episodes_per_epoch:
                    break

                new_profile = next_obs['user_profile']
                uid = new_profile['user_id'][idx].item()
                feats = {k: new_profile[k][idx].detach().cpu().numpy()
                         for k in user_feat_keys}
                traj_per_slot[idx] = Trajectory(uid, feats)


        observation = next_obs

    if epoch_total_returns:
        arr = np.array(epoch_total_returns, dtype=np.float32)
        ps = [1, 25, 50, 80, 90, 95, 99]
        print("[Collect] Total return percentiles for this epoch (based on finished episodes):")
        for p in ps:
            val = float(np.percentile(arr, p))
            print(f"  p{p:2d} = {val:.4f}")
        print(f"  mean = {arr.mean():.4f}")
    else:
        print("[Collect] No finished episodes with clicks this epoch, skip percentile stats.")
    mean_return = float(arr.mean()) if epoch_total_returns else None
    return collected_episodes, mean_return


def train_dt_in_env(args):
    if args.cuda >= 0 and torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    args.device = device
    print(f"[Info] Using device: {device}")

    print("[Env] Initializing KREnvironment_WholeSession_GPU ...")
    env = KREnvironment_WholeSession_GPU(args)

    reader_stats = env.reader.get_statistics()
    print("[Env] Reader stats for DT training:", reader_stats)

    model = DecisionTransformerRec(args, reader_stats, device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    model.optimizer = optimizer
    print(model)

    save_path = args.model_path  #  shell  DT_ENV_MODEL_PATH

    if args.init_dt_model:
        ckpt_prefix = args.init_dt_model.rstrip('.checkpoint')
        print(f"[Init] Loading pretrained DT from {ckpt_prefix}.checkpoint")
        model.load_from_checkpoint(ckpt_prefix, with_optimizer=False)
        if save_path:  # 
            model.model_path = save_path
            print(f"[Init] Will SAVE env-trained DT to: {model.model_path}.checkpoint")

    policy = DTPolicy(model, env, args)
    buffer = ReplayBuffer(max_episodes=args.buffer_max_episodes)

    global_step = 0
    best_mean_return = None
    for epoch in range(1, args.epoch + 1):
        t0 = time.time()
        print(f"\n===== Epoch {epoch} =====")

        collected, mean_return = collect_episodes(env, policy, buffer, args, device)
        print(f"[Collect] Epoch {epoch}: collected {collected} episodes, "
              f"buffer size = {len(buffer)}")

        model.train()
        epoch_losses = []

        for it in range(args.train_updates_per_epoch):
            episodes = buffer.sample_episodes(args.train_batch_size)
            if len(episodes) == 0:
                break

            feed_dict = build_batch_from_episodes(episodes, model, args, device)
            if feed_dict is None:
                continue

            out_dict = model.do_forward_and_loss(feed_dict)
            loss = out_dict['loss']

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            global_step += 1
            epoch_losses.append(loss.item())

            if (it + 1) % 10 == 0:
                print(f"[Train] Epoch {epoch} Iter {it+1}/{args.train_updates_per_epoch} "
                      f"Loss = {loss.item():.4f}")

        if epoch_losses:
            print(f"[Epoch {epoch}] Avg Loss = {np.mean(epoch_losses):.4f}, "
                  f"Min = {np.min(epoch_losses):.4f}, Max = {np.max(epoch_losses):.4f}")
        else:
            print(f"[Epoch {epoch}] No training updates (buffer empty).")

        print(f"[Epoch {epoch}] Time used: {time.time() - t0:.2f}s")

        if not args.model_path:
            args.model_path = "./dt_env_trained"
        if hasattr(model, "model_path"):
            model.model_path = args.model_path


        model.save_checkpoint()
        print(f"checkpoint saved to {args.model_path}.checkpoint")


def parse_args():
    parser = argparse.ArgumentParser()

    parser = KREnvironment_WholeSession_GPU.parse_model_args(parser)

    parser = DecisionTransformerRec.parse_model_args(parser)

    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.0)

    parser.add_argument('--epoch', type=int, default=10,
                        help='Training epochs (outer loop).')
    parser.add_argument('--cuda', type=int, default=0)

    parser.add_argument('--collect_episodes_per_epoch', type=int, default=200)
    parser.add_argument('--collect_max_steps', type=int, default=2000)

    parser.add_argument('--buffer_max_episodes', type=int, default=5000)
    parser.add_argument('--max_hist_seq_len', type=int, default=50)

    parser.add_argument('--train_batch_size', type=int, default=64)
    parser.add_argument('--train_updates_per_epoch', type=int, default=200)

    parser.add_argument('--init_dt_model', type=str,
                        default=str(PROJECT_ROOT / 'output/KuaiRand_Pure/env/DT_Pure_len50_lr0.0001_L3.model'),
                        help='Path to the offline DT checkpoint (without .checkpoint suffix).')

    parser.add_argument('--eps_greedy', type=float, default=0.3,
                        help='Epsilon-greedy exploration rate.')
    parser.add_argument('--temperature', type=float, default=0.0,
                        help='Sampling temperature. 0 means greedy decoding.')
    parser.add_argument('--target_return', type=float, default=10.0,
                        help='Target return used for RTG conditioning.')
    parser.add_argument('--target_return_min', type=float, default=None,
                        help='Lower bound for random target return sampling.')
    parser.add_argument('--target_return_max', type=float, default=None,
                        help='Upper bound for random target return sampling.')

    parser.add_argument('--forbid_repeat_item', action='store_true',
                        help='Disallow recommending the same item twice in one episode.')
    parser.add_argument('--forbid_clicked_item', action='store_true',
                        help='Disallow recommending items already clicked in one episode.')
    
    parser.add_argument('--coverage_penalty_coef', type=float, default=0.0,
                        help='Optional coverage penalty coefficient; set 0 to disable.')


    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    utils.set_random_seed(args.seed)
    train_dt_in_env(args)
