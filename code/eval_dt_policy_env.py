
import os
import argparse
import time
import numpy as np
import torch

from env.KREnvironment_WholeSession_GPU import KREnvironment_WholeSession_GPU
from model.DecisionTransformerRec import DecisionTransformerRec
from model.reward import get_immediate_reward
import utils  #  HAC random seed

def parse_args():
    parser = argparse.ArgumentParser()
    parser = KREnvironment_WholeSession_GPU.parse_model_args(parser)  # 

    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--n_layer', type=int, default=3)
    parser.add_argument('--n_head', type=int, default=4)
    parser.add_argument('--max_timestep', type=int, default=500)
    parser.add_argument('--dropout_rate', type=float, default=0.1)
    parser.add_argument('--rtg_scale', type=float, default=10.0)
    parser.add_argument('--eps_greedy', type=float, default=0.0,
                    help='eval  epsilon-greedy , for coverage(default 0 )')


    parser.add_argument('--temperature', type=float, default=0.0,
                    help=': 0  topk, >0  softmax(logits/temperature) ')

    parser.add_argument('--max_hist_seq_len', type=int, default=50)
    parser.add_argument('--l2_coef', type=float, default=0.0)
    parser.add_argument('--loss', type=str, default='ce')
    parser.add_argument("--model_path", type=str, default="", help="")

    parser.add_argument('--dt_model_path', type=str, required=True,
                        help='DT model,  .checkpoint,  .../DT_Pure_len50_lr0.0001_L3.model')

    parser.add_argument("--num_episodes", type=int, default=200)
    parser.add_argument("--max_steps_per_episode", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument('--target_return', type=float, default=10.0,
                        help=' episode  ( rtg_scale )')
    parser.add_argument('--forbid_repeat_item', action='store_true',
                        help=',  episode  item')
    parser.add_argument('--forbid_clicked_item', action='store_true',
                        help=',  episode click item')
    parser.add_argument('--debug_padding', action='store_true',
                        help='[Debug] :  env  history padding , hist_len ,  logits[:, -1]  CLS/decision token')
    parser.add_argument('--debug_topk', type=int, default=10,
                        help='debug_padding candidate topK  K')

    args = parser.parse_args()
    return args


class DTPolicy:
    """
  DecisionTransformerRec : 
 - usercurrent" RTG"(cur_rtg)
 - : cur_rtg -= step_reward
 """
    def __init__(self, model, env, args):
        self.model = model
        self.env = env
        self.device = args.device
        self.slate_size = args.slate_size
        self.rtg_scale = args.rtg_scale
        self.max_timestep = args.max_timestep
        self.target_return = args.target_return
        self.eps_greedy = getattr(args, "eps_greedy", 0.0)
        self.temperature = getattr(args, "temperature", 0.0)
        self.user_feat_names = list(getattr(self.model, 'user_feat_emb', {}).keys())

        self.reset_state(env.episode_batch_size)

        self.forbid_repeat_item = getattr(args, "forbid_repeat_item", False)
        self.forbid_clicked_item = getattr(args, "forbid_clicked_item", False)

        self.debug_padding = getattr(args, 'debug_padding', False)
        self.debug_topk = getattr(args, 'debug_topk', 10)
        self._debug_printed = False


    def reset_state(self, batch_size):
        self.cur_rtg = torch.full(
            (batch_size,),
            self.target_return,
            dtype=torch.float32,
            device=self.device
        )
        self.cur_timestep = torch.zeros(batch_size, dtype=torch.long, device=self.device)

        self.served_items = [set() for _ in range(batch_size)]
        self.clicked_items = [set() for _ in range(batch_size)]


    @torch.no_grad()
    def act(self, observation, candidate_info):
        """
 observation:
 - user_profile: dict
 - user_history: dict with keys:
 * history: [B, L] (padding = 0)
 * history_length: [B] ( token ;  token )

 candidate_info:
 - item_id: [N] (candidate set encoded item id,  DT vocab )

 :
 - action_idx: [B, slate_size], candidate set( item id)
 """
        user_profile = observation['user_profile']
        user_history = observation['user_history']

        item_seq_raw = user_history['history'].long()             # [B, L]
        hist_len = user_history['history_length'].long()          # [B]
        B, L = item_seq_raw.shape
        device = self.device

        pos = torch.arange(L, device=device).unsqueeze(0)         # [1, L]
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
            'attention_mask': attn_mask,
        }

        for feat_name in self.user_feat_names:
            key = f'uf_{feat_name}'
            if key in user_profile:
                feed_dict[key] = user_profile[key].float()

        out = self.model(feed_dict)
        logits = out['logits'][:, -1, :]  # [B, V]
        logits[:, 0] = -float('inf')      # padding id 

        candidate_ids = candidate_info['item_id'].to(device)       # [N]
        cand_logits = logits.index_select(1, candidate_ids)        # [B, N]
        N = cand_logits.shape[1]

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
                    valid = torch.isfinite(cand_logits[b])
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

            greedy_idx = action_idx
        else:
            _, greedy_idx = torch.topk(cand_logits, k=self.slate_size, dim=1)  # [B, slate]

        if getattr(self, "_printed_padcheck", False) is False:
            self._printed_padcheck = True
            with torch.no_grad():
                b0 = 0
                raw = item_seq_raw[b0].detach().cpu().tolist()
                rep = item_seq[b0].detach().cpu().tolist()
                msk = attn_mask[b0].detach().cpu().tolist()
                tms = time_seq[b0].detach().cpu().tolist()
                nz = [i for i,x in enumerate(raw) if x != 0]
                print("[Debug][PadCheck] ====== env history padding check (print once) ======")
                print(f"[Debug][PadCheck] B={B}, L={L} | hist_len0={int(hist_len[b0].item())} | nonzero_cnt0={len(nz)}")
                print(f"[Debug][PadCheck] raw head10={raw[:10]}")
                print(f"[Debug][PadCheck] raw tail10={raw[-10:]}")
                if nz:
                    print(f"[Debug][PadCheck] raw nonzero_pos={nz[:10]}{'...' if len(nz)>10 else ''} tail={nz[-10:]}")
                print(f"[Debug][PadCheck] rep tail10={rep[-10:]}")
                print(f"[Debug][PadCheck] attn_mask tail10={msk[-10:]}, sum={sum(msk):.0f}")
                print(f"[Debug][PadCheck] time_seq tail10={tms[-10:]}")
        return greedy_idx

    def update_after_step(self, step_reward, done, candidate_ids=None, action=None, click_mask=None):
        """
 environment reward user RTG  timestep; 
  step  / clickdedupe. 
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
            self.cur_rtg[idxs] = self.target_return
            self.cur_timestep[idxs] = 0
            for idx in idxs:
                i = int(idx.item())
                self.served_items[i].clear()
                self.clicked_items[i].clear()



def run_eval(args):
    device = torch.device(args.device)
    args.device = device
    print(f"[Info] Using device: {device}")

    print("[Env] Initializing environment.")
    env = KREnvironment_WholeSession_GPU(args)

    if hasattr(env, "set_seed"):
        env.set_seed(args.seed)
        print(f"[Seed] env.set_seed({args.seed}) called.")

    reader_stats = env.reader.get_statistics()
    print("[DT] Reader stats for DT:", reader_stats)

    model = DecisionTransformerRec(args, reader_stats, device).to(device)

    ckpt_path = args.dt_model_path + ".checkpoint"
    if os.path.exists(ckpt_path):
        print(f"[DT] Loading checkpoint from {ckpt_path}")
        model.load_from_checkpoint(args.dt_model_path, with_optimizer=False)
    else:
        print(f"[Warning] Checkpoint {ckpt_path} not found, using random init weights.")

    model.eval()

    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"[Model] Total parameters: {num_params:,d}")

    policy = DTPolicy(model, env, args)

    print(f"[Eval] DecisionTransformer Policy")
    print(f"      Slate Size: {args.slate_size}")
    print(f"      Max Steps: {args.max_steps_per_episode}")
    print(f"      Target Return: {args.target_return} (scale={args.rtg_scale})")
    print(f"      Single Response: {args.single_response}")

    observation = env.reset({'batch_size': args.episode_batch_size})
    batch_size = args.episode_batch_size

    cur_returns = torch.zeros(batch_size, device=device)
    cur_lengths = torch.zeros(batch_size, device=device)
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
    cur_beh_counts = torch.zeros(batch_size, K, device=device)   # (B, K)
    cur_impr = torch.zeros(batch_size, device=device)            # (B,)
    total_beh_counts = torch.zeros(K, device=device)             # (K,)
    total_impr = 0.0


    response_weights = env.response_weights  #  random eval  reward

    start_time = time.time()

    try:
        while finished < args.num_episodes:

            try:
                cand = env.get_candidate_info(feed_dict=None)
            except TypeError:
                cand = env.get_candidate_info()

            candidate_ids = cand['item_id'].to(device)
            cand['item_id'] = candidate_ids  #  policy.act

            action = policy.act(observation, cand)   # [B, slate_size]

            next_obs, resp, _ = env.step({'action': action})

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
                'immediate_response': resp['immediate_response'],
                'immediate_response_weight': response_weights
            }
            step_r = get_immediate_reward(rw_dict)  # [B]

            im_resp = torch.as_tensor(resp['immediate_response'], device=device)
            B, slate_size = action.shape
            if im_resp.dim() == 3:
                click_mask = (im_resp[..., 0] > 0)  # [B, slate_size]
            else:
                click_mask = torch.zeros(B, slate_size, dtype=torch.bool, device=device)

            policy.update_after_step(
                step_r,
                resp['done'],
                candidate_ids=candidate_ids,
                action=action,
                click_mask=click_mask,
            )


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

                        if finished % 20 == 0:
                            print(f"Progress {finished}/{args.num_episodes} | "
                                  f"Avg Ret: {np.mean(all_ret):.4f} | Avg Len: {np.mean(all_len):.2f}")

                cur_returns[done] = 0
                cur_lengths[done] = 0

                cur_beh_counts[done] = 0
                cur_impr[done] = 0
            observation = next_obs

    except KeyboardInterrupt:
        print("Interrupted by user.")

    if all_ret:
        avg_ret = np.mean(all_ret)
        avg_len = np.mean(all_len)
        avg_step_r = avg_ret / avg_len if avg_len > 0 else 0

        print("=" * 40)

        print(f"Total Reward: {avg_ret:.4f}")
        print(f"Depth: {avg_len:.2f}")
        print(f"Avg Step Reward: {avg_step_r:.4f}")

        coverage = 0.0
        ild = 0.0
        if hasattr(env, "get_report"):
            env_report = env.get_report()
            coverage = float(env_report.get("coverage", 0.0))
            coverage_ratio = float(env_report.get("coverage_ratio", 0.0))
            catalog_coverage = float(env_report.get("catalog_coverage", 0.0))
            ild = float(env_report.get("ILD", 0.0))

        print(f"Coverage: {coverage:.2f}")
        print(f"ILD: {ild:.4f}")

        print("Table-style metrics:")
        print(f"Depth: {avg_len:.2f}")
        print(f"Average reward: {avg_step_r:.4f}")
        print(f"Total reward: {avg_ret:.4f}")
        print(f"Coverage: {coverage:.2f}")
        print(f"coverage_ratio: {coverage_ratio:.2f}")
        print(f"catalog_coverage: {catalog_coverage:.2f}")
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
