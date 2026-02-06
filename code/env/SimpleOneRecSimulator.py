import torch
import numpy as np
import pandas as pd
import utils
import os

from reader.KRMBSeqReaderOneRec import KRMBSeqReaderOneRec
from model.OneRecUserResponse import OneRecUserResponse

class SimpleOneRecSimulator:
    '''
 [fix v6]
 1. supports Slate Recommendation (List-wise). 
 2.  (B, Slate) -> (B*Slate) . 
 '''
    def __init__(self, args, device):
        self.device = device
        self.batch_size = args.val_batch_size
        
        if not hasattr(args, 'model_path'):
            args.model_path = getattr(args, 'simulator_ckpt', '')
        if not hasattr(args, 'loss'): args.loss = 'bce'
        if not hasattr(args, 'l2_coef'): args.l2_coef = 0.0
        
        self.args = args 
        
        print("\n[Simulator] Initializing Reader...")
        self.reader = KRMBSeqReaderOneRec(args)
        
        print("[Simulator] Caching Feature Matrices...")
        self.user_feat_tensors = {}
        for k, v in self.reader.user_feat_matrix.items():
            self.user_feat_tensors[k] = torch.from_numpy(v).float().to(device)
            
        self.item_feat_tensors = {}
        for k, v in self.reader.item_feat_matrix.items():
            self.item_feat_tensors[k] = torch.from_numpy(v).float().to(device)

        print("[Simulator] Loading OneRec Model...")
        stats = self.reader.get_statistics()
        self.model = OneRecUserResponse(args, stats, device)
        self.model.to(device)
        self.model.eval()
        
        ckpt_path = getattr(args, 'simulator_ckpt', args.model_path)
        if ckpt_path:
            candidates = [ckpt_path, ckpt_path + ".checkpoint", ckpt_path + ".pt"]
            loaded = False
            for path in candidates:
                if os.path.exists(path):
                    try:
                        checkpoint = torch.load(path, map_location=device)
                        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                            self.model.load_state_dict(checkpoint['model_state_dict'])
                        else:
                            self.model.load_state_dict(checkpoint)
                        print(f"[Simulator] Checkpoint loaded: {path}")
                        loaded = True
                        break
                    except Exception as e:
                        print(f"[Simulator] Error loading {path}: {e}")
            if not loaded:
                print(f"[Simulator] WARNING: Using RANDOM weights!")
        
        self.max_hist_len = args.max_hist_seq_len
        if hasattr(self.model, 'feedback_types'):
            self.feedback_types = self.model.feedback_types
        else:
            self.feedback_types = ['is_click', 'long_view', 'is_like', 'is_comment', 'is_forward', 'is_follow', 'is_hate']
        
        self.fb_map = {name: i for i, name in enumerate(self.feedback_types)}
        
        self.response_weights = torch.ones(len(self.feedback_types), device=device)
        if hasattr(args, 'single_response') and args.single_response:
            print("[Simulator] Single Response Mode: ON")
            self.response_weights.fill_(0.0)
            if 'is_click' in self.fb_map:
                self.response_weights[self.fb_map['is_click']] = 1.0
        
        self.reader.set_phase("test")
        self.num_test_samples = len(self.reader.data['test'])
        
        self.current_user_ids = None 
        self.current_history = None  
        self.current_history_response = {} 
        self.current_temper = None   
        
        self.initial_temper = getattr(args, 'initial_temper', 10.0)
        self.click_bonus = 2.0
        self.step_cost = 1.0
        self.logit_calibration = -3.0 

    def _collate_batch(self, batch_list):
        if not batch_list: return {}
        collated = {}
        keys = batch_list[0].keys()
        for key in keys:
            collated[key] = np.array([sample[key] for sample in batch_list])
        return collated

    def reset(self, batch_size=None):
        if batch_size is None: batch_size = self.batch_size
        indices = np.random.randint(0, self.num_test_samples, batch_size)
        batch_list = [self.reader[i] for i in indices]
        batch_dict = self._collate_batch(batch_list)
        batch_data = utils.wrap_batch(batch_dict, device=self.device)
        
        self.current_user_ids = batch_data['user_id']
        self.current_history = batch_data['history']
        self.current_temper = torch.full((batch_size,), self.initial_temper, device=self.device)
        
        self.current_history_response = {}
        for fb in self.feedback_types:
            key = f"history_{fb}"
            if key in batch_data:
                self.current_history_response[fb] = batch_data[key]
            else:
                self.current_history_response[fb] = torch.zeros_like(self.current_history, dtype=torch.float)
        
        return self._get_observation()

    def _get_observation(self):
        return {
            'user_id': self.current_user_ids,
            'history': self.current_history
        }

    def step(self, action_dict):
        rec_items = action_dict['action'].long()
        
        is_slate = (rec_items.dim() == 2)
        if is_slate:
            B, Slate = rec_items.shape
            flat_items = rec_items.view(-1)
            flat_users = self.current_user_ids.unsqueeze(1).expand(B, Slate).reshape(-1)
            flat_hist = self.current_history.unsqueeze(1).expand(B, Slate, -1).reshape(B*Slate, -1)
            flat_hist_resp = {}
            for k, v in self.current_history_response.items():
                flat_hist_resp[k] = v.unsqueeze(1).expand(B, Slate, -1).reshape(B*Slate, -1)
                
            eff_users = flat_users
            eff_items = flat_items
            eff_hist = flat_hist
            eff_hist_resp = flat_hist_resp
            eff_B = B * Slate
        else:
            B = rec_items.shape[0]
            Slate = 1
            eff_users = self.current_user_ids
            eff_items = rec_items
            eff_hist = self.current_history
            eff_hist_resp = self.current_history_response
            eff_B = B

        feed_dict = {
            'user_id': eff_users,
            'item_id': eff_items,
            'history': eff_hist,
            'history_length': (eff_hist > 0).sum(dim=1)
        }
        for k, v in self.user_feat_tensors.items(): feed_dict[k] = v[eff_users]
        for k, v in self.item_feat_tensors.items(): feed_dict[k] = v[eff_items]
        
        flat_hist_items = eff_hist.view(-1)
        for k, v in self.item_feat_tensors.items():
            feat_val = v[flat_hist_items].view(eff_B, self.max_hist_len, -1)
            feed_dict[f'history_{k}'] = feat_val

        for fb in self.feedback_types:
            feed_dict[f"history_{fb}"] = eff_hist_resp[fb]

        with torch.no_grad():
            out = self.model(feed_dict)
            logits = out['preds'] 
            if logits.dim() == 3: logits = logits.squeeze(1)
            
            logits = logits + self.logit_calibration
            probs = torch.sigmoid(logits)
            
        rand_vals = torch.rand_like(probs)
        feedback_bool = (rand_vals < probs) # (B*Slate, n_feedback)
        
        step_res = {}
        if is_slate:
            step_res['immediate_response'] = feedback_bool.view(B, Slate, -1).float()
        else:
            step_res['immediate_response'] = feedback_bool.unsqueeze(1).float()
            
        step_res['immediate_response_weight'] = self.response_weights
        
        
        
        shift = Slate
        if shift < self.max_hist_len:
            new_hist = torch.cat([self.current_history[:, shift:], rec_items.view(B, Slate)], dim=1)
        else:
            new_hist = rec_items.view(B, Slate)[:, -self.max_hist_len:]
        self.current_history = new_hist
        
        click_idx = self.fb_map.get('is_click', 0)
        
        for fb_name in self.feedback_types:
            idx = self.fb_map[fb_name]
            current_val = feedback_bool[:, idx].float().view(B, Slate)
            
            if shift < self.max_hist_len:
                old_hist = self.current_history_response[fb_name][:, shift:]
                self.current_history_response[fb_name] = torch.cat([old_hist, current_val], dim=1)
            else:
                self.current_history_response[fb_name] = current_val[:, -self.max_hist_len:]
        
        
        clicks_per_user = feedback_bool[:, click_idx].float().view(B, Slate).sum(dim=1)
        
        self.current_temper += (clicks_per_user * self.click_bonus - self.step_cost)
        done_mask = (self.current_temper <= 0)
        step_res['done'] = done_mask
        
        if done_mask.any():
            num_reset = done_mask.sum().item()
            new_indices = np.random.randint(0, self.num_test_samples, num_reset)
            new_list = [self.reader[i] for i in new_indices]
            new_batch_dict = self._collate_batch(new_list)
            new_batch = utils.wrap_batch(new_batch_dict, device=self.device)
            
            self.current_user_ids[done_mask] = new_batch['user_id']
            self.current_history[done_mask] = new_batch['history']
            self.current_temper[done_mask] = self.initial_temper
            
            for fb in self.feedback_types:
                key = f"history_{fb}"
                if key in new_batch:
                    self.current_history_response[fb][done_mask] = new_batch[key]
                else:
                    self.current_history_response[fb][done_mask] = 0.0
            
        next_obs = self._get_observation()
        return next_obs, step_res, None