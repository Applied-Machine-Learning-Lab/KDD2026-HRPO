import torch
import numpy as np
import os
import pandas as pd
from model.agent.BaseRLAgent import BaseRLAgent
from model.DecisionTransformerRec import DecisionTransformerRec
from reader.KRMBSeqReaderDT import KRMBSeqReaderDT
import utils

class DTAgent(BaseRLAgent):
    '''
    Decision Transformer Agent adapted for SimpleOneRecSimulator.
    '''
    def __init__(self, args, device):
        self.args = args
        self.device = device
        
        print(f"[DTAgent] Initializing DT Reader (to align features with training)...")
        self.reader = KRMBSeqReaderDT(args)
        
        print("[DTAgent] Caching User Features...")
        self.user_feat_tensors = {}
        self._cache_user_features()

        print(f"[DTAgent] Loading Decision Transformer Model...")
        self.model = DecisionTransformerRec(args, self.reader.get_statistics(), device)
        self.model.to(device)
        
        if args.model_path:
            self._load_model(args.model_path)
        else:
            print("[DTAgent] Warning: No model path provided!")
            
        self.model.eval()
        self.max_len = args.max_hist_seq_len
        
        self.current_targets = None 

    def _cache_user_features(self):
        '''
 user features GPU Tensor,  IO
 '''
        try:
            user_df = pd.read_csv(self.args.user_meta_file, sep=self.args.meta_file_sep).fillna('unknown')
            user_df.set_index('user_id', inplace=True)
        except Exception as e:
            print(f"[DTAgent] Error reading user meta: {e}")
            return

        for feat_name in self.reader.selected_user_features:
            vocab_map = self.reader.user_vocab[feat_name]
            
            sample_val = list(vocab_map.values())[0]
            if hasattr(sample_val, '__len__') and not isinstance(sample_val, str):
                dim = len(sample_val) # One-Hot vector
            else:
                dim = 1 # ID index
            
            matrix = np.zeros((len(self.reader.users) + 2, dim), dtype=np.float32)
            
            for uid in self.reader.users: # raw user id
                if uid in user_df.index:
                    raw_feat_val = user_df.loc[uid, feat_name]
                    if raw_feat_val in vocab_map:
                        vec = vocab_map[raw_feat_val]
                        enc_uid = self.reader.user_id_vocab[uid]
                        matrix[enc_uid] = vec
            
            key = f"uf_{feat_name}"
            if dim == 1:
                self.user_feat_tensors[key] = torch.tensor(matrix, device=self.device, dtype=torch.long).squeeze(-1)
            else:
                self.user_feat_tensors[key] = torch.tensor(matrix, device=self.device, dtype=torch.float32)
            
            print(f"   Cached {key}: {self.user_feat_tensors[key].shape}")

    def _load_model(self, load_path):
        print(f"[DTAgent] Loading weights: {load_path}")
        if not os.path.exists(load_path):
            if os.path.exists(load_path + '.checkpoint'): load_path += '.checkpoint'
            elif os.path.exists(load_path + '.pt'): load_path += '.pt'
        
        try:
            checkpoint = torch.load(load_path, map_location=self.device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            print("[DTAgent] Model loaded successfully.")
        except Exception as e:
            print(f"[DTAgent] Error loading model: {e}")

    def reset_session(self, batch_size, initial_target_return):
        '''
        Reset RTG for a new batch
        '''
        self.current_targets = torch.full((batch_size,), initial_target_return, 
                                          dtype=torch.float32, device=self.device)

    def update_targets(self, rewards, done_mask, initial_target_return):
        '''
 Step RTG
 rtg_new = rtg_old - reward
 '''
        if self.current_targets is None: return
        
        self.current_targets -= rewards
        
        self.current_targets = torch.clamp(self.current_targets, min=0.0)
        
        if done_mask is not None and done_mask.any():
            self.current_targets[done_mask] = initial_target_return

    def choose_action(self, observation):
        '''
        Batch Inference
        observation: {'user_id': Tensor(B,), 'history': Tensor(B, L)}
        '''
        user_ids = observation['user_id'].to(self.device).long()
        B = user_ids.shape[0]
        
        if self.current_targets is None or self.current_targets.shape[0] != B:
            self.reset_session(B, 20.0) # Fallback

        history_raw = observation['history'].to(self.device).long()
        if history_raw.shape[1] > self.max_len:
            history_raw = history_raw[:, -self.max_len:]

        curr_seq_len = history_raw.shape[1]
        pad_len = self.max_len - curr_seq_len
        
        if pad_len > 0:
            pad_tensor = torch.zeros((B, pad_len), dtype=torch.long, device=self.device)
            input_items = torch.cat([pad_tensor, history_raw], dim=1)
        else:
            input_items = history_raw
            
        input_rtgs = self.current_targets.unsqueeze(1).expand(B, self.max_len)
        
        steps = torch.arange(curr_seq_len, device=self.device).unsqueeze(0).expand(B, -1)
        if pad_len > 0:
            pad_steps = torch.zeros((B, pad_len), dtype=torch.long, device=self.device)
            input_timesteps = torch.cat([pad_steps, steps], dim=1)
        else:
            input_timesteps = steps
            
        mask = torch.ones((B, curr_seq_len), device=self.device)
        if pad_len > 0:
            pad_mask = torch.zeros((B, pad_len), device=self.device)
            mask = torch.cat([pad_mask, mask], dim=1)
        if curr_seq_len == 0: mask[:, -1] = 1.0

        feed_dict = {
            'user_id': user_ids,
            'input_items': input_items,
            'input_rtgs': input_rtgs,
            'input_timesteps': input_timesteps.long(),
            'attention_mask': mask
        }
        
        for feat_name, feat_tensor in self.user_feat_tensors.items():
            batch_feats = feat_tensor[user_ids] 
            feed_dict[feat_name] = batch_feats

        with torch.no_grad():
            out = self.model(feed_dict)
            last_logits = out['logits'][:, -1, :] # (B, V)
            
            last_logits[:, 0] = -float('inf')
            
            if curr_seq_len > 0:
                valid_hist_mask = (history_raw > 0)
                last_logits.scatter_(1, history_raw * valid_hist_mask.long(), -float('inf'))

            actions = torch.argmax(last_logits, dim=1) # (B,)
            
        return actions