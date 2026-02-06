import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from reader.KRMBSeqReader import KRMBSeqReader
from utils import padding_and_clip

class KRMBSeqReaderDT(KRMBSeqReader):
    '''
    Decision Transformer Reader (Robust Version)
    '''

    @staticmethod
    def parse_data_args(parser):
        """
  KRMBSeqReader.parse_data_args,  --single_response . 
 """
        parser = KRMBSeqReader.parse_data_args(parser)
        parser.add_argument(
            '--single_response',
            action='store_true',
            help='if,  reward  is_click ()'
        )
        return parser

    def print_stats(name, arr):
        print(f"\n====== {name}  ======")
        for p in [0, 25, 50, 75, 90, 95, 99, 100]:
            v = np.percentile(arr, p)
            print(f"  p{p:2d} = {v:.4f}")
        print(f"  mean = {arr.mean():.4f}")

    
    def __init__(self, args):
        print("Initiating Decision Transformer Reader (KRMBSeqReaderDT)...")
        self.max_len = args.max_hist_seq_len
        self.rtg_scale = args.rtg_scale if hasattr(args, 'rtg_scale') else 10.0
        self.single_response = getattr(args, "single_response", False)
        super().__init__(args)

    def _read_data(self, args):
        super()._read_data(args)
        
        print("DT Reader: optimizing temporal order & cleaning...")
        self.log_data.fillna(0, inplace=True)

        if 'time_ms' in self.log_data.columns:
            sort_keys = (self.log_data['time_ms'].values, self.log_data['user_id'].values)
            sorted_indices = np.lexsort(sort_keys)
            
            new_history = {}
            df_sorted = self.log_data.iloc[sorted_indices]
            grouped = df_sorted.groupby('user_id', sort=False)
            
            for uid, group in tqdm(grouped):
                new_history[uid] = group.index.tolist()
            self.user_history = new_history
        
        rewards = np.zeros(len(self.log_data), dtype=np.float32)

        if self.single_response:
            if 'is_click' in self.log_data.columns:
                rewards += self.log_data['is_click'].values.astype(np.float32) * 1.0
        else:
            if 'is_click' in self.log_data.columns:
                rewards += self.log_data['is_click'].values.astype(np.float32) * 1.0
            if 'long_view' in self.log_data.columns:
                rewards += self.log_data['long_view'].values.astype(np.float32) * 1.0
            if 'is_like' in self.log_data.columns:
                rewards += self.log_data['is_like'].values.astype(np.float32) * 2.0
            if 'is_follow' in self.log_data.columns:
                rewards += self.log_data['is_follow'].values.astype(np.float32) * 5.0
            if 'is_hate' in self.log_data.columns:
                rewards -= self.log_data['is_hate'].values.astype(np.float32) * 5.0

        self.log_data['temp_reward'] = rewards
        
        df_sorted = self.log_data.iloc[sorted_indices].copy()
        df_sorted['rtg'] = df_sorted.groupby('user_id')['temp_reward'].transform(lambda x: x[::-1].cumsum()[::-1])
        
        self.log_data['rtg'] = 0.0
        self.log_data.loc[df_sorted.index, 'rtg'] = df_sorted['rtg']
        
        self.all_user_ids = self.log_data['user_id'].map(self.user_id_vocab).fillna(0).values.astype(np.int32)
        self.all_item_ids = self.log_data['video_id'].map(self.item_id_vocab).fillna(0).values.astype(np.int32)
        
        raw_rtgs = self.log_data['rtg'].values.astype(np.float32)
        raw_rtgs = np.nan_to_num(raw_rtgs, nan=0.0, posinf=0.0, neginf=0.0)
        self.all_rtgs = raw_rtgs / (self.rtg_scale + 1e-6)
        
        del df_sorted
        self.log_data.drop(columns=['temp_reward'], inplace=True)
        print(f"DT Reader Ready.")

    def __getitem__(self, idx):
        row_id = self.data[self.phase][idx]
        user_id_enc = self.all_user_ids[row_id]
        
        user_id_raw = self.log_data.at[row_id, 'user_id']
        if user_id_raw not in self.user_history:
             return self._get_empty_sample(user_id_enc)

        history_indices = self.user_history[user_id_raw]
        try:
            t_idx = history_indices.index(row_id)
        except ValueError:
            t_idx = 0
            
        start_idx = max(0, t_idx - self.max_len + 1)
        end_idx = t_idx + 1
        window_row_ids = history_indices[start_idx : end_idx]
        L = len(window_row_ids)
        
        if L == 0:
             return self._get_empty_sample(user_id_enc)

        seq_items = self.all_item_ids[window_row_ids]
        seq_rtgs = self.all_rtgs[window_row_ids]
        seq_timesteps = np.arange(0, L, dtype=np.int32)
        
        pad_len = self.max_len - L
        
        actions_padded = np.pad(seq_items, (pad_len, 0), 'constant', constant_values=0)
        rtgs_padded = np.pad(seq_rtgs, (pad_len, 0), 'constant', constant_values=0)
        timesteps_padded = np.pad(seq_timesteps, (pad_len, 0), 'constant', constant_values=0)
        mask = np.concatenate([np.zeros(pad_len), np.ones(L)])
        
        record = {
            'user_id': user_id_enc,
            'item_id': seq_items[-1],
            'input_items': actions_padded,
            'input_rtgs': rtgs_padded,
            'input_timesteps': timesteps_padded,
            'attention_mask': mask
        }
        
        record.update(self.get_user_meta_data(user_id_enc))
        for f in self.response_list: record[f] = 0
        return record

    def _get_empty_sample(self, uid):
        mask = np.zeros(self.max_len, dtype=float)
        mask[-1] = 1.0 
        
        return {
            'user_id': uid,
            'item_id': 0,
            'input_items': np.zeros(self.max_len, dtype=int),
            'input_rtgs': np.zeros(self.max_len, dtype=np.float32),
            'input_timesteps': np.zeros(self.max_len, dtype=int),
            'attention_mask': mask,
            **self.get_user_meta_data(uid),
            **{f:0 for f in self.response_list}
        }