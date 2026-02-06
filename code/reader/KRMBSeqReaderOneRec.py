import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from reader.KRMBSeqReader import KRMBSeqReader
from utils import padding_and_clip, get_onehot_vocab, get_multihot_vocab


class KRMBSeqReaderOneRec(KRMBSeqReader):
    '''
 KuaiRand Multi-Behavior Data Reader (High Performance Vectorized Version)
 : 
 1. feature Numpy ,  __getitem__ . 
 2. [FIX] feature 2D,  dim=1 error. 
 3. [FIX] filterfeaturefeature, modelinput. 
 '''

    def __init__(self, args):
        print("initiate KuaiRand OneRec Full-Feature sequence reader (Optimized & Cleaned)")
        super().__init__(args)

    def _read_data(self, args):
        print(f"Loading log data files...")
        self.log_data = pd.read_table(args.train_file, sep=args.data_separator)

        print("Load item meta data...")
        item_meta_df = pd.read_csv(args.item_meta_file, sep=args.meta_file_sep).fillna('unknown')
        
        print("Load user meta data...")
        user_meta_df = pd.read_csv(args.user_meta_file, sep=args.meta_file_sep).fillna('unknown')

        self.users = list(self.log_data['user_id'].unique())
        self.items = list(self.log_data['video_id'].unique())
        
        print("Building User History Index...")
        self.user_history = {uid: list(self.log_data[self.log_data['user_id'] == uid].index) for uid in self.users}

        self.user_id_vocab = {uid: i + 1 for i, uid in enumerate(self.users)}
        self.item_id_vocab = {iid: i + 1 for i, iid in enumerate(self.items)}

        ignore_user_cols = {
            'user_id', 
            'follow_user_num',    # ~2500 -> ,  follow_user_num_range
            'fans_user_num',      # ~2200 -> ,  fans_user_num_range
            'friend_user_num',    # ~1400 -> ,  friend_user_num_range
            'register_days',      # ~2800 -> ,  register_days_range
            'is_lowactive_period' # 1 () -> 
        }
        
        ignore_item_cols = {
            'video_id',
            'video_duration',     # ~5700 -> , 
            'visible_status'      # 1 () -> 
        }

        self.selected_user_features = [c for c in user_meta_df.columns if c not in ignore_user_cols]
        self.selected_item_features = [c for c in item_meta_df.columns if c not in ignore_item_cols]

        print(f"Selected User Features ({len(self.selected_user_features)}): {self.selected_user_features}")
        print(f"Selected Item Features ({len(self.selected_item_features)}): {self.selected_item_features}")

        user_meta_dict = user_meta_df.set_index('user_id').to_dict('index')
        item_meta_dict = item_meta_df.set_index('video_id').to_dict('index')

        self.user_vocab = get_onehot_vocab(user_meta_df, self.selected_user_features)

        multihot_cols = ['tag']
        onehot_cols = [c for c in self.selected_item_features if c not in multihot_cols]
        self.item_vocab = get_onehot_vocab(item_meta_df, onehot_cols)
        if 'tag' in self.selected_item_features:
            self.item_vocab.update(get_multihot_vocab(item_meta_df, multihot_cols))

        self.response_list = ['is_click', 'long_view', 'is_like', 'is_comment',
                              'is_forward', 'is_follow', 'is_hate']
        self.response_dim = len(self.response_list)
        self.response_neg_sample_rate = self.get_response_weights()

        print("Pre-computing Meta Feature Matrices...")

        self.user_feat_matrix = {}
        for f in self.selected_user_features:
            vocab_map = self.user_vocab[f]
            first_val = list(vocab_map.values())[0]
            
            if hasattr(first_val, '__len__') and not isinstance(first_val, str):
                dim = len(first_val)
            else:
                dim = 1
                
            matrix = np.zeros((len(self.users) + 2, dim), dtype=np.float32)

            for uid in self.users:
                eid = self.user_id_vocab[uid]
                raw_val = user_meta_dict[uid][f]
                matrix[eid] = vocab_map[raw_val]
                
            self.user_feat_matrix[f'uf_{f}'] = matrix

        self.item_feat_matrix = {}
        for f in self.selected_item_features:
            vocab_map = self.item_vocab[f]
            is_multihot = (f in multihot_cols)
            
            if is_multihot:
                sample_key = list(vocab_map.keys())[0]
                dim = len(vocab_map[sample_key])
            else:
                first_val = list(vocab_map.values())[0]
                if hasattr(first_val, '__len__') and not isinstance(first_val, str):
                    dim = len(first_val)
                else:
                    dim = 1
            
            matrix = np.zeros((len(self.items) + 2, dim), dtype=np.float32)

            for iid in self.items:
                eid = self.item_id_vocab[iid]
                raw_val = item_meta_dict[iid][f]
                
                if is_multihot:
                    vecs = [vocab_map[v] for v in str(raw_val).split(',')]
                    matrix[eid] = np.sum(vecs, axis=0)
                else:
                    matrix[eid] = vocab_map[raw_val]
                    
            self.item_feat_matrix[f'if_{f}'] = matrix

        del user_meta_dict, item_meta_dict, user_meta_df, item_meta_df

        print("Converting Log Data to Numpy Arrays for fast access...")
        self.log_numpy = {}
        
        raw_uids = self.log_data['user_id'].values
        raw_iids = self.log_data['video_id'].values
        
        self.log_numpy['user_id_enc'] = np.array([self.user_id_vocab[u] for u in raw_uids], dtype=np.int32)
        self.log_numpy['item_id_enc'] = np.array([self.item_id_vocab[i] for i in raw_iids], dtype=np.int32)
        self.log_numpy['user_id_raw'] = raw_uids 

        for resp in self.response_list:
            if resp in self.log_data.columns:
                self.log_numpy[resp] = self.log_data[resp].values.astype(np.float32)
            else:
                self.log_numpy[resp] = np.zeros(len(self.log_data), dtype=np.float32)

        self.data = self._sequence_holdout(args)

    def get_user_meta_data(self, user_id_encoded):
        return {k: self.user_feat_matrix[k][user_id_encoded] for k in self.user_feat_matrix}

    def get_item_meta_data(self, item_id_encoded):
        return {k: self.item_feat_matrix[k][item_id_encoded] for k in self.item_feat_matrix}

    def get_user_history(self, H_rowIDs):
        '''
 []  Numpy Indexing
 '''
        L = len(H_rowIDs)

        if L == 0:
            item_ids_padded = [0] * self.max_hist_seq_len
            indices = [0] * self.max_hist_seq_len
            
            hist_meta = {k: matrix[indices] for k, matrix in self.item_feat_matrix.items()}
            history_response = {resp: np.zeros(self.max_hist_seq_len, dtype=np.float32)\
                                for resp in self.response_list}
        else:
            raw_item_ids_enc = self.log_numpy['item_id_enc'][H_rowIDs]
            item_ids_padded = padding_and_clip(raw_item_ids_enc.tolist(), self.max_hist_seq_len)

            hist_meta = {k: matrix[item_ids_padded] for k, matrix in self.item_feat_matrix.items()}

            history_response = {}
            for resp in self.response_list:
                all_vals = self.log_numpy[resp][H_rowIDs]
                if len(all_vals) > self.max_hist_seq_len:
                    valid_vals = all_vals[-self.max_hist_seq_len:]
                else:
                    valid_vals = all_vals
                
                arr = np.zeros(self.max_hist_seq_len, dtype=np.float32)
                if len(valid_vals) > 0:
                    arr[self.max_hist_seq_len - len(valid_vals):] = valid_vals
                
                history_response[resp] = arr

        return item_ids_padded, L, hist_meta, history_response

    def __getitem__(self, idx):
        row_id = self.data[self.phase][idx]
        
        user_id_enc = self.log_numpy['user_id_enc'][row_id]
        item_id_enc = self.log_numpy['item_id_enc'][row_id]
        user_id_raw = self.log_numpy['user_id_raw'][row_id]

        record = {
            'user_id': user_id_enc,
            'item_id': item_id_enc,
        }

        for f in self.response_list:
            record[f] = self.log_numpy[f][row_id]

        loss_weight = np.array([1. if record[f] == 1\
                                    else -self.response_neg_sample_rate[f] if f == 'is_hate'\
            else self.response_neg_sample_rate[f]\
                                for i, f in enumerate(self.response_list)])
        record["loss_weight"] = loss_weight

        user_meta = self.get_user_meta_data(user_id_enc)
        record.update(user_meta)

        item_meta = self.get_item_meta_data(item_id_enc)
        record.update(item_meta)

        full_hist_indices = self.user_history[user_id_raw]
        
        try:
             current_idx_in_hist = full_hist_indices.index(row_id)
             
             start_idx = max(0, current_idx_in_hist - self.max_hist_seq_len)
             H_rowIDs = full_hist_indices[start_idx : current_idx_in_hist]
        except:
             H_rowIDs = []

        history, hist_length, hist_meta, hist_response = self.get_user_history(H_rowIDs)

        record['history'] = np.array(history)
        record['history_length'] = hist_length
        for f, v in hist_meta.items():
            record[f'history_{f}'] = v

        for f, v in hist_response.items():
            record[f'history_{f}'] = v

        return record