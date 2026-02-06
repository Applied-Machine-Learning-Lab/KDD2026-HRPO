import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from model.general import BaseModel

class DecisionTransformerRec(BaseModel):
    '''
 Decision Transformer (Final Fixed Version)
  Reader output One-Hot user features,  expand . 
 '''

    @staticmethod
    def parse_model_args(parser):
        parser = BaseModel.parse_model_args(parser)
        parser.add_argument('--n_head', type=int, default=4, help='Number of attention heads')
        parser.add_argument('--n_layer', type=int, default=3, help='Number of transformer layers')
        parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension')
        parser.add_argument('--max_timestep', type=int, default=500, help='Max timestep capacity')
        parser.add_argument('--dropout_rate', type=float, default=0.1, help='Dropout rate')
        parser.add_argument('--rtg_scale', type=float, default=20.0, help='rtg_scale')
        return parser

    def __init__(self, args, reader_stats, device):
        super().__init__(args, reader_stats, device)
        
        self.hidden_dim = args.hidden_dim
        self.max_len = args.max_hist_seq_len
        self.n_users = reader_stats['n_user']
        self.n_items = reader_stats['n_item']
        self.device = device
        self.n_head = args.n_head
        
        self.feedback_types = [] 
        
        self.item_emb = nn.Embedding(self.n_items + 1, self.hidden_dim, padding_idx=0)
        self.user_emb = nn.Embedding(self.n_users + 1, self.hidden_dim, padding_idx=0)
        
        self.user_feat_emb = nn.ModuleDict()
        if 'user_feature_dims' in reader_stats:
            print(f"[Model Init] Loading {len(reader_stats['user_feature_dims'])} extra user features (Linear Projection)...")
            for feat_name, feat_dim in reader_stats['user_feature_dims'].items():
                self.user_feat_emb[feat_name] = nn.Linear(feat_dim, self.hidden_dim)
        
        self.ret_emb = nn.Sequential(nn.Linear(1, self.hidden_dim), nn.Tanh())
        self.time_emb = nn.Embedding(args.max_timestep, self.hidden_dim)
        
        self.emb_norm = nn.LayerNorm(self.hidden_dim)
        self.dropout = nn.Dropout(args.dropout_rate)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=args.n_head,
            dim_feedforward=self.hidden_dim * 4,
            dropout=args.dropout_rate,
            batch_first=True,
            norm_first=True 
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=args.n_layer)
        
        self.predict_head = nn.Linear(self.hidden_dim, self.n_items + 1)
        
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def do_forward_and_loss(self, feed_dict):
        out_dict = self.forward(feed_dict)
        loss, _ = self.get_loss(feed_dict, out_dict)
        out_dict['loss'] = loss
        return out_dict

    def forward(self, feed_dict):
        item_seq = feed_dict['input_items']          # (B, L)
        rtg_seq = feed_dict['input_rtgs'].unsqueeze(-1) # (B, L, 1)
        time_seq = feed_dict['input_timesteps']      # (B, L)
        user_ids = feed_dict['user_id']              # (B,)
        attn_mask = feed_dict['attention_mask']      # (B, L)
        
        B, L = item_seq.shape
        device = item_seq.device
        
        zero_pad = torch.zeros((B, 1), dtype=torch.long, device=device)
        input_actions = torch.cat([zero_pad, item_seq[:, :-1]], dim=1)
        a_emb = self.item_emb(input_actions)
        
        s_emb = self.user_emb(user_ids).unsqueeze(1).expand(-1, L, -1)
        
        for feat_name, proj_layer in self.user_feat_emb.items():
            key = f"uf_{feat_name}"
            if key in feed_dict:
                feat_val = feed_dict[key].float()
                
                feat_vec = proj_layer(feat_val)
                
                s_emb = s_emb + feat_vec.unsqueeze(1).expand(-1, L, -1)
        
        r_emb = self.ret_emb(rtg_seq)
        t_emb = self.time_emb(torch.clamp(time_seq, 0, self.time_emb.num_embeddings - 1))
        
        x = r_emb + s_emb + a_emb + t_emb
        x = self.emb_norm(x)
        x = self.dropout(x)
        
        causal_mask = torch.triu(torch.ones(L, L, device=device) * float('-inf'), diagonal=1)
        
        padding_mask_val = torch.zeros((B, L), device=device)
        padding_mask_val.masked_fill_(attn_mask == 0, float('-inf'))
        padding_mask_broadcast = padding_mask_val.unsqueeze(1).unsqueeze(1)
        
        combined_mask = causal_mask.unsqueeze(0).unsqueeze(0) + padding_mask_broadcast
        
        is_pad_query = (attn_mask == 0) 
        is_pad_query_expanded = is_pad_query.unsqueeze(1).unsqueeze(-1)
        combined_mask = combined_mask.masked_fill(is_pad_query_expanded, 0.0)
        
        combined_mask = combined_mask.repeat_interleave(self.n_head, dim=0)
        combined_mask = combined_mask.squeeze(1)
        
        out = self.transformer(x, mask=combined_mask)
        
        logits = self.predict_head(out)
        last_pred = logits[:, -1, :] 
        
        return {'logits': logits, 'preds': last_pred}

    def get_loss(self, feed_dict, out_dict):
        logits = out_dict['logits']          # (B, L, V)
        targets = feed_dict['input_items']   # (B, L)
        mask = feed_dict['attention_mask']   # (B, L)
    
        B, L = targets.shape
    
        logits = logits.view(-1, self.n_items + 1)   # (B*L, V)
        targets = targets.view(-1).long()            # (B*L,)
        mask = mask.view(-1)                         # (B*L,)
    
        loss_element = F.cross_entropy(logits, targets, reduction='none')  # (B*L,)
    
        weight_mask = mask
    
        if 'sample_weight' in feed_dict:
            ep_w = feed_dict['sample_weight'].view(B)           # (B,)
            ep_w_seq = ep_w.unsqueeze(1).expand(-1, L).reshape(-1)  # (B*L,)
            weight_mask = weight_mask * ep_w_seq
    
        sum_weight = weight_mask.sum()
        if sum_weight < 1e-6:
            return torch.tensor(0.0, device=logits.device, requires_grad=True), {}
    
        loss = (loss_element * weight_mask).sum() / sum_weight
    
        return loss, {}

