import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleTransformerEncoder(nn.Module):
    """
    Wrapper of nn.TransformerEncoder with batch_first=True.
    Input: src [B, S, D]
    """
    def __init__(self, d_model, nhead, num_layers, dim_feedforward=None, dropout=0.1):
        super().__init__()
        dim_feedforward = dim_feedforward or (4 * d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.net = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, src, src_key_padding_mask=None):
        return self.net(src, src_key_padding_mask=src_key_padding_mask)


class SimpleTransformerDecoder(nn.Module):
    """
    Wrapper of nn.TransformerDecoder with batch_first=True.
    Provides forward(tgt, tgt_mask, memory, memory_key_padding_mask).
    """
    def __init__(self, d_model, nhead, num_layers, dim_feedforward=None, dropout=0.1):
        super().__init__()
        dim_feedforward = dim_feedforward or (4 * d_model)
        layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.net = nn.TransformerDecoder(layer, num_layers=num_layers)

    def forward(self, tgt, tgt_mask=None, memory=None, memory_key_padding_mask=None):
        return self.net(tgt, tgt_mask=tgt_mask, memory=memory, memory_key_padding_mask=memory_key_padding_mask)


class OneRecSIDWithContext(nn.Module):
    """
  OneRec backbone:
 - history encoder: encode sequence of item vectors ( item  sid_depth  embedding mean)
 - decoder: masked self-attn + cross-attn (attend to enc_out)
 - supports _build_decoder_input (shift-right + BOS), _build_causal_mask
 API ( onerec_value.py):
 - sid_embedding
 - pos_embedding
 - bos (Parameter) used to replace first position in decoder input
 - hist_encoder(src, src_key_padding_mask)
 - decoder(tgt=..., tgt_mask=..., memory=..., memory_key_padding_mask=...)
 - out_proj(dec_feats) -> logits
 - user_proj, ctx_proj
 - encode_history_seq(hist_sid, hist_len) can override
 """
    def __init__(
        self,
        num_decoder_block: int,
        hid_dim: int,
        nhead: int,
        sid_depth: int,
        num_classes: int,
        user_feat_dim: int,
        max_hist_len: int,
        hist_num_layers: int = 1,
        dropout_ratio: float = 0.1,
        **kwargs
    ):
        super().__init__()

        self.hid_dim = hid_dim
        self.nhead = nhead
        self.sid_depth = sid_depth
        self.max_hist_len = max_hist_len
        self.num_classes = num_classes

        self.sid_embedding = nn.Embedding(num_classes, hid_dim)

        self.sid_pos_embedding = nn.Embedding(sid_depth, hid_dim)

        self.sid_merge = nn.Linear(hid_dim * sid_depth, hid_dim)

        self.hist_pos_embedding = nn.Embedding(max_hist_len, hid_dim)

        max_pos = max(256, max_hist_len * sid_depth + sid_depth + 16)
        self.pos_embedding = nn.Embedding(max_pos, hid_dim)

        self.bos = nn.Parameter(torch.zeros(hid_dim))

        self.hist_encoder = SimpleTransformerEncoder(d_model=hid_dim, nhead=nhead,
                                                     num_layers=hist_num_layers, dropout=dropout_ratio)

        self.decoder = SimpleTransformerDecoder(d_model=hid_dim, nhead=nhead,
                                                num_layers=num_decoder_block, dropout=dropout_ratio)

        self.encoder = self.decoder

        self.out_proj = nn.Linear(hid_dim, num_classes)

        self.user_proj = nn.Linear(user_feat_dim, hid_dim)
        self.ctx_proj = nn.Linear(hid_dim * 2, hid_dim)  # concat(hist_vec, user_vec) -> hid_dim

        self.training_mode = kwargs.get("training_mode", "teacher_force")
        self.sample_cnt = kwargs.get("sample_cnt", 1000000)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.normal_(self.sid_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_embedding.weight, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)
        nn.init.xavier_uniform_(self.user_proj.weight)
        nn.init.constant_(self.user_proj.bias, 0.0)
        nn.init.xavier_uniform_(self.ctx_proj.weight)
        nn.init.constant_(self.ctx_proj.bias, 0.0)
        nn.init.normal_(self.bos, mean=0.0, std=0.02)

    def encode_history_seq(self, hist_sid, hist_len=None, user_ctx: torch.Tensor = None):
        """
        Encode history sequence (optionally fused with user vector):

          - hist_sid: [B, H, L], L = sid_depth (PAD item should be all-zero SID, i.e. [0,0,0,0])
          - hist_len: [B] or None. **Not required** if PAD items are all-zero.
          - user_ctx: [B, D] or None (already projected to hid_dim)

        Returns:
          - enc_out: [B, H, D]  (decoder cross-attn memory)
          - mask:    [B, H]     (bool, True = padding position)
        """
        B, H, L = hist_sid.shape
        device = hist_sid.device

        emb = self.sid_embedding(hist_sid)  # [B,H,L,D]

        depth_pos = torch.arange(L, device=device).view(1, 1, L)
        emb = emb + self.sid_pos_embedding(depth_pos)  # broadcast -> [B,H,L,D]

        item_vec = self.sid_merge(emb.reshape(B, H, L * self.hid_dim))  # [B,H,D]

        hist_pos = torch.arange(H, device=device).view(1, H)
        item_vec = item_vec + self.hist_pos_embedding(hist_pos)  # [B,H,D]

        mask = (hist_sid == 0).all(dim=2)  # [B,H], True = pad

        if (not mask.any()) and (hist_len is not None):
            hist_len = hist_len.to(device).long().clamp(min=0, max=H)
            idx = torch.arange(H, device=device).unsqueeze(0)  # [1,H]
            mask = idx >= hist_len.unsqueeze(1)

        all_pad = mask.all(dim=1)
        if all_pad.any():
            mask[all_pad, -1] = False

        if user_ctx is not None:
            if user_ctx.dim() != 2 or user_ctx.size(0) != B or user_ctx.size(1) != self.hid_dim:
                raise ValueError(f"user_ctx shape expected [B, {self.hid_dim}], got {tuple(user_ctx.shape)}")
            item_vec = item_vec + user_ctx.unsqueeze(1)  # [B,H,D]

        enc_out = self.hist_encoder(src=item_vec, src_key_padding_mask=mask)  # [B,H,D]
        return enc_out, mask



    def _build_decoder_input(self, target_sid: torch.LongTensor):
        """
        Build decoder input embeddings (shift-right):
          - target_sid: [B, L] (the ground-truth tokens we want to predict)
        Returns:
          - x: [B, L, D] where
              x[:, 0] = BOS
              x[:, i] = embed(target_sid[:, i-1]) for i >= 1
          - plus positional embeddings already added (so x = token_emb + pos_emb)
        """
        B, L = target_sid.shape
        device = target_sid.device
        inp = torch.zeros_like(target_sid, device=device)
        if L > 1:
            inp[:, 1:] = target_sid[:, :-1]
        emb = self.sid_embedding(inp)  # [B, L, D]
        emb[:, 0, :] = self.bos
        pos_idx = torch.arange(L, device=device).unsqueeze(0)  # [1,L]
        pos = self.pos_embedding(pos_idx)  # [1,L,D]
        return emb + pos

    def _build_causal_mask(self, L: int, device):
        """
        Return broadcastable tgt_mask for nn.TransformerDecoder:
          shape (L, L), float mask with 0.0 for allowed and -inf for masked (future) positions.
        """
        if L <= 0:
            return None
        attn_shape = (L, L)
        subsequent = torch.triu(torch.ones(attn_shape, device=device, dtype=torch.bool), diagonal=1)
        mask = torch.full(attn_shape, fill_value=float("-inf"), device=device)
        mask = mask.masked_fill(~subsequent, 0.0)
        return mask  # shape [L, L], dtype float

    def decode(self, tgt_embedded, tgt_mask, memory, memory_key_padding_mask):
        """
        tgt_embedded: [B, T, D], tgt_mask: [T, T] float, memory: [B, S, D], memory_key_padding_mask: [B, S] bool
        returns: dec_feats [B, T, D]
        """
        return self.decoder(tgt=tgt_embedded, tgt_mask=tgt_mask, memory=memory, memory_key_padding_mask=memory_key_padding_mask)
