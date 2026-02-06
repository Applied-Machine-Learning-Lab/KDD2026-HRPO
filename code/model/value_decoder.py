from __future__ import annotations

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.scale


class ValueDecoderBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1, ffn_expansion: int = 4, cross_gate_init: float = 0.5):
        super().__init__()
        self.ln_self = RMSNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        self.ln_cross = RMSNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        self.ln_ffn = RMSNorm(d_model)
        hidden = ffn_expansion * d_model
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

        self.cross_gate = nn.Parameter(torch.tensor(float(cross_gate_init)))

    def forward(self, x: torch.Tensor, enc_mem: torch.Tensor, enc_key_padding_mask: torch.Tensor | None) -> torch.Tensor:
        B, Lq, _ = x.shape

        h = self.ln_self(x)
        attn_mask = torch.triu(torch.ones(Lq, Lq, device=x.device, dtype=torch.bool), diagonal=1)
        self_out, _ = self.self_attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + self_out

        h = self.ln_cross(x)
        cross_out, _ = self.cross_attn(h, enc_mem, enc_mem, key_padding_mask=enc_key_padding_mask, need_weights=False)
        x = x + torch.sigmoid(self.cross_gate) * cross_out

        h = self.ln_ffn(x)
        x = x + self.ffn(h)
        return x


class ValueDecoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        sid_depth: int,
        dropout: float = 0.1,
        detach_dec_feats: bool = True,
        use_enc_padding_mask: bool = True,
        max_q_len: int | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.sid_depth = sid_depth
        self.detach_dec_feats = bool(detach_dec_feats)
        self.use_enc_padding_mask = bool(use_enc_padding_mask)

        q_len = 2 + sid_depth
        if max_q_len is None:
            max_q_len = max(16, q_len)

        self.boc = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_emb = nn.Embedding(max_q_len, d_model)

        self.enc_norm = RMSNorm(d_model)
        self.blocks = nn.ModuleList(
            [ValueDecoderBlock(d_model, nhead, dropout=dropout) for _ in range(num_layers)]
        )

        self.head_norm = RMSNorm(d_model)
        self.head_mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.rev_head = nn.Linear(d_model, 1)
        self.ltv_head = nn.Linear(d_model, 1)

        nn.init.normal_(self.boc, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_emb.weight, mean=0.0, std=0.02)

    def forward(
        self,
        sequence_emb: torch.Tensor,     # [B,S,D]
        dec_feats: torch.Tensor,        # [B,L,D]
        enc_padding_mask: torch.Tensor | None = None,  # [B,S]
    ):
        if sequence_emb.dim() != 3 or dec_feats.dim() != 3:
            raise ValueError("sequence_emb and dec_feats must be 3D")
        B, S, D = sequence_emb.shape
        B2, L, D2 = dec_feats.shape
        if B2 != B or D2 != D:
            raise ValueError(f"ValueDecoder mismatch: enc={tuple(sequence_emb.shape)}, dec={tuple(dec_feats.shape)}")

        enc_mem = self.enc_norm(sequence_emb)

        if self.use_enc_padding_mask and enc_padding_mask is not None and enc_padding_mask.shape == (B, S):
            m = enc_padding_mask.to(torch.bool)
            valid = (~m).float().unsqueeze(-1)
            denom = valid.sum(dim=1).clamp(min=1.0)
            user_tok = (enc_mem * valid).sum(dim=1) / denom
        else:
            user_tok = enc_mem.mean(dim=1)
        user_tok = user_tok.unsqueeze(1)  # [B,1,D]

        if self.detach_dec_feats:
            dec_feats = dec_feats.detach()

        boc = self.boc.expand(B, -1, -1)
        x = torch.cat([boc, user_tok, dec_feats], dim=1)  # [B,2+L,D]

        q_len = x.size(1)
        if q_len > self.pos_emb.num_embeddings:
            raise ValueError(f"q_len={q_len} exceeds max_q_len={self.pos_emb.num_embeddings}")
        pos_ids = torch.arange(q_len, device=x.device).unsqueeze(0)
        x = x + self.pos_emb(pos_ids)

        key_padding_mask = enc_padding_mask.to(torch.bool) if (self.use_enc_padding_mask and enc_padding_mask is not None) else None
        for blk in self.blocks:
            x = blk(x, enc_mem, key_padding_mask)

        feat = x[:, 2:, :]  # [B,L,D]
        h = self.head_mlp(self.head_norm(feat))
        rev_pred = self.rev_head(h).squeeze(-1)
        ltv_pred = self.ltv_head(h).squeeze(-1)
        return rev_pred, ltv_pred
