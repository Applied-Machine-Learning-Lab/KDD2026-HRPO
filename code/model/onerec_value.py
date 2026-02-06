# -*- coding: utf-8 -*-
"""
OneRecWithValue = OneRecSIDWithContext + ValueDecoder.

- NTP loss: predicts each SID token (teacher forcing).
- Value head: predicts per-token immediate reward + long-term value (LTV).
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.onerec import OneRecSIDWithContext
from model.value_decoder import ValueDecoder


class OneRecWithValue(OneRecSIDWithContext):
    def __init__(
        self,
        num_decoder_block: int,
        hid_dim: int,
        nhead: int,
        sid_depth: int,
        num_classes: int,
        user_feat_dim: int,
        max_hist_len: int,
        value_layers: int = 2,
        detach_value_dec_feats: bool = True,
        **kwargs,
    ):
        super().__init__(
            num_decoder_block=num_decoder_block,
            hid_dim=hid_dim,
            nhead=nhead,
            sid_depth=sid_depth,
            num_classes=num_classes,
            user_feat_dim=user_feat_dim,
            max_hist_len=max_hist_len,
            **kwargs,
        )
        self.use_decoder_ctx = bool(kwargs.get('use_decoder_ctx', False))
        self.use_user_token = bool(kwargs.get('use_user_token', False))
        self.value_decoder = ValueDecoder(
            d_model=hid_dim,
            nhead=nhead,
            num_layers=value_layers,
            sid_depth=sid_depth,
            dropout=kwargs.get("dropout_ratio", 0.1),
            detach_dec_feats=detach_value_dec_feats,
        )

    @staticmethod
    def _pad_or_trim_user_feat(user_feat, target_dim: int, batch_size: Optional[int] = None) -> torch.Tensor:
        """Pad/trim user feature tensor to target_dim.
    
        Accepts: torch.Tensor / np.ndarray / list-like.
        Returns: float32 tensor [B,target_dim].
        """
        if user_feat is None:
            B = int(batch_size) if batch_size is not None else 1
            return torch.zeros((B, target_dim), dtype=torch.float32)
    
        if isinstance(user_feat, np.ndarray):
            user_feat = torch.from_numpy(user_feat)
        elif not torch.is_tensor(user_feat):
            user_feat = torch.tensor(user_feat)
    
        user_feat = user_feat.float()
        if user_feat.dim() == 1:
            user_feat = user_feat.unsqueeze(0)
    
        B, F = user_feat.shape
        if batch_size is not None and B != int(batch_size):
            if B == 1:
                user_feat = user_feat.expand(int(batch_size), -1)
                B = int(batch_size)
    
        if F == target_dim:
            return user_feat
        if F > target_dim:
            return user_feat[:, :target_dim]
    
        pad = torch.zeros((B, target_dim - F), dtype=user_feat.dtype, device=user_feat.device)
        return torch.cat([user_feat, pad], dim=1)

    def forward_with_cache(
        self,
        target_sid: torch.LongTensor,
        user_feat: torch.Tensor,
        hist_sid: torch.LongTensor,
        hist_len: torch.LongTensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """
        Returns:
          logits: [B,L,V]
          dec_feats: [B,L,D]
          enc_out: [B,S,D]
          enc_mask: [B,S]
          ctx: [B,D] or None
          user_vec: [B,D]
        """
        user_feat = self._pad_or_trim_user_feat(user_feat, self.user_proj.in_features)
        user_vec = self.user_proj(user_feat)

        enc_out, enc_mask = self.encode_history_seq(hist_sid, hist_len, user_ctx=user_vec)

        x = self._build_decoder_input(target_sid)  # [B,L,D]
        ctx = None
        if getattr(self, 'use_decoder_ctx', False):
            ctx = self.build_ctx(enc_out, enc_mask, user_vec)
            x = x + ctx.unsqueeze(1)

        L = x.size(1)
        tgt_mask = self._build_causal_mask(L, device=x.device)
        dec_feats = self.decode(
            x,
            tgt_mask=tgt_mask,
            memory=enc_out,
            memory_key_padding_mask=enc_mask,
        )  # [B,L,D]
        logits = self.out_proj(dec_feats)  # [B,L,V]
        return logits, dec_feats, enc_out, enc_mask, ctx, user_vec

    def compute_loss_with_value(
        self,
        target_sid: torch.LongTensor,
        user_feat: torch.Tensor,
        hist_sid: torch.LongTensor,
        hist_len: torch.LongTensor,
        immediate_rewards: Optional[torch.Tensor] = None,  # [B, L] or None
        ltv: Optional[torch.Tensor] = None,               # [B] or [B,1] or None
        token_weights: Optional[torch.Tensor] = None,     # [L] or None
        w_cls: float = 1.0,
        w_rev: float = 1.0,
        w_ltv: float = 1.0,
        value_only: bool = False,
        label_smoothing: float = 0.0,
    ) -> Tuple[torch.Tensor, Dict[str, float], Dict[str, torch.Tensor]]:
        """
        Returns:
          total_loss, log_dict, aux_tensors
        """
        B, L = target_sid.shape
        device = target_sid.device

        logits, dec_feats, enc_out, enc_mask, _, _ = self.forward_with_cache(
            target_sid=target_sid,
            user_feat=user_feat,
            hist_sid=hist_sid,
            hist_len=hist_len,
        )

        cls_loss = torch.tensor(0.0, device=device)
        if not value_only and w_cls > 0:
            if label_smoothing > 0:
                logp = F.log_softmax(logits, dim=-1)
                nll = -logp.gather(-1, target_sid.unsqueeze(-1)).squeeze(-1)  # [B,L]
                smooth = -logp.mean(dim=-1)                                   # [B,L]
                cls_loss = ((1 - label_smoothing) * nll + label_smoothing * smooth).mean()
            else:
                cls_loss = F.cross_entropy(
                    logits.view(B * L, -1),
                    target_sid.view(B * L),
                    ignore_index=-1,
                )

        rev_loss = torch.tensor(0.0, device=device)
        ltv_loss = torch.tensor(0.0, device=device)

        if (w_rev > 0 and immediate_rewards is not None) or (w_ltv > 0 and ltv is not None):
            sequence_emb = enc_out  # [B,S,D]

            reward_pred, ltv_pred = self.value_decoder(
                sequence_emb=sequence_emb,
                dec_feats=dec_feats,
                enc_padding_mask=enc_mask,
            )

            if immediate_rewards is not None and w_rev > 0:
                tgt_r = immediate_rewards.to(device).float()
                if tgt_r.shape != reward_pred.shape:
                    raise ValueError(f"immediate_rewards must be [B,L], got {tuple(tgt_r.shape)}")
                if token_weights is not None:
                    w = token_weights.to(device).float().view(1, L)
                    rev_loss = ((reward_pred - tgt_r) ** 2 * w).sum() / (w.sum() * B + 1e-8)
                else:
                    rev_loss = F.mse_loss(reward_pred, tgt_r)

            if ltv is not None and w_ltv > 0:
                tgt_ltv = ltv.to(device).float().view(B, 1)
                if token_weights is not None:
                    w = token_weights.to(device).float().view(1, L)
                    pred = (ltv_pred * w).sum(dim=1, keepdim=True) / (w.sum() + 1e-8)  # [B,1]
                else:
                    pred = ltv_pred.mean(dim=1, keepdim=True)
                ltv_loss = F.mse_loss(pred, tgt_ltv)

        total = w_cls * cls_loss + w_rev * rev_loss + w_ltv * ltv_loss
        logs = {
            "loss": float(total.detach().cpu()),
            "cls": float(cls_loss.detach().cpu()),
            "rev": float(rev_loss.detach().cpu()),
            "ltv": float(ltv_loss.detach().cpu()),
        }
        aux = {
            "logits": logits,
        }
        return total, logs, aux
