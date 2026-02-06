#!/usr/bin/env bash
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
set -e


MODEL_SIZE="mini"
ROOT="${PROJECT_ROOT}"
CODE="${ROOT}/code"
DATA="${ROOT}/dataset/kuairand/kuairand-Pure"
INIT_CKPT="${CODE}/checkpoints/checkpoints/onerec_value_v2_32_mask_${MODEL_SIZE}/epoch_5.pt"
OUT_DIR="${CODE}/checkpoints/checkpoints/onerec_value_v2_32_mask_${MODEL_SIZE}/sprec"

CUDA_VISIBLE_DEVICES=0 \
python "${PROJECT_ROOT}/code/train_sprec.py" \
  --log_paths "${DATA}/data/log_session_4_08_to_5_08_Pure.csv" \
  --sid_mapping_path "${CODE}/dataset/kuairand/kuairand-Pure/sid/32_mask/video_sid_mapping.csv" \
  --user_feat_path "${DATA}/data/user_features_Pure_fillna.csv" \
  --label_col is_click \
  --model_size "${MODEL_SIZE}" \
  --sid_depth 4 \
  --num_classes 32 \
  --max_hist_len 50 \
  --max_hist_len_model 50 \
  --min_hist_len 1 \
  --gt_spec "is_click:1" \
  --gt_gate "is_click" \
  --hist_from_gt 1 \
  --gamma 1.0 \
  --gt_weight_norm 1 \
  --init_ckpt "${INIT_CKPT}" \
  --num_neg 1 \
  --neg_beam 12 \
  --neg_pick rank_range \
  --beta 0.01 \
  --sft_coef 1.0 \
  --dpo_coef 0.1 \
  --dpo_last_n 2 \
  --sp_iters 2 \
  --sft_steps 0 \
  --dpo_steps 150 \
  --batch_size 2048 \
  --lr 5e-6 \
  --weight_decay 0.01 \
  --epochs 15 \
  --eval_beam 50 \
  --eval_every 50 \
  --log_every 50 \
  --model_dir "${OUT_DIR}"
