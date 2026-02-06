#!/usr/bin/env bash
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
set -e


MODEL_SIZE="mini"
ROOT="${PROJECT_ROOT}"
CODE="${ROOT}/code"
DATA="${ROOT}/dataset/kuairand/kuairand-Pure"
INIT_CKPT="${CODE}/checkpoints/checkpoints/onerec_value_v2_32_mask_${MODEL_SIZE}/epoch_5.pt"
OUT_DIR="${CODE}/checkpoints/checkpoints/onerec_value_v2_32_mask_${MODEL_SIZE}/s_dpo"


python "${PROJECT_ROOT}/code/train_s_dpo.py" \
  --log_paths "${DATA}/data/log_session_4_08_to_5_08_Pure.csv" \
  --sid_mapping_path "${CODE}/dataset/kuairand/kuairand-Pure/sid/32_mask/video_sid_mapping.csv" \
  --user_feat_path "${DATA}/data/user_features_Pure_fillna.csv" \
  --label_col "is_click" \
  --gt_spec "is_click:1" \
  --gt_gate "is_click" \
  --hist_from_gt 0 \
  --model_size small \
  --num_layers 3 \
  --sid_depth 4 \
  --num_classes 32 \
  --init_ckpt "${INIT_CKPT}" \
  --num_neg 6 \
  --neg_beam 16 \
  --neg_pick rank_range \
  --neg_rank_low 2 \
  --neg_rank_high 12 \
  --beta 0.01 \
  --dpo_coef 0.05 \
  --sft_coef 0.5 \
  --dpo_last_n 2 \
  --use_old_model 1 \
  --old_model_update_every 1000 \
  --batch_size 2048 \
  --lr 5e-6 \
  --epochs 10 \
  --eval_beam 50 \
  --eval_every 50 \
  --model_dir "${OUT_DIR}"
