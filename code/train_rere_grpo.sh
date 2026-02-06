#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ROOT="${PROJECT_ROOT}"
CODE="${ROOT}/code"
DATA="${ROOT}/dataset/kuairand/kuairand-Pure"

MODEL_SIZE="mini"
SID_VOCAB=32
SID_DEPTH=4

GT_SPEC="is_click:1,long_view:0.0,is_like:0.0,is_comment:0.0,is_follow:0.0,is_forward:0.0,is_hate:0,is_profile_enter:0"
GT_GATE="is_click"
HIST_FROM_GT=1

INIT_CKPT="${CODE}/checkpoints/checkpoints/onerec_value_v2_32_mask_${MODEL_SIZE}/epoch_5.pt"
OUT_DIR="${CODE}/checkpoints/checkpoints/onerec_value_v2_32_mask_${MODEL_SIZE}/rere_grpo_grpo"

python "${CODE}/train_rere_grpo.py" \
  --log_paths "${DATA}/data/log_session_4_08_to_5_08_Pure.csv" \
  --sid_mapping_path "${CODE}/dataset/kuairand/kuairand-Pure/sid/32_mask/video_sid_mapping.csv" \
  --user_feat_path "${DATA}/data/user_features_Pure_fillna.csv" \
  --label_col "is_click" \
  --gt_spec "${GT_SPEC}" \
  --gt_gate "${GT_GATE}" \
  --hist_from_gt ${HIST_FROM_GT} \
  --sid_depth ${SID_DEPTH} \
  --num_classes ${SID_VOCAB} \
  --model_size "${MODEL_SIZE}" \
  --batch_size 1024 \
  --skip_nohit 1 \
  --num_workers 8 \
  --lr 1e-5 \
  --weight_decay 0.01 \
  --grad_clip 1.0 \
  --epochs 5 \
  --policy_coef 0.5 \
  --group_size 16 \
  --w_rank 1.0 \
  --sft_coef 0.5 \
  --ref_update_mode ema \
  --ref_update_tau 0.05 \
  --ref_update_every 1 \
  --kl_coef 0.001 \
  --use_old_model 0 \
  --old_model_update_every 50 \
  --log_every 50 \
  --eval_every 50 \
  --eval_beam 50 \
  --topk "1,5,10,20,50" \
  --init_ckpt "${INIT_CKPT}" \
  --model_dir "${OUT_DIR}"
