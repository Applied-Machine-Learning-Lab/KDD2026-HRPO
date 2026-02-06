#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ROOT_PATH="${PROJECT_ROOT}"
MODEL_SIZE="${MODEL_SIZE:-mini}"  # mini | medium | large
ONEREC_CKPT="${ONEREC_CKPT:-${ROOT_PATH}/code/checkpoints/checkpoints/onerec_value_v2_32_mask_${MODEL_SIZE}/hrpo_rrpo_ntp/epoch_1.pt}"

python "${ROOT_PATH}/code/eval_onerec_value_rerank.py" \
  --onerec_ckpt "${ONEREC_CKPT}" \
  --num_episodes 1000 \
  --uirm_log_path "${ROOT_PATH}/code/output/Kuairand_Pure/env/log/user_KRMBUserResponse_lr0.0001_reg0_nlayer2.model.log" \
  --sid_mapping_path "${ROOT_PATH}/code/dataset/kuairand/kuairand-Pure/sid/32_mask/video_sid_mapping.csv" \
  --slate_size 1 \
  --sid_depth 4 \
  --num_classes 32 \
  --max_step_per_episode 20 \
  --beam_width 64 \
  --rerank_alpha 0.0 \
  --rerank_formula add \
  --seed 2026 \
  --single_response \
  --report_debias_metrics
