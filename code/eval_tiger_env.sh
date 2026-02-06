#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ROOT_PATH="${PROJECT_ROOT}"

SEED="${SEED:-2026}"
MODEL_SIZE="${MODEL_SIZE:-mini}"  # mini | medium | large
MAX_STEP="${MAX_STEP:-1}"
DEVICE="${DEVICE:-cuda:0}"

python "${ROOT_PATH}/code/eval_tiger_env.py" \
  --tiger_ckpt "${ROOT_PATH}/output/KuaiRand_Pure/env/tiger_sid_krpure_${MODEL_SIZE}.pth" \
  --sid_mapping_path "${ROOT_PATH}/code/dataset/kuairand/kuairand-Pure/sid/32_mask/video_sid_mapping.csv" \
  --uirm_log_path "${ROOT_PATH}/code/output/Kuairand_Pure/env/log/user_KRMBUserResponse_lr0.0001_reg0_nlayer2.model.log" \
  --slate_size 1 \
  --episode_batch_size 32 \
  --model_size "${MODEL_SIZE}" \
  --num_episodes 1000 \
  --max_steps_per_episode "${MAX_STEP}" \
  --max_step_per_episode "${MAX_STEP}" \
  --beam_width 128 \
  --single_response \
  --seed "${SEED}" \
  --max_hist_items 50 \
  --device "${DEVICE}"
