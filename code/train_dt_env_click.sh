#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

KR_FLAG="Pure"
ROOT_PATH="${PROJECT_ROOT}"
OUTPUT_PATH="${ROOT_PATH}/output/KuaiRand_${KR_FLAG}"

UIRM_LOG_PATH="${ROOT_PATH}/code/output/Kuairand_Pure/env/log/user_KRMBUserResponse_lr0.0001_reg0_nlayer2.model.log"
INIT_DT_MODEL="${ROOT_PATH}/output/KuaiRand_Pure/env/DT_log_session_single_response.model"
DT_ENV_MODEL_PATH="${OUTPUT_PATH}/env/dt_env_trained_click.model"
CUDA_ID="${CUDA_ID:-0}"

python "${ROOT_PATH}/code/train_dt_env_click.py" \
  --uirm_log_path "${UIRM_LOG_PATH}" \
  --slate_size 1 \
  --episode_batch_size 1024 \
  --item_correlation 0 \
  --single_response \
  --init_dt_model "${INIT_DT_MODEL}" \
  --max_step_per_episode 20 \
  --initial_temper 20.0 \
  --model_path "${DT_ENV_MODEL_PATH}" \
  --loss ce \
  --l2_coef 0.0 \
  --max_hist_seq_len 50 \
  --hidden_dim 256 \
  --n_layer 3 \
  --n_head 4 \
  --max_timestep 500 \
  --rtg_scale 20.0 \
  --seed 2026 \
  --lr 2e-5 \
  --weight_decay 0.0 \
  --epoch 80 \
  --cuda "${CUDA_ID}" \
  --collect_episodes_per_epoch 2000 \
  --collect_max_steps 20000 \
  --buffer_max_episodes 4000 \
  --train_batch_size 64 \
  --train_updates_per_epoch 100 \
  --eps_greedy 0.0 \
  --temperature 0.0 \
  --target_return 8.0 \
  --target_return_min 8.0 \
  --target_return_max 8.5 \
  --coverage_penalty_coef 0.00
