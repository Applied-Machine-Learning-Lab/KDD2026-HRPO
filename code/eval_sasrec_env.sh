#!/bin/bash
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

ROOT_PATH="${PROJECT_ROOT}"

python "${ROOT_PATH}/code/eval_sasrec_env.py" \
  --sasrec_ckpt "${ROOT_PATH}/code/checkpoints/checkpoints/sasrec/best.pt" \
  --num_episodes 1000 \
  --uirm_log_path "${ROOT_PATH}/code/output/Kuairand_Pure/env/log/user_KRMBUserResponse_lr0.0001_reg0_nlayer2.model.log" \
  --slate_size 1 \
  --max_steps_per_episode 20 \
  --seed 2026 
