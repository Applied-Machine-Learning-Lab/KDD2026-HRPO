#!/usr/bin/env bash
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
set -e

ROOT="${PROJECT_ROOT}"
CODE="${ROOT}/code"
DATA="${ROOT}/dataset/kuairand/kuairand-Pure"

CSV_PATH=${CSV_PATH:-${DATA}/data/log_session_4_08_to_5_08_Pure.csv}
SAVE_DIR=${SAVE_DIR:-${CODE}/checkpoints/checkpoints/sasrec}
CACHE_DIR=${CACHE_DIR:-${CODE}/cache_sasrec}

DEVICE=${DEVICE:-cuda:0}

python "${PROJECT_ROOT}/code/train_sasrec_baseline.py" \
  --csv_path "${CSV_PATH}" \
  --save_dir "${SAVE_DIR}" \
  --device "${DEVICE}" \
  --max_len 50 \
  --d_model 128 \
  --n_heads 4 \
  --n_layers 2 \
  --dropout 0.2 \
  --batch_size 2048 \
  --epochs 80 \
  --lr 1e-3 \
  --weight_decay 0.0 \
  --early_stop_patience 10 \
  --grad_clip 1.0 \
  --seed 2025 \
  --num_workers 4 \
  --save_metric_k 10
