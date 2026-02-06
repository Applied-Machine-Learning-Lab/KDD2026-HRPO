#!/usr/bin/env bash
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
set -e

ROOT="${PROJECT_ROOT}"
CODE="${ROOT}/code"
DATA="${ROOT}/dataset/kuairand/kuairand-Pure"

CSV_PATH=${CSV_PATH:-${DATA}/data/log_session_4_08_to_5_08_Pure.csv}
SAVE_DIR=${SAVE_DIR:-${CODE}/checkpoints/checkpoints/gru4rec_pure}
CACHE_DIR=${CACHE_DIR:-${CODE}/cache_gru4rec}
DEVICE=${DEVICE:-cuda:0}

python "${CODE}/train_gru4rec_baseline.py" \
  --csv_path "${CSV_PATH}" \
  --save_dir "${SAVE_DIR}" \
  --cache_dir "${CACHE_DIR}" \
  --device "${DEVICE}" \
  --train_mode session \
  --loss bpr-max \
  --neg_mode inbatch \
  --batch_size 1024 \
  --epochs 80 \
  --optim adagrad \
  --lr 5e-2 \
  --dropout 0.2 \
  --d_model 128 \
  --n_layers 1 \
  --final_act elu \
  --bprmax_reg 0.0 \
  --early_stop_patience 20 \
  --grad_clip 1.0 \
  --seed 2025 \
  --num_workers 4 \
  --save_metric_k 10
