#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ROOT_PATH="${PROJECT_ROOT}"

MODEL_SIZE="${MODEL_SIZE:-mini}"  # mini | medium | large
TRAIN_VALUE="${TRAIN_VALUE:-0}"   # 1: train value head, 0: NTP-only setup

COMMON_ARGS=(
  --log_paths "${ROOT_PATH}/dataset/kuairand/kuairand-Pure/data/log_session_4_08_to_5_08_Pure.csv"
  --sid_mapping_path "${ROOT_PATH}/code/dataset/kuairand/kuairand-Pure/sid/32_mask/video_sid_mapping.csv"
  --user_feat_path "${ROOT_PATH}/dataset/kuairand/kuairand-Pure/data/user_features_Pure_fillna.csv"
  --model_dir "${ROOT_PATH}/code/checkpoints/checkpoints/onerec_value_v2_32_mask_${MODEL_SIZE}"
  --model_size "${MODEL_SIZE}"
  --max_hist_len 50
  --sid_depth 4
  --num_classes 32
  --lr 1e-4
  --gamma 0.99
  --epochs 5
  --beam_width 50
  --topk_list 1,5,10,20
  --use_user_token
  --value_token_weights "0,0,0,1"
)

if [[ "${TRAIN_VALUE}" == "1" ]]; then
  python "${ROOT_PATH}/code/train_onerec_value.py" \
    "${COMMON_ARGS[@]}" \
    --train_value \
    --w_rev 0.5 \
    --w_ltv 0.5 \
    --w_cls 1.0
else
  python "${ROOT_PATH}/code/train_onerec_value.py" \
    "${COMMON_ARGS[@]}" \
    --w_rev 0.0 \
    --w_ltv 0.0 \
    --w_cls 1.0
fi
