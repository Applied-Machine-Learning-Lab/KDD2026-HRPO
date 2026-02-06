#!/usr/bin/env bash
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
set -e


ROOT=${PROJECT_ROOT}



NO_RESIDUAL_CREDIT=0
NO_RESIDUAL_CREDIT=${NO_RESIDUAL_CREDIT:-0}
EXTRA_ARGS=()
if [ "${NO_RESIDUAL_CREDIT}" = "1" ]; then
  EXTRA_ARGS+=(--no_residual_credit)
fi

CTX_MODE=bucket
ROOT_MAIN=${ROOT}
CODE=${ROOT_MAIN}/code
PKL_ROOT=${ROOT_MAIN}/code/dataset/kuairand/kuairand-Pure/hrpo_bucket


SID_MAP=${CODE}/dataset/kuairand/kuairand-Pure/sid/32_mask/video_sid_mapping.csv
USER_FEAT=${ROOT_MAIN}/dataset/kuairand/kuairand-Pure/data/user_features_Pure_fillna.csv
LOG_CSV=${ROOT_MAIN}/dataset/kuairand/kuairand-Pure/data/log_session_4_08_to_5_08_Pure.csv
HRPO_TABLE=${CODE}/dataset/kuairand/kuairand-Pure/hrpo/hrpo_table.pkl

INIT_CKPT=${CODE}/checkpoints/checkpoints/onerec_value_v2_32_mask_mini/epoch_5.pt

python3 "${CODE}/train_hrpo_rrpo_ntp.py" \
  --log_paths "${LOG_CSV}" \
  --sid_mapping_path "${SID_MAP}" \
  --user_feat_path "${USER_FEAT}" \
  --hrpo_table_path "${HRPO_TABLE}" \
  --init_ckpt "${INIT_CKPT}" \
  --model_size mini \
  --hrpo_table_paths \
    "${PKL_ROOT}/hrpo_click.pkl" "${PKL_ROOT}/hrpo_long_view.pkl" "${PKL_ROOT}/hrpo_like.pkl" "${PKL_ROOT}/hrpo_comment.pkl" "${PKL_ROOT}/hrpo_forward.pkl" "${PKL_ROOT}/hrpo_follow.pkl" "${PKL_ROOT}/hrpo_hate.pkl" \
  --reward_weights "1.0,0.7,0.5,0.5,0.5,0.5,0.0" \
  --sid_depth 4 --num_classes 32 \
  --max_hist_len 50 --max_hist_len_model 50 \
  --batch_size 1024 --num_workers 8 \
  --epochs 1 \
  --group_size 18 \
  --clip_eps 0.2 \
  --reward_scale 30.0 \
  --kl_coef 0.1 \
  --sft_coef 0.4 \
  --old_update_freq 20 \
  --freeze_value_decoder 1 \
  --lr 1e-5 --weight_decay 0.01 --grad_clip 1.0 \
  --plot_every 200 --plot_smooth 30 --clear_metrics 1 \
  --metrics_dir "${PROJECT_ROOT}/code/output" \
  --device cuda \
  --log_every 50 --eval_every 40 --eval_beam 50 --topk 1,5,10,20,50 \
  --model_dir "${CODE}/checkpoints/checkpoints/onerec_value_v2_32_mask_mini/hrpo_rrpo_ntp" "${EXTRA_ARGS[@]}"






