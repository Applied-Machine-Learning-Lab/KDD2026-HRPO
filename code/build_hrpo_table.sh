#!/usr/bin/env bash
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
set -euo pipefail


ROOT_PATH=${PROJECT_ROOT}

USE_UNBIASED="${USE_UNBIASED:-0}"
DATE_COL="${DATE_COL:-date}"
UNBIASED_START="${UNBIASED_START:-20220422}"
UNBIASED_END="${UNBIASED_END:-20220508}"

RTG="${RTG:-0}"
RTG_GAMMA="${RTG_GAMMA:-0.99}"
RTG_HORIZON="${RTG_HORIZON:-0}"
NO_COHORTING=0
NO_COHORTING="${NO_COHORTING:-0}"

DATA_DIR=${ROOT_PATH}/dataset/kuairand/kuairand-Pure/data
LOG_SESSION="${DATA_DIR}/log_session_4_08_to_5_08_Pure.csv"

SID_MAPPING_PATH=${ROOT_PATH}/code/dataset/kuairand/kuairand-Pure/sid/32_mask/video_sid_mapping.csv
USER_FEAT_PATH=${ROOT_PATH}/dataset/kuairand/kuairand-Pure/data/user_features_Pure_fillna.csv

CTX_MODE="${CTX_MODE:-bucket}"
KMEANS_K=384
MAX_CTX=512
MIN_CTX_USERS=20
MIN_PREFIX_COUNT=1
SMOOTHING_ALPHA=100
CHUNKSIZE=200000

TASTE_SID_COLS="${TASTE_SID_COLS:-sid_1,sid_2}"
TASTE_EVENT_COL="${TASTE_EVENT_COL:-is_click}"
TASTE_EVENT_MIN="${TASTE_EVENT_MIN:-1}"
TASTE_EMBED_METHOD="${TASTE_EMBED_METHOD:-svd}"   # raw|svd|nmf|lda
TASTE_DIM="${TASTE_DIM:-32}"
TASTE_MAX_HIST_LEN="${TASTE_MAX_HIST_LEN:-50}"

ONEREC_CKPT="${ONEREC_CKPT:-${ROOT_PATH}/code/checkpoints/checkpoints/onerec_value_v2_32_mask_mini/epoch_5.pt}"                     # REQUIRED for taste_onerec
ONEREC_DEVICE=cuda:0
ONEREC_CODE_ROOT="${ONEREC_CODE_ROOT:-${ROOT_PATH}/code}"
ONEREC_DEVICE="${ONEREC_DEVICE:-cpu}"              # e.g. cuda:0
ONEREC_BATCH_SIZE="${ONEREC_BATCH_SIZE:-1024}"
ONEREC_USE_USER_FEAT=1
ONEREC_CONCAT_USER_VEC=1
ONEREC_USE_USER_FEAT="${ONEREC_USE_USER_FEAT:-0}"  # 1 -> pass --onerec_use_user_feat
ONEREC_CONCAT_USER_VEC="${ONEREC_CONCAT_USER_VEC:-0}"  # 1 -> pass --onerec_concat_user_vec

EXTRA_ARGS=()
CTX_ARGS=()

case "${CTX_MODE}" in
  bucket|kmeans)
    ;;
  taste_hist)
    CTX_ARGS+=(--taste_sid_cols "${TASTE_SID_COLS}")
    CTX_ARGS+=(--taste_event_col "${TASTE_EVENT_COL}" --taste_event_min "${TASTE_EVENT_MIN}")
    CTX_ARGS+=(--taste_embed_method "${TASTE_EMBED_METHOD}" --taste_dim "${TASTE_DIM}")
    ;;
  taste_onerec)
    if [[ -z "${ONEREC_CKPT}" ]]; then
      echo "[HRPO][ERR] CTX_MODE=taste_onerec requires ONEREC_CKPT=/path/to/ckpt.pt"
      exit 1
    fi
    CTX_ARGS+=(--taste_event_col "${TASTE_EVENT_COL}" --taste_event_min "${TASTE_EVENT_MIN}")
    CTX_ARGS+=(--taste_max_hist_len "${TASTE_MAX_HIST_LEN}")
    CTX_ARGS+=(--onerec_ckpt "${ONEREC_CKPT}" --onerec_code_root "${ONEREC_CODE_ROOT}")
    CTX_ARGS+=(--onerec_device "${ONEREC_DEVICE}" --onerec_batch_size "${ONEREC_BATCH_SIZE}")
    if [[ "${ONEREC_USE_USER_FEAT}" == "1" ]]; then
      CTX_ARGS+=(--onerec_use_user_feat)
    fi
    if [[ "${ONEREC_CONCAT_USER_VEC}" == "1" ]]; then
      CTX_ARGS+=(--onerec_concat_user_vec)
    fi
    ;;
  *)
    echo "[HRPO][ERR] Unknown CTX_MODE='${CTX_MODE}'. Supported: bucket | kmeans | taste_hist | taste_onerec"
    exit 1
    ;;
esac

OUT_SUFFIX=""

if [[ "${USE_UNBIASED}" == "1" ]]; then
  echo "[HRPO][CFG] USE_UNBIASED=1 -> slicing ${DATE_COL} in [${UNBIASED_START}, ${UNBIASED_END}] from ${LOG_SESSION}"
  EXTRA_ARGS+=(--use_unbiased --date_col "${DATE_COL}" --unbiased_start "${UNBIASED_START}" --unbiased_end "${UNBIASED_END}")
  OUT_SUFFIX="_unbiased"
else
  echo "[HRPO][CFG] USE_UNBIASED=0 -> using ALL rows from ${LOG_SESSION}"
fi

if [[ "${RTG}" == "1" ]]; then
  echo "[HRPO][CFG] RTG=1 -> target=RTG (gamma=${RTG_GAMMA}, horizon=${RTG_HORIZON})"
  EXTRA_ARGS+=(--use_rtg --rtg_gamma "${RTG_GAMMA}" --rtg_horizon "${RTG_HORIZON}")
  OUT_SUFFIX="${OUT_SUFFIX}_rtg_g${RTG_GAMMA}_h${RTG_HORIZON}"
else
  echo "[HRPO][CFG] RTG=0 -> target=single_step_reward"
fi

if [[ "${NO_COHORTING}" == "1" ]]; then
  echo "[HRPO][CFG] NO_COHORTING=1 -> disable cohort conditioning (ctx_id=0 for all users)"
  EXTRA_ARGS+=(--no_cohorting)
  OUT_SUFFIX="${OUT_SUFFIX}"
fi

OUT_DIR=${ROOT_PATH}/code/dataset/kuairand/kuairand-Pure/hrpo_${CTX_MODE}${OUT_SUFFIX}
mkdir -p "${OUT_DIR}"

BEHAVIOR_NAMES=(click long_view like comment forward follow hate)
BEHAVIOR_COLS=(is_click long_view is_like is_comment is_forward is_follow is_hate)

for i in "${!BEHAVIOR_NAMES[@]}"; do
  name="${BEHAVIOR_NAMES[$i]}"
  col="${BEHAVIOR_COLS[$i]}"
  out="${OUT_DIR}/hrpo_${name}.pkl"

  echo "[HRPO][BUILD] ${name} (${col}) -> ${out}"

  python "${PROJECT_ROOT}/code/build_hrpo_table.py" \
    --log_paths "${LOG_SESSION}" \
    --sid_mapping_path "${SID_MAPPING_PATH}" \
    --user_feat_path "${USER_FEAT_PATH}" \
    --reward_weights "{\"${col}\": 1.0}" \
    --ctx_mode "${CTX_MODE}" \
    --max_ctx "${MAX_CTX}" \
    --min_ctx_users "${MIN_CTX_USERS}" \
    --min_prefix_count "${MIN_PREFIX_COUNT}" \
    --smoothing_alpha "${SMOOTHING_ALPHA}" \
    --chunksize "${CHUNKSIZE}" \
    --kmeans_k "${KMEANS_K}" \
    "${CTX_ARGS[@]-}" \
    "${EXTRA_ARGS[@]-}" \
    --out_path "${out}"
done

echo "[HRPO][DONE] all behavior tables saved under: ${OUT_DIR}"
