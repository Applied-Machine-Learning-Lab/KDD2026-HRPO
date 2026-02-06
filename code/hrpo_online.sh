#!/usr/bin/env bash
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
set -euo pipefail


ROOT=${PROJECT_ROOT}
CODE=${ROOT}/code

export PYTHONPATH=${CODE}:${PYTHONPATH:-}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

SID_MAP=${SID_MAP:-${CODE}/dataset/kuairand/kuairand-Pure/sid/32_mask/video_sid_mapping.csv}
USER_FEAT=${USER_FEAT:-${ROOT}/dataset/kuairand/kuairand-Pure/data/user_features_Pure_fillna.csv}
BASE_LOG=${BASE_LOG:-${ROOT}/dataset/kuairand/kuairand-Pure/data/log_session_4_08_to_5_08_Pure.csv}

INIT_CKPT=${INIT_CKPT:-${CODE}/checkpoints/checkpoints/onerec_value_v2_32_mask_mini/epoch_5.pt}

HRPO_TABLE=${HRPO_TABLE:-${CODE}/dataset/kuairand/kuairand-Pure/hrpo/hrpo_table.pkl}

UIRM_LOG=${UIRM_LOG:-${CODE}/output/Kuairand_Pure/env/log/user_KRMBUserResponse_lr0.0001_reg0_nlayer2.model.log}

RUN_TAG=${RUN_TAG:-online_$(date +%Y%m%d_%H%M%S)}
OUT_DIR=${OUT_DIR:-${CODE}/checkpoints/checkpoints/${RUN_TAG}}
mkdir -p "${OUT_DIR}"

Q=${Q:-10}                  # number of online rounds
COLLECT_STEPS=${COLLECT_STEPS:-2000}
TRAIN_STEPS=${TRAIN_STEPS:-1000}
RESET_EVERY=${RESET_EVERY:-200}

SLATE_SIZE=${SLATE_SIZE:-1}
BEAM_WIDTH=${BEAM_WIDTH:-24}
TEMPERATURE=${TEMPERATURE:-1.0}

BATCH_SIZE=${BATCH_SIZE:-1024}
NUM_WORKERS=${NUM_WORKERS:-8}
LR=${LR:-1e-5}
WD=${WD:-0.01}
CLIP_EPS=${CLIP_EPS:-0.2}
KL_COEF=${KL_COEF:-0.1}
SFT_COEF=${SFT_COEF:-0.4}
SYNC_OLD_EVERY=${SYNC_OLD_EVERY:-200}

MAX_HIST_LEN_MODEL=${MAX_HIST_LEN_MODEL:-50}
MAX_HIST_LEN_TRAIN=${MAX_HIST_LEN_TRAIN:-50}
LABEL_COL=${LABEL_COL:-is_click}
GAMMA=${GAMMA:-0.99}

REWARD_WEIGHTS=${REWARD_WEIGHTS:-"is_click:1,long_view:0.5,is_like:0.2,is_comment:0.1,is_forward:0.0,is_follow:0.0,is_hate:0.0"}

EPISODE_BATCH_SIZE=${EPISODE_BATCH_SIZE:-32}
ITEM_CORR=${ITEM_CORR:-0}
MAX_STEP_PER_EPISODE=${MAX_STEP_PER_EPISODE:-20}
INITIAL_TEMPER=${INITIAL_TEMPER:-20}

DEVICE=${DEVICE:-cuda}
SEED=${SEED:-2026}

HRPO_ONLINE_PY=${HRPO_ONLINE_PY:-${CODE}/hrpo_online.py}

echo "[run] OUT_DIR=${OUT_DIR}"
echo "[run] HRPO_ONLINE_PY=${HRPO_ONLINE_PY}"

[[ -f "${HRPO_ONLINE_PY}" ]] || { echo "[error] hrpo_online.py not found: ${HRPO_ONLINE_PY}"; exit 1; }
[[ -f "${SID_MAP}" ]] || { echo "[error] SID_MAP not found: ${SID_MAP}"; exit 1; }
[[ -f "${USER_FEAT}" ]] || { echo "[error] USER_FEAT not found: ${USER_FEAT}"; exit 1; }
[[ -f "${BASE_LOG}" ]] || { echo "[error] BASE_LOG not found: ${BASE_LOG}"; exit 1; }
[[ -f "${INIT_CKPT}" ]] || { echo "[error] INIT_CKPT not found: ${INIT_CKPT}"; exit 1; }
[[ -f "${UIRM_LOG}" ]] || { echo "[error] UIRM_LOG not found: ${UIRM_LOG} (KuaiSim env requires it)"; exit 1; }

if [[ -n "${HRPO_TABLE}" ]] && [[ ! -f "${HRPO_TABLE}" ]]; then
  echo "[warn] HRPO_TABLE not found: ${HRPO_TABLE}; will start with an empty table."
  HRPO_TABLE=""
fi

python3 "${HRPO_ONLINE_PY}" \
  --model_size mini \
  --init_ckpt "${INIT_CKPT}" \
  --sid_mapping_path "${SID_MAP}" \
  --user_feat_path "${USER_FEAT}" \
  --base_log_paths "${BASE_LOG}" \
  --hrpo_table_path "${HRPO_TABLE}" \
  --output_dir "${OUT_DIR}" \
  --Q "${Q}" \
  --collect_steps "${COLLECT_STEPS}" \
  --train_steps "${TRAIN_STEPS}" \
  --reset_every "${RESET_EVERY}" \
  --beam_width "${BEAM_WIDTH}" \
  --temperature "${TEMPERATURE}" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --lr "${LR}" \
  --wd "${WD}" \
  --clip_eps "${CLIP_EPS}" \
  --kl_coef "${KL_COEF}" \
  --sft_coef "${SFT_COEF}" \
  --sync_old_every "${SYNC_OLD_EVERY}" \
  --label_col "${LABEL_COL}" \
  --gamma "${GAMMA}" \
  --max_hist_len_model "${MAX_HIST_LEN_MODEL}" \
  --max_hist_len_train "${MAX_HIST_LEN_TRAIN}" \
  --reward_weights "${REWARD_WEIGHTS}" \
  --device "${DEVICE}" \
  --seed "${SEED}" \
  --uirm_log_path "${UIRM_LOG}" \
  --slate_size "${SLATE_SIZE}" \
  --episode_batch_size "${EPISODE_BATCH_SIZE}" \
  --item_correlation "${ITEM_CORR}" \
  --max_step_per_episode "${MAX_STEP_PER_EPISODE}" \
  --initial_temper "${INITIAL_TEMPER}"
