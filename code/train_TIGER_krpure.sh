PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MODEL_SIZE="mini"   # [mini,medium,large]
ROOT_PATH="${PROJECT_ROOT}"
DATA_PATH="${ROOT_PATH}/dataset/kuairand/kuairand-Pure/data"
SID_PATH="${ROOT_PATH}/code/dataset/kuairand/kuairand-Pure/sid/32_mask/video_sid_mapping.csv"

python "${ROOT_PATH}/code/train_TIGER_krpure.py" \
  --log_paths "${PROJECT_ROOT}/dataset/kuairand/kuairand-Pure/data/log_session_4_08_to_5_08_Pure.csv" \
  --sid_mapping_path "${SID_PATH}" \
  --max_hist_items 50 \
  --batch_size 64 \
  --infer_size 64 \
  --num_epochs 3 \
  --model_size ${MODEL_SIZE} \
  --lr 1e-4 \
  --save_path "${ROOT_PATH}/output/KuaiRand_Pure/env/tiger_sid_krpure_${MODEL_SIZE}.pth" \
  --log_path  "${ROOT_PATH}/output/KuaiRand_Pure/env/log/tiger_sid_krpure.log"
