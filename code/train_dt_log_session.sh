PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
python "${PROJECT_ROOT}/code/train_dt_log_session.py" \
  --seed 2026 \
  --cuda 0 \
  --epoch 50 \
  --batch_size 2048 \
  --val_batch_size 512 \
  --lr 1e-4 \
  --early_stop_patience 5 \
  \
  --train_file "${PROJECT_ROOT}/dataset/kuairand/kuairand-Pure/data/log_session_4_08_to_5_08_Pure.csv" \
  --user_meta_file "${PROJECT_ROOT}/dataset/kuairand/kuairand-Pure/data/user_features_Pure_fillna.csv" \
  --item_meta_file "${PROJECT_ROOT}/dataset/kuairand/kuairand-Pure/data/video_features_basic_Pure_fillna.csv" \
  --data_separator ',' \
  --meta_file_sep ',' \
  --max_hist_seq_len 50 \
  --n_worker 4 \
  --val_holdout_per_user 5 \
  --test_holdout_per_user 5 \
  \
  --hidden_dim 256 \
  --n_layer 3 \
  --n_head 4 \
  --dropout_rate 0.1 \
  --max_timestep 500 \
  --rtg_scale 20.0 \
  \
  --single_response \
  --model_path "${PROJECT_ROOT}/output/KuaiRand_Pure/env/DT_log_session_single_response.model" 
  
