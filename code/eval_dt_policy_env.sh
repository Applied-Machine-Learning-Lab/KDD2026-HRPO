PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ROOT_PATH="${PROJECT_ROOT}/code"
UIRM_LOG="${PROJECT_ROOT}/code/output/Kuairand_Pure/env/log/user_KRMBUserResponse_lr0.0001_reg0_nlayer2.model.log"
DT_MODEL="${PROJECT_ROOT}/output/KuaiRand_Pure/env/dt_env_trained_click.model"


python "${PROJECT_ROOT}/code/eval_dt_policy_env.py" \
  --uirm_log_path "${UIRM_LOG}" \
  --slate_size 1 \
  --episode_batch_size 32 \
  --item_correlation 0 \
  --max_step_per_episode 20 \
  --num_episodes 1000 \
  --device cuda:0 \
  --hidden_dim 256 \
  --n_layer 3 \
  --n_head 4 \
  --max_timestep 500 \
  --rtg_scale 20.0 \
  --max_hist_seq_len 50 \
  --dt_model_path "${DT_MODEL}" \
  --target_return 8.5 \
  --eps_greedy 0.00 \
  --temperature 0.0 \
  --debug_padding \
  --debug_topk 10 \
  --seed 2026 \
  --single_response 

