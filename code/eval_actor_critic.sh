#!/usr/bin/env bash
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
set -euo pipefail

CODE_DIR="${PROJECT_ROOT}/code"
cd "${CODE_DIR}"

UIRM_LOG_PATH="${UIRM_LOG_PATH:-${CODE_DIR}/output/Kuairand_Pure/env/log/user_KRMBUserResponse_lr0.0001_reg0_nlayer2.model.log}"
SAVE_PATH="${SAVE_PATH:-${CODE_DIR}/output/Kuairand_Pure/agents/A2C_OneStageHyperPolicy_with_DotScore_actor0.00001_critic0.001_niter20000_reg0.00001_ep0.01_noise0.1_bs128_epbs32_step20_seed2026_slatesize1/model}"

python3 eval_actor_critic.py \
    --env_class KREnvironment_WholeSession_GPU \
    --policy_class OneStageHyperPolicy_with_DotScore \
    --critic_class VCritic \
    --buffer_class HyperActorBuffer \
    --agent_class A2C \
    --uirm_log_path "${UIRM_LOG_PATH}" \
    --slate_size 1 \
    --episode_batch_size 32 \
    --item_correlation 0.2 \
    --max_step_per_episode 20 \
    --eval_episodes 1000 \
    --log_every 20 \
    --initial_temper 20 \
    --seed 2026 \
    --single_response \
    --save_path "${SAVE_PATH}" \
    --policy_action_hidden 256 64 \
    --policy_noise_var 0.1 \
    --state_user_latent_dim 16 \
    --state_item_latent_dim 16 \
    --state_transformer_enc_dim 32 \
    --state_transformer_n_head 4 \
    --state_transformer_d_forward 64 \
    --state_transformer_n_layer 3 \
    --state_dropout_rate 0.1
