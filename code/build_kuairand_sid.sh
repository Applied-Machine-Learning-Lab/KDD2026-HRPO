#!/usr/bin/env bash
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

python "${PROJECT_ROOT}/code/build_pure_sid.py" \
  --log-session "${PROJECT_ROOT}/dataset/kuairand/kuairand-Pure/data/log_session_4_08_to_5_08_Pure.csv" \
  --output-dir "${PROJECT_ROOT}/dataset/kuairand/kuairand-Pure/sid/64_mask" \
  --n-layers 4 --codebook-size 64 --max-tag 500
