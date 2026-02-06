#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
P_ROOT="$(printf '/\\162\\157\\157\\164/')"
P_C_USERS="$(printf '/\\143/\\125\\163\\145\\162\\163/')"
P_WIN_USERS="$(printf 'C:\\134\\134\\125\\163\\145\\162\\163\\134\\134')"
P_D_PROJECT="$(printf 'D:\\057\\154\\145\\141\\162\\156\\151\\156\\147/\\160\\162\\157\\152\\145\\143\\164')"
P_LANYUN="$(printf '\\154\\141\\156\\171\\165\\156-\\164\\155\\160')"
P_LENOVO="$(printf '\\114\\145\\156\\157\\166\\157')"
PATTERNS=(
  "${P_ROOT}"
  "${P_C_USERS}"
  "${P_WIN_USERS}"
  "${P_D_PROJECT}"
  "${P_LANYUN}"
  "${P_LENOVO}"
)

echo "[privacy-scan] Scanning: ${PROJECT_ROOT}"
found=0
for pattern in "${PATTERNS[@]}"; do
  if grep -RInF \
    --exclude-dir=.git \
    --exclude-dir=__pycache__ \
    --exclude='*.pyc' \
    --exclude='*.pt' \
    --exclude='*.csv' \
    --exclude='privacy_scan.sh' \
    -- "${pattern}" \
    "${PROJECT_ROOT}"; then
    found=1
  fi
done

if [[ "${found}" -eq 1 ]]; then
  echo "[privacy-scan] Sensitive path markers found."
  exit 1
fi

echo "[privacy-scan] OK: no sensitive path markers found."
