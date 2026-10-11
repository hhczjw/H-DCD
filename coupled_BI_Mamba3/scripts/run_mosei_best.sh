#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-/home/zjw/anaconda3/envs/coupled_mamba_3/bin/python}"
export CUDA_VISIBLE_DEVICES="${MOSEI_GPU:-1}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
"$PYTHON_BIN" scripts/prepare_mosei_best.py --full-check
exec "$PYTHON_BIN" -u train.py --config_path configs/mosei_p0B.json \
    --dataset MOSEI --seed "${MOSEI_SEED:-2024}" --exp_tag p0B_mosei "$@"
