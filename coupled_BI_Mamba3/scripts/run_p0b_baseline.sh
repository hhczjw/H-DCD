#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
exec /home/zjw/anaconda3/envs/hdcd/bin/python -u train.py \
  --dataset MOSI --seed 2024 --exp_tag p0B_rollback "$@"
