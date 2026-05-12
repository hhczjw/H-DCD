#!/usr/bin/env bash
# =============================================================
# Run E: Run D 配方 + sub_loss_lambda=0.3 (对齐 MSAmba sub_fc_T/A/V)
#
# 配方:
#   - alpha (aux_cls_weight) = 0.3   ← 7 类离散 CE
#   - sub_loss_lambda        = 0.3   ← 模态级 SmoothL1 (T/A/V 各一路)
#   - KeyEval = MAE                  ← 主监控
#   - EMA decay = 0.999              ← Polyak 平均
#   - 三 seed: 42 / 2024 / 0
#
# 目标: 在 D-Ensemble3 (Acc2 84.30) 基础上再提升, 逼近 MSAmba (86.11)
# 预期: 单 seed Acc2 ≥ 84.0%, Ensemble3 ≥ 85.0%
# =============================================================
set -e
cd "$(dirname "$0")/.."

SEEDS=(42 2024 0)
ALPHA=0.3
SUB_LAMBDA=0.3
EMA=0.999
DATASET=MOSI

mkdir -p logs results checkpoints

echo "=========================================="
echo "Run E: alpha=${ALPHA}, sub_loss_lambda=${SUB_LAMBDA}, ema=${EMA}, key=MAE"
echo "Seeds: ${SEEDS[*]}"
echo "Started: $(date)"
echo "=========================================="

for SEED in "${SEEDS[@]}"; do
  TAG="E_subloss_seed${SEED}"
  echo ""
  echo ">>> [$(date +%H:%M:%S)] Running ${TAG} ..."
  python train.py \
    --dataset "${DATASET}" \
    --seed "${SEED}" \
    --aux_cls_weight "${ALPHA}" \
    --aux_num_classes 7 \
    --sub_loss_lambda "${SUB_LAMBDA}" \
    --key_eval MAE \
    --ema_decay "${EMA}" \
    --exp_tag "${TAG}" \
    2>&1 | tee "logs/${TAG}.console.log"
  echo ">>> [$(date +%H:%M:%S)] ${TAG} done."
done

echo ""
echo "=========================================="
echo "All Run E seeds finished. Aggregating ..."
echo "=========================================="

python scripts/agg_run_C.py --pattern "MOSI_E_subloss_seed*_test.json"

echo ""
echo "Run E completed: $(date)"