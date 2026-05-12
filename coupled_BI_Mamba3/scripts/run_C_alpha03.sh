#!/usr/bin/env bash
# =============================================================
# Run C: alpha=0.3 三 seed 稳健性验证
#   基于 Run B 消融结论 (B2/alpha=0.3 secondary 最优), 验证跨 seed 是否稳定
# 评估指标: test_secondary (best MAE ckpt) 三 seed 均值/std
# 通过门槛: Acc7 mean >= 0.42 且 std <= 4%; Acc2 mean >= 0.84
# =============================================================
set -e
cd "$(dirname "$0")/.."

SEEDS=(42 2024 0)
ALPHA=0.3
DATASET=MOSI

mkdir -p logs results checkpoints

echo "=========================================="
echo "Run C: alpha=${ALPHA}, seeds=${SEEDS[*]}"
echo "Started: $(date)"
echo "=========================================="

for SEED in "${SEEDS[@]}"; do
  TAG="C_alpha03_seed${SEED}"
  echo ""
  echo ">>> [$(date +%H:%M:%S)] Running ${TAG} ..."
  python train.py \
    --dataset "${DATASET}" \
    --seed "${SEED}" \
    --aux_cls_weight "${ALPHA}" \
    --aux_num_classes 7 \
    --exp_tag "${TAG}" \
    2>&1 | tee "logs/${TAG}.console.log"
  echo ">>> [$(date +%H:%M:%S)] ${TAG} done."
done

echo ""
echo "=========================================="
echo "All seeds finished. Aggregating ..."
echo "=========================================="

python scripts/agg_run_C.py --pattern "MOSI_C_alpha03_seed*_test.json"

echo ""
echo "Run C completed: $(date)"