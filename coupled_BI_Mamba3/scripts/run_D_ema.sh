#!/usr/bin/env bash
# =============================================================
# Run D: alpha=0.3 + KeyEval=MAE + EMA(0.999) 三 seed
#   目标: 在 Run C (Acc2 83.0% / Acc7 43.8%) 基础上再提升 Acc2
#   预期: Acc2 84%+, Acc7 44%+, std 进一步降低
# =============================================================
set -e
cd "$(dirname "$0")/.."

SEEDS=(42 2024 0)
ALPHA=0.3
EMA=0.999
DATASET=MOSI

mkdir -p logs results checkpoints

echo "=========================================="
echo "Run D: alpha=${ALPHA}, ema=${EMA}, key_eval=MAE, seeds=${SEEDS[*]}"
echo "Started: $(date)"
echo "=========================================="

for SEED in "${SEEDS[@]}"; do
  TAG="D_alpha03_ema_seed${SEED}"
  echo ""
  echo ">>> [$(date +%H:%M:%S)] Running ${TAG} ..."
  python train.py \
    --dataset "${DATASET}" \
    --seed "${SEED}" \
    --aux_cls_weight "${ALPHA}" \
    --aux_num_classes 7 \
    --key_eval MAE \
    --ema_decay "${EMA}" \
    --exp_tag "${TAG}" \
    2>&1 | tee "logs/${TAG}.console.log"
  echo ">>> [$(date +%H:%M:%S)] ${TAG} done."
done

echo ""
echo "=========================================="
echo "All seeds finished. Aggregating ..."
echo "=========================================="

python scripts/agg_run_C.py --pattern "MOSI_D_alpha03_ema_seed*_test.json"

echo ""
echo "Run D completed: $(date)"