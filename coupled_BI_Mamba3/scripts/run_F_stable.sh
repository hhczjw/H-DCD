#!/usr/bin/env bash
# =============================================================
# Run F: Run E + 稳定性增强 + 训练计划放宽
#
# 改动相对 Run E:
#   方案 A (稳定性):
#     sub_loss_lambda  : 0.3  -> 0.2   (复合损失梯度幅度 ↓15%)
#     learning_rate    : 5e-4 -> 4e-4  (下游 lr ↓20%)
#     warmup_ratio     : 0.05 -> 0.10  (热身更稳, A_log 不易漂)
#     grad_clip        : 1.0  -> 0.5   (砍末期梯度尖峰)
#   方案 B (训练计划):
#     epochs           : 40   -> 60    (Acc7 末期还在涨, 给它涨完)
#     early_stop       : 10   -> 15    (避免 Acc2 横盘期被误停)
#
# 配方 (与 Run E 共同保留):
#   alpha (aux_cls_weight) = 0.3   ← 7 类离散 CE
#   KeyEval                = MAE
#   EMA decay              = 0.999
#   三 seed: 42 / 2024 / 0
#
# 目标: 在 trainer 三处 EMA 修复基础上, 彻底消除 NaN, 让 Acc7 跑满
# 预期: 单 seed Acc2 ≥ 85.0, Acc7 ≥ 43.0; Ensemble3 Acc2 ≥ 86.0
# =============================================================
set -e
cd "$(dirname "$0")/.."

SEEDS=(42 2024 0)
ALPHA=0.3
SUB_LAMBDA=0.2          # ← Run E: 0.3
EMA=0.999
DATASET=MOSI
EPOCHS=60               # ← Run E: 40
EARLY_STOP=15           # ← Run E: 10
LR=4e-4                 # ← Run E: 5e-4
WARMUP=0.10             # ← Run E: 0.05
GRAD_CLIP=0.5           # ← Run E: 1.0 (硬编码在 trainer)

mkdir -p logs results checkpoints

echo "=========================================="
echo "Run F: stability + extended training"
echo "  alpha=${ALPHA}, sub_lambda=${SUB_LAMBDA}, ema=${EMA}, key=MAE"
echo "  epochs=${EPOCHS}, early_stop=${EARLY_STOP}"
echo "  lr=${LR}, warmup=${WARMUP}, grad_clip=${GRAD_CLIP}"
echo "  Seeds: ${SEEDS[*]}"
echo "  Started: $(date)"
echo "=========================================="

for SEED in "${SEEDS[@]}"; do
  TAG="F_stable_seed${SEED}"
  echo ""
  echo ">>> [$(date +%H:%M:%S)] Running ${TAG} ..."
  python train.py \
    --dataset "${DATASET}" \
    --seed "${SEED}" \
    --epochs "${EPOCHS}" \
    --early_stop "${EARLY_STOP}" \
    --lr "${LR}" \
    --warmup_ratio "${WARMUP}" \
    --grad_clip "${GRAD_CLIP}" \
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
echo "All Run F seeds finished. Aggregating ..."
echo "=========================================="

python scripts/agg_run_C.py --pattern "MOSI_F_stable_seed*_test.json"

echo ""
echo "Run F completed: $(date)"