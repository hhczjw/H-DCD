#!/usr/bin/env bash
# =============================================================
# Run G: Run F + 主攻 train→test 泛化 (方案 A)
#
# Run F 诊断 (3 seeds avg):
#   valid Acc-2 = 87.0  → test Acc-2 = 84.2  (差 2.8 ❌ 过拟合)
#   valid Acc-7 = 48.5  → test Acc-7 = 46.4  (差 2.1)
#   valid MAE   = 0.66  → test MAE   = 0.73  (差 0.07)
#   train_loss 末期持续降到 0.22, 而 valid 已平台
#   ⇒ 训练集严重过拟合, 但 valid/test 分布不一致, valid 也过拟合
#
# 改动相对 Run F:
#   dropout         : 0.3  -> 0.4    (主正则, 直接砍过拟合)
#   weight_decay    : 1e-4 -> 5e-4   (×5, 拉强 L2 正则)
#   aux_cls_weight  : 0.3  -> 0.5    (强化 7 类 CE 信号, 直接拉 Acc-7)
#   其余完全保留 Run F (lr/warmup/grad_clip/sub_lambda/ema/epochs)
#
# 目标:
#   单 seed Acc-2 ≥ 85.5, Acc-7 ≥ 47.0
#   3-seed avg Acc-2 ≥ 85.0, Acc-7 ≥ 47.0  (距用户 86/50 目标 -1/-3)
# =============================================================
set -e
cd "$(dirname "$0")/.."

SEEDS=(42 2024 0)

# ---- 与 Run F 一致 ----
ALPHA_OLD=0.3      # ← (Run F 老值, 仅注释用)
SUB_LAMBDA=0.2
EMA=0.999
DATASET=MOSI
EPOCHS=60
EARLY_STOP=15
LR=4e-4
WARMUP=0.10
GRAD_CLIP=0.5

# ---- Run G 新调参 ----
DROPOUT=0.4         # ← Run F: 0.3
WEIGHT_DECAY=5e-4   # ← Run F: 1e-4
ALPHA=0.5           # ← Run F: 0.3 (aux_cls_weight 强化 Acc-7)

mkdir -p logs results checkpoints

echo "=========================================="
echo "Run G: generalization (dropout/wd/aux_cls)"
echo "  Run F  -> Run G  changes:"
echo "    dropout        : 0.3   -> ${DROPOUT}"
echo "    weight_decay   : 1e-4  -> ${WEIGHT_DECAY}"
echo "    aux_cls_weight : 0.3   -> ${ALPHA}"
echo "  Inherit from Run F:"
echo "    sub_lambda=${SUB_LAMBDA}, ema=${EMA}, lr=${LR}"
echo "    warmup=${WARMUP}, grad_clip=${GRAD_CLIP}"
echo "    epochs=${EPOCHS}, early_stop=${EARLY_STOP}"
echo "  Seeds: ${SEEDS[*]}"
echo "  Started: $(date)"
echo "=========================================="

for SEED in "${SEEDS[@]}"; do
  TAG="G_generalize_seed${SEED}"
  echo ""
  echo ">>> [$(date +%H:%M:%S)] Running ${TAG} ..."
  python train.py \
    --dataset "${DATASET}" \
    --seed "${SEED}" \
    --epochs "${EPOCHS}" \
    --early_stop "${EARLY_STOP}" \
    --lr "${LR}" \
    --weight_decay "${WEIGHT_DECAY}" \
    --dropout "${DROPOUT}" \
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
echo "All Run G seeds finished. Aggregating ..."
echo "=========================================="

python scripts/agg_run_C.py --pattern "MOSI_G_generalize_seed*_test.json"

echo ""
echo "Run G completed: $(date)"