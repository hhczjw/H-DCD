#!/usr/bin/env bash
# =============================================================
# Run H2-A: Run H + 稳定性补强 (修复 epoch 22 NaN 爆破)
#
# Run H 故障 (seed42):
#   epoch 1-21 正常, valid Acc-2 已到 0.8565, valid MAE=0.95
#   epoch 22 step 78 首次 NaN -> epoch 23/24 全部 NaN -> 模型死亡
#   ⇒ ism_depth=3 让 3 层 BiMamba3 串联, 内部 A_log 累积漂移失控
#   ⇒ grad_clip=0.5 在 2 层够用, 3 层压不住
#
# Run H2 思路:
#   保留 ism_depth=3 (容量收益), 加固稳定性
#   - grad_clip : 0.5  -> 0.3   (砍更狠的尖峰)
#   - lr        : 4e-4 -> 3e-4  (-25%, 减小 A_log 更新步长)
#   其余完全继承 Run H
#
# 预期:
#   能跨过 epoch 22 崩溃点, 跑完 60 epoch
#   单 seed Acc-2 ≥ 85.0, Acc-7 ≥ 47.0
#   3-seed avg Acc-2 ≥ 84.8 (+0.6 vs Run F), Acc-7 ≥ 47.5
#
# 兜底:
#   若仍 NaN, 直接回退 Run H2-B (ism_depth=2 + num_layers=4)
# =============================================================
set -e
cd "$(dirname "$0")/.."

SEEDS=(42 2024 0)

# ---- 继承 Run F/H 基线 ----
ALPHA=0.3
SUB_LAMBDA=0.2
EMA=0.999
DATASET=MOSI
EPOCHS=60
EARLY_STOP=15
WARMUP=0.10

# ---- Run H 改动 (保留) ----
ISM_DEPTH=3       # ← Run F: 2

# ---- Run H2 稳定性补强 ----
LR=3e-4           # ← Run H: 4e-4 (-25%)
GRAD_CLIP=0.3     # ← Run H: 0.5  (-40%)

mkdir -p logs results checkpoints

echo "=========================================="
echo "Run H2-A: ism3 + stability patch"
echo "  Inherit Run F/H:"
echo "    alpha=${ALPHA}, sub_lambda=${SUB_LAMBDA}, ema=${EMA}"
echo "    warmup=${WARMUP}, epochs=${EPOCHS}, early_stop=${EARLY_STOP}"
echo "    ism_depth=${ISM_DEPTH}"
echo "  Run H2 stability changes:"
echo "    lr             : 4e-4 -> ${LR}"
echo "    grad_clip      : 0.5  -> ${GRAD_CLIP}"
echo "  Seeds: ${SEEDS[*]}"
echo "  Started: $(date)"
echo "=========================================="

for SEED in "${SEEDS[@]}"; do
  TAG="H2_ism3_stable_seed${SEED}"
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
    --ism_depth "${ISM_DEPTH}" \
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
echo "All Run H2 seeds finished. Aggregating ..."
echo "=========================================="

python scripts/agg_run_C.py --pattern "MOSI_H2_ism3_stable_seed*_test.json"

echo ""
echo "Run H2 completed: $(date)"