#!/usr/bin/env bash
# =============================================================
# Run H: Run F 基线 + 结构升级 (ism_depth 2 -> 3)
#
# Run G 失败教训 (3 seeds avg vs Run F):
#   Acc-2: 84.10 (-0.12), Acc-7: 45.09 (-1.34), MAE: 0.7496 (+0.018)
#   ⇒ 强正则 (dropout/wd/aux_cls↑) 让模型欠拟合, 全面回退
#   ⇒ 纯调超参的天花板已到 84.2/46.4 (Run F)
#
# Run H 思路:
#   不动正则, 只升级模型容量 — 加深 ISM (跨模态交互模块) 一层
#   ism_depth: 2 -> 3
#   双向跨模态交互更深, 模态融合质量↑
#
# 配方 (与 Run F 一致, 只改 1 个旋钮):
#   ism_depth        : 2    -> 3       (★ 唯一改动)
#   dropout          : 0.3  (Run F)
#   weight_decay     : 1e-4 (Run F, config 默认)
#   aux_cls_weight   : 0.3  (Run F)
#   sub_loss_lambda  : 0.2  (Run F)
#   ema_decay        : 0.999
#   lr=4e-4, warmup=0.10, grad_clip=0.5
#   epochs=60, early_stop=15
#
# 预期:
#   单 seed Acc-2 ≥ 85.0, Acc-7 ≥ 47.5
#   3-seed avg Acc-2 ≥ 84.8 (+0.6), Acc-7 ≥ 47.5 (+1.0)
#   显存增加约 10%, 单 seed 训练时长 +10~15%
# =============================================================
set -e
cd "$(dirname "$0")/.."

SEEDS=(42 2024 0)

# ---- Run F 基线参数 (全部继承) ----
ALPHA=0.3
SUB_LAMBDA=0.2
EMA=0.999
DATASET=MOSI
EPOCHS=60
EARLY_STOP=15
LR=4e-4
WARMUP=0.10
GRAD_CLIP=0.5

# ---- Run H 唯一改动 ----
ISM_DEPTH=3       # ← Run F (config 默认): 2

mkdir -p logs results checkpoints

echo "=========================================="
echo "Run H: structure upgrade (ism_depth 2->3)"
echo "  Inherit Run F:"
echo "    alpha=${ALPHA}, sub_lambda=${SUB_LAMBDA}, ema=${EMA}"
echo "    lr=${LR}, warmup=${WARMUP}, grad_clip=${GRAD_CLIP}"
echo "    epochs=${EPOCHS}, early_stop=${EARLY_STOP}"
echo "  Run H change:"
echo "    ism_depth      : 2 -> ${ISM_DEPTH}"
echo "  Seeds: ${SEEDS[*]}"
echo "  Started: $(date)"
echo "=========================================="

for SEED in "${SEEDS[@]}"; do
  TAG="H_ism3_seed${SEED}"
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
echo "All Run H seeds finished. Aggregating ..."
echo "=========================================="

python scripts/agg_run_C.py --pattern "MOSI_H_ism3_seed*_test.json"

echo ""
echo "Run H completed: $(date)"