#!/usr/bin/env bash
# =============================================================
# Run I-quick: 单 seed=42 快验脚本 (调参用, ~15 分钟)
#
# 用法: 
#   bash scripts/run_I_quick.sh                    # 默认 v_self_ratio=0.3
#   V_SELF_RATIO=0.2 bash scripts/run_I_quick.sh   # 探索其他取值
#   V_SELF_RATIO=0.4 TAG_SUFFIX=_v04 bash ...      # 加 tag 后缀防覆盖
#
# 基线 H2·seed=42:  Acc2=84.76  Acc7=47.08  MAE=0.7245
# 通过阈值:        Acc2≥85.0   Acc7≥47.5   MAE≤0.720  → 触发三 seed 全量跑
# 警戒线:          Acc2<84.5   Acc7<46.5  → 改动有害, 回滚或调参
# =============================================================
set -e
cd "$(dirname "$0")/.."

# --- 可通过环境变量覆盖 ---
SEED="${SEED:-42}"
V_SELF_RATIO="${V_SELF_RATIO:-0.3}"
AUG_PROB="${AUG_PROB:-0.5}"            # 代码已"三选一", 实际等效噪声 ↓ 50%
SUB_LAMBDA="${SUB_LAMBDA:-0.2}"
ALPHA="${ALPHA:-0.3}"
LR="${LR:-3e-4}"
GRAD_CLIP="${GRAD_CLIP:-0.3}"
ISM_DEPTH="${ISM_DEPTH:-3}"
EPOCHS="${EPOCHS:-60}"
EARLY_STOP="${EARLY_STOP:-15}"
WARMUP_RATIO="${WARMUP_RATIO:-0.1}"
DROPOUT="${DROPOUT:-0.3}"
EMA="${EMA:-0.999}"
TAG_SUFFIX="${TAG_SUFFIX:-}"

# 自动给 tag 拼上当前 v_self_ratio (除非显式传 TAG_SUFFIX)
if [ -z "${TAG_SUFFIX}" ]; then
  # 0.3 -> 03, 0.25 -> 025
  TAG_SUFFIX="_v$(echo "${V_SELF_RATIO}" | tr -d '.')"
fi

TAG="I_quick_seed${SEED}${TAG_SUFFIX}"

mkdir -p logs results checkpoints

echo "=========================================="
echo "Run I-quick (single seed) | $(date)"
echo "  tag           = ${TAG}"
echo "  seed          = ${SEED}"
echo "  v_self_ratio  = ${V_SELF_RATIO}"
echo "  augment_prob  = ${AUG_PROB} (三选一)"
echo "  sub_lambda    = ${SUB_LAMBDA}  alpha = ${ALPHA}"
echo "  lr=${LR}  grad_clip=${GRAD_CLIP}  ism_depth=${ISM_DEPTH}"
echo "  epochs=${EPOCHS}  early_stop=${EARLY_STOP}  warmup=${WARMUP_RATIO}"
echo "  dropout=${DROPOUT}  ema=${EMA}"
echo "Baseline H2·seed42: Acc2=84.76 / Acc7=47.08 / MAE=0.7245"
echo "=========================================="

python train.py \
  --dataset MOSI \
  --seed "${SEED}" \
  --epochs "${EPOCHS}" \
  --early_stop "${EARLY_STOP}" \
  --warmup_ratio "${WARMUP_RATIO}" \
  --lr "${LR}" \
  --grad_clip "${GRAD_CLIP}" \
  --dropout "${DROPOUT}" \
  --augment_prob "${AUG_PROB}" \
  --ism_depth "${ISM_DEPTH}" \
  --ism_d_state 64 \
  --ism_mixer_type bimamba3 \
  --v_self_ratio "${V_SELF_RATIO}" \
  --aux_cls_weight "${ALPHA}" \
  --aux_num_classes 7 \
  --sub_loss_lambda "${SUB_LAMBDA}" \
  --key_eval MAE \
  --ema_decay "${EMA}" \
  --exp_tag "${TAG}" \
  2>&1 | tee "logs/${TAG}.console.log"

echo ""
echo "=========================================="
echo "Done: $(date)"
echo "Result: results/MOSI_${TAG}_seed${SEED}_test.json"
echo "=========================================="

# 打印关键指标 (依赖 jq, 没装则跳过)
RESULT_JSON="results/MOSI_${TAG}_seed${SEED}_test.json"
if command -v jq >/dev/null 2>&1 && [ -f "${RESULT_JSON}" ]; then
  echo ""
  echo "Quick metrics (primary_MAE ckpt):"
  jq '.primary_MAE | {MAE, Acc2, Acc5, Acc7}' "${RESULT_JSON}"
fi