#!/usr/bin/env bash
# =============================================================
# Run K2-quick: text-mask only (audio/vision mask 已禁用)
#
# 修复 Run K 的全维度退化:
#   K 的根因 = adaptive_avg_pool1d 把原始稀疏信号摊匀到 50 步,
#   audio/vision 按 valid_len*Lt/orig_L 缩放 mask 错误屏蔽 95% 位置.
#
# K2 改动:
#   ① audio/vision 不构造 mask, 全 50 位置参与 attention
#   ② 仅保留 text 的 BERT attention_mask (pad_ratio=70.4%, 收益最大)
#   ③ 其余与 K 完全一致 (V_SELF_RATIO=0, AUG_PROB=0.5, sub_loss=0.2, alpha=0.3)
#
# 基线 H2·seed=42:    Acc2=84.76, Acc7=47.08, MAE=0.7245
# Run K (错误 mask):  Acc2=84.60, Acc7=45.19, MAE=0.7366
# 通过阈值:           Acc7 ≥ 47.5  → 启动三 seed
# 警戒线:             Acc7 < 46.5  → text mask 也无效, 回滚
# =============================================================
set -e
cd "$(dirname "$0")/.."

SEED="${SEED:-42}"
V_SELF_RATIO="${V_SELF_RATIO:-0.0}"
AUG_PROB="${AUG_PROB:-0.5}"
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

TAG="${TAG:-K2_textmask_seed${SEED}}"

mkdir -p logs results checkpoints

echo "=========================================="
echo "Run K2-quick: text-mask only | $(date)"
echo "  tag           = ${TAG}"
echo "  seed          = ${SEED}"
echo "  v_self_ratio  = ${V_SELF_RATIO}"
echo "  augment_prob  = ${AUG_PROB}"
echo "  改动: 禁用 audio/vision mask, 仅保留 text mask"
echo "Baseline H2·seed42: Acc2=84.76 / Acc7=47.08 / MAE=0.7245"
echo "Run K (bad mask):    Acc2=84.60 / Acc7=45.19 / MAE=0.7366"
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

RESULT_JSON="results/MOSI_${TAG}_seed${SEED}_test.json"
if command -v jq >/dev/null 2>&1 && [ -f "${RESULT_JSON}" ]; then
  echo ""
  echo "Quick metrics (primary_MAE ckpt):"
  jq '.primary_MAE | {MAE, Acc2, Acc5, Acc7}' "${RESULT_JSON}"
fi

echo ""
echo "Done: $(date)"