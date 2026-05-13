#!/usr/bin/env bash
# =============================================================
# Run K-quick: padding mask 单 seed=42 快验
#
# 改动相对于 H2 baseline:
#   ① AttentionPooling 接受 mask, pad 位置不参与 softmax
#   ② BertTextEncoder 输出 attention_mask, text 池化用 BERT mask
#   ③ audio/vision 池化用 audio_lengths/vision_lengths 缩放后构造 mask
#   ④ 数据增强函数保护 pad 位置 (时间 mask + 高斯噪声仅在有效区)
#
# 不带 V_self (V_SELF_RATIO=0), 单独验证 padding mask 收益.
#
# 数据真实分布 (来自 check_seq_lengths):
#   text:   pad_ratio = 70.4%  (mean=14.78, P90=26)  ← 收益最大
#   audio:  pad_ratio = 22.3%  (mean=38.85)
#   vision: pad_ratio = 14.7%  (mean=42.63)
#
# 基线 H2·seed=42: Acc2=84.76, Acc7=47.08, MAE=0.7245
# 通过阈值:        Acc7 ≥ 47.5  → 启动三 seed
# 警戒线:          Acc7 < 46.0  → 改动有 bug, 排查
# =============================================================
set -e
cd "$(dirname "$0")/.."

SEED="${SEED:-42}"
V_SELF_RATIO="${V_SELF_RATIO:-0.0}"   # 默认关闭, 纯测 padding mask
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

TAG="${TAG:-K_padmask_seed${SEED}}"

mkdir -p logs results checkpoints

echo "=========================================="
echo "Run K-quick: padding mask only | $(date)"
echo "  tag           = ${TAG}"
echo "  seed          = ${SEED}"
echo "  v_self_ratio  = ${V_SELF_RATIO} (默认关, 纯测 padding mask)"
echo "  augment_prob  = ${AUG_PROB} (三选一 + pad-protect)"
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

RESULT_JSON="results/MOSI_${TAG}_seed${SEED}_test.json"
if command -v jq >/dev/null 2>&1 && [ -f "${RESULT_JSON}" ]; then
  echo ""
  echo "Quick metrics (primary_MAE ckpt):"
  jq '.primary_MAE | {MAE, Acc2, Acc5, Acc7}' "${RESULT_JSON}"
fi

echo ""
echo "Done: $(date)"