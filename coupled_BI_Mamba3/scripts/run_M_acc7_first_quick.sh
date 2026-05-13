#!/usr/bin/env bash
# =============================================================
# Run M-quick: Acc7-first + padded-aware + dual ckpt (Acc7 主, MAE 副)
#
# 设计目标:
#   - 将主监控从 Acc2 切换为 Acc7, 直接优化离散分类一致性
#   - 保留 padding-aware 全链路 mask 修复
#   - 通过降低 sub_loss 的回归约束, 减少对 Acc7 的“拉回”效应
#
# 相对 Run L 的核心变化:
#   1) KEY_EVAL=Acc7 (早停/主 ckpt 改为 Acc7)
#   2) SECONDARY_METRIC=MAE (保留回归侧备份)
#   3) alpha 提升, sub_loss_lambda 降低, 更偏向 Acc7
#   4) v_self_ratio 适度打开, 给跨模态 V 通道一个目标模态锚
#
# 参考基线 (seed42, MOSI):
#   - L primary_Acc2: Acc7=0.3732
#   - L secondary_Acc7: Acc7=0.4286
#
# M 的期望:
#   - 让 primary_Acc7 至少不低于 secondary_Acc7
#   - 尽量把 test Acc7 推到 0.43 以上
# =============================================================
set -e
cd "$(dirname "$0")/.."

SEED="${SEED:-42}"
V_SELF_RATIO="${V_SELF_RATIO:-0.15}"
AUG_PROB="${AUG_PROB:-0.5}"
SUB_LAMBDA="${SUB_LAMBDA:-0.05}"
ALPHA="${ALPHA:-0.5}"
LR="${LR:-3e-4}"
GRAD_CLIP="${GRAD_CLIP:-0.3}"
ISM_DEPTH="${ISM_DEPTH:-3}"
EPOCHS="${EPOCHS:-120}"
EARLY_STOP="${EARLY_STOP:-15}"
WARMUP_RATIO="${WARMUP_RATIO:-0.1}"
DROPOUT="${DROPOUT:-0.3}"
EMA="${EMA:-0.999}"
KEY_EVAL="${KEY_EVAL:-Acc7}"
SECONDARY_METRIC="${SECONDARY_METRIC:-MAE}"

TAG="${TAG:-M_acc7_first_seed${SEED}}"

mkdir -p logs results checkpoints

echo "=========================================="
echo "Run M-quick: Acc7-first padding-aware | $(date)"
echo "  tag              = ${TAG}"
echo "  seed             = ${SEED}"
echo "  v_self_ratio     = ${V_SELF_RATIO}  (适度打开, 给 V 通道目标模态锚)"
echo "  augment_prob     = ${AUG_PROB}"
echo "  alpha            = ${ALPHA}  (提高离散 CE 权重)"
echo "  sub_loss_lambda  = ${SUB_LAMBDA}  (降低回归侧干扰)"
echo "  key_eval         = ${KEY_EVAL}  (主监控/早停)"
echo "  secondary_metric = ${SECONDARY_METRIC}  (回归备份 ckpt)"
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
  --key_eval "${KEY_EVAL}" \
  --secondary_metric "${SECONDARY_METRIC}" \
  --ema_decay "${EMA}" \
  --exp_tag "${TAG}" \
  2>&1 | tee "logs/${TAG}.console.log"

RESULT_JSON="results/MOSI_${TAG}_seed${SEED}_test.json"
if command -v jq >/dev/null 2>&1 && [ -f "${RESULT_JSON}" ]; then
  echo ""
  echo "=== Test metrics (both ckpts) ==="
  jq '. | to_entries | map({ckpt: .key, MAE: .value.MAE, Acc2: .value.Acc2, Acc5: .value.Acc5, Acc7: .value.Acc7})' "${RESULT_JSON}"
fi

echo ""
echo "Done: $(date)"