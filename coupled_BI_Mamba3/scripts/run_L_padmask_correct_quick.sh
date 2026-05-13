#!/usr/bin/env bash
# =============================================================
# Run L-quick: 端到端 padding-aware + triple ckpt (Acc2 主, Acc7 副, MAE 备)
#
# 本脚本对应 4 项修复 (相对 Run K):
#   P0-a  models/classifier.py:_encode 全链路 zero-out
#         (投影后 / ISM 出口 / 每层 Fork 出口都 zero-out pad)
#   P0-b  layers/ism.py:ISMEncoder.forward 接收 mask, 每层 Block 后 zero-out
#         (避免 BiMamba conv1d/SSM 把 0 输入经 bias 污染状态)
#   P1    train.py 新增 --secondary_metric Acc7  (本脚本用 Acc7 作辅助 ckpt)
#         以前硬编码 MAE, 现在 best_Acc7 单独保存
#   P2    classifier._make_av_mask: L_in==L_out 直接构造, 否则比例缩放兜底
#         (当前 MOSI L=50 不触发, 跨数据集安全)
#
# 数据流确认 (check_data_flow.py 实测):
#   audio  原始 (1284, 375, 5)  -> 模型输入 (B, 50, 5)  pad_ratio≈35.7%
#   vision 原始 (1284, 500, 20) -> 模型输入 (B, 50, 20) pad_ratio≈32.4%
#   pad 区严格 0 (||tail||=0.000000), mask 语义 100% 对齐
#
# 基线对比 (MOSI seed42):
#   H2   (无 mask):                       Acc2=84.76 / Acc7=47.08 / MAE=0.7245
#   K    (错误 mask, 全维度退化):          Acc2=84.60 / Acc7=45.19 / MAE=0.7366
#   L    (本次, 端到端 padding-aware):    期望 Acc2 ≥ 85.0 或 Acc7 ≥ 47.5
#
# Triple ckpt (本次新加):
#   primary_Acc2     - 主监控早停, 报 Acc2/Acc7/Acc5/MAE
#   secondary_Acc7   - 单独保存验证集 Acc7 最优, 直冲 Acc7 上限
#   (MAE 不再保存; 如想换回, 设 SECONDARY_METRIC=MAE)
#
# 通过阈值: ANY ckpt Acc2 ≥ 85.0 OR Acc7 ≥ 47.5
# 警戒线:  全部 ckpt 都低于 H2 → 排查 ISM mask 接口 / smoke test
# =============================================================
set -e
cd "$(dirname "$0")/.."

SEED="${SEED:-42}"
V_SELF_RATIO="${V_SELF_RATIO:-0.15}"   # 默认关, 单变量验证 mask 收益
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
KEY_EVAL="${KEY_EVAL:-Acc2}"            # 主监控 (早停)
SECONDARY_METRIC="${SECONDARY_METRIC:-Acc7}"  # 辅助 ckpt 指标 (新, 默认 Acc7)

TAG="${TAG:-L_padmask_correct_seed${SEED}}"

mkdir -p logs results checkpoints

echo "=========================================="
echo "Run L-quick: end-to-end padding-aware | $(date)"
echo "  tag              = ${TAG}"
echo "  seed             = ${SEED}"
echo "  v_self_ratio     = ${V_SELF_RATIO}  (默认关, 单测 mask)"
echo "  augment_prob     = ${AUG_PROB}"
echo "  key_eval         = ${KEY_EVAL}      (主监控/早停)"
echo "  secondary_metric = ${SECONDARY_METRIC}  (辅助 ckpt, 直冲 Acc7)"
echo "Baselines:"
echo "  H2  (no mask):     Acc2=84.76 / Acc7=47.08 / MAE=0.7245"
echo "  K   (bad mask):    Acc2=84.60 / Acc7=45.19 / MAE=0.7366"
echo "  L   (target):      Acc2 ≥ 85.0 OR Acc7 ≥ 47.5"
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