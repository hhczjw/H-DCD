#!/usr/bin/env bash
# =============================================================
# Run I: V_self 注入 + 增强降档 (基于 Run H2 = 当前最佳基线)
#   基线 H2: Acc2 mean=84.71%, Acc7 mean=46.21% (3 seeds)
#   改动:
#     ① V_self 注入 (问题 ③):  --v_self_ratio 0.3
#         x = 0.3 * x_default(tgt) + 0.7 * x_src_weighted
#         给 V 通道一个 tgt 自身锚, 防止 src 含噪带偏 tgt
#     ② 增强降档 (问题 ④):     --augment_prob 0.5 (沿用) + 三选一(代码已改)
#         单样本 P(增强)=0.5, 若增强则 P(每种)=1/3, 不再叠加 → SNR 不再下降
#   保留: ism_depth=3, lr=3e-4, grad_clip=0.3, ema=0.999
#         alpha=0.3, sub_loss_lambda=0.2 (已归一)
#   目标: Acc2 ≥ 85.0, Acc7 ≥ 47.0
# =============================================================
set -e
cd "$(dirname "$0")/.."

SEEDS=(42 2024 0)
DATASET=MOSI
TAG_PREFIX="I_vself_aug3choose1"

# --- 与 H2 一致的稳定基线 (config.json 默认值与 H2 不同, 必须 CLI 全部覆盖!) ---
LR=3e-4              # config 默认 5e-4 → 必须覆盖
GRAD_CLIP=0.3        # config 无, trainer 默认 0.5 → 必须覆盖
ISM_DEPTH=3          # config 默认 2 → 必须覆盖
ISM_D_STATE=64
ALPHA=0.3
SUB_LAMBDA=0.2       # config 默认 0.0 → 必须覆盖
EMA=0.999            # config 无 → 必须覆盖
DROPOUT=0.3
AUG_PROB=0.5         # 代码已改"三选一", 等效噪声 ↓ ~50%
EPOCHS=60            # config 默认 40 → 必须覆盖
EARLY_STOP=15        # config 默认 10 → 必须覆盖
WARMUP_RATIO=0.1     # config 默认 0.05 → 必须覆盖

# --- Run I 新增 ---
V_SELF_RATIO=0.3

mkdir -p logs results checkpoints

echo "=========================================="
echo "Run I: v_self_ratio=${V_SELF_RATIO}, aug_prob=${AUG_PROB}(三选一)"
echo "Baseline H2: ism_depth=${ISM_DEPTH}, lr=${LR}, grad_clip=${GRAD_CLIP}, epochs=${EPOCHS}"
echo "Seeds: ${SEEDS[*]}"
echo "Started: $(date)"
echo "=========================================="

for SEED in "${SEEDS[@]}"; do
  TAG="${TAG_PREFIX}_seed${SEED}"
  echo ""
  echo ">>> [$(date +%H:%M:%S)] Running ${TAG} ..."
  python train.py \
    --dataset "${DATASET}" \
    --seed "${SEED}" \
    --epochs "${EPOCHS}" \
    --early_stop "${EARLY_STOP}" \
    --warmup_ratio "${WARMUP_RATIO}" \
    --lr "${LR}" \
    --grad_clip "${GRAD_CLIP}" \
    --dropout "${DROPOUT}" \
    --augment_prob "${AUG_PROB}" \
    --ism_depth "${ISM_DEPTH}" \
    --ism_d_state "${ISM_D_STATE}" \
    --ism_mixer_type bimamba3 \
    --v_self_ratio "${V_SELF_RATIO}" \
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
echo "All seeds finished. Aggregating ..."
echo "=========================================="

python scripts/agg_run_C.py --pattern "MOSI_${TAG_PREFIX}_seed*_test.json"

echo ""
echo "Run I completed: $(date)"