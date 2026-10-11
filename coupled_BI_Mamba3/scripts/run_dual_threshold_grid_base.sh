#!/bin/bash
# ============================================================
# Coupled-BI-Mamba3 双阈值精调网格 (base 版)
# 目标:
#   - 尝试同时达到 Acc2 > 0.90 且 Acc7 > 0.50
# 说明:
#   - 固定 bert-base-uncased
#   - 固定 batch_size=16
#   - 固定 GPU0
# ============================================================
set -e

cd "$(dirname "$0")/.."
source /home/zjw/anaconda3/bin/activate coupled_mamba_3

# ---- 固定参数 ----
DATASET="MOSI"
FEATURE_A="features/mosi_audio_data2vec_full.pkl"
FEATURE_V="features/split_vision_openface3.pkl"
BERT="roberta-base"
ISM_FULL="true"
MULTI_TASK="true"
KEY_EVAL="MAE"
D_STATE=64

# ---- 搜索空间 ----
# 由 Acc2/Acc7 前五名交集整理得到
DROPOUTS=(0.5 0.6)
LRS=(3e-4 4e-4 5e-4)
ISM_DEPTHS=(1 2 3)
SUB_LOSS_LAMBDAS=(0.1 0.3)
BATCH_SIZES=(16)
SEEDS=(42 1111 2024)

OUT_DIR="./results/dual_threshold_grid_base"
LOG_DIR="./logs/dual_threshold_grid_base"
mkdir -p "${OUT_DIR}" "${LOG_DIR}" ./checkpoints

run_one() {
  local tag="$1"
  local seed="$2"
  local batch_size="$3"
  local dropout="$4"
  local lr="$5"
  local ism_depth="$6"
  local sub_loss_lambda="$7"

  echo ""
  echo "=== ${tag} seed=${seed} @ $(date '+%H:%M:%S') ==="
  echo "  参数: batch_size=${batch_size}, dropout=${dropout}, lr=${lr}, ism_depth=${ism_depth}, sub_loss_lambda=${sub_loss_lambda}"

  CUDA_VISIBLE_DEVICES=0 python -u train.py \
    --dataset "${DATASET}" \
    --seed "${seed}" \
    --feature_A "${FEATURE_A}" \
    --feature_V "${FEATURE_V}" \
    --bert_pretrained "${BERT}" \
    --ism_full_frame "${ISM_FULL}" \
    --multi_task "${MULTI_TASK}" \
    --dropout "${dropout}" \
    --lr "${lr}" \
    --d_state "${D_STATE}" \
    --ism_depth "${ism_depth}" \
    --sub_loss_lambda "${sub_loss_lambda}" \
    --batch_size "${batch_size}" \
    --key_eval "${KEY_EVAL}" \
    --exp_tag "${tag}" \
    2>&1 | tee "${LOG_DIR}/${tag}.log"

  local src_json="results/${DATASET}_${tag}_seed${seed}_test.json"
  if [[ -f "${src_json}" ]]; then
    cp "${src_json}" "${OUT_DIR}/${tag}_seed${seed}_test.json"
  fi

  echo "=== ${tag} seed=${seed} DONE @ $(date '+%H:%M:%S') ==="
}

TOTAL=$(( ${#DROPOUTS[@]} * ${#LRS[@]} * ${#ISM_DEPTHS[@]} * ${#SUB_LOSS_LAMBDAS[@]} * ${#BATCH_SIZES[@]} * ${#SEEDS[@]} ))

echo "============================================"
echo " Dual-Threshold Grid Search (base)"
echo "   Target: Acc2 > 0.90 and Acc7 > 0.50"
echo "   Fixed: bert-base-uncased | d_state=${D_STATE} | OpenFace3 split | Data2Vec | ism_full_frame"
echo "   dropout: ${DROPOUTS[*]}"
echo "   lr:      ${LRS[*]}"
echo "   depth:   ${ISM_DEPTHS[*]}"
echo "   subloss: ${SUB_LOSS_LAMBDAS[*]}"
echo "   batch:   ${BATCH_SIZES[*]}"
echo "   seeds:   ${SEEDS[*]}"
echo "   total runs: ${TOTAL}"
echo "============================================"

i=0
for dropout in "${DROPOUTS[@]}"; do
  for lr in "${LRS[@]}"; do
    for ism_depth in "${ISM_DEPTHS[@]}"; do
      for sub_loss_lambda in "${SUB_LOSS_LAMBDAS[@]}"; do
        for batch_size in "${BATCH_SIZES[@]}"; do
          for seed in "${SEEDS[@]}"; do
            i=$((i+1))
            tag="dtb_bs${batch_size}_sd${seed}_do${dropout}_lr${lr}_ds${D_STATE}_id${ism_depth}_sl${sub_loss_lambda}"
            echo "=== [${i}/${TOTAL}] ${tag} ==="
            run_one "${tag}" "${seed}" "${batch_size}" "${dropout}" "${lr}" "${ism_depth}" "${sub_loss_lambda}"
          done
        done
      done
    done
  done
done

echo ""
echo "============================================"
echo " Dual-Threshold Search Complete (base)"
echo " Results: ${OUT_DIR}"
echo " Logs:    ${LOG_DIR}"
echo "============================================"