#!/bin/bash
# ============================================================
# Coupled-BI-Mamba3 网格搜索 — 基于最佳配置的精细调优
# 固定: RoBERTa + Data2Vec音频 + OpenFace3视频 + ism_full_frame
# ============================================================
set -e

cd "$(dirname "$0")/.."
source /home/zjw/anaconda3/bin/activate coupled_mamba_3

# ---- 固定参数 ----
DATASET="MOSI"
SEED=42
FEATURE_A="features/mosi_audio_data2vec_full.pkl"
FEATURE_V="features/split_vision_openface3.pkl"
BERT="roberta-base"
ISM_FULL="true"
MULTI_TASK="true"
BATCH_SIZE=16
KEY_EVAL="MAE"
# ism_depth / sub_loss_lambda 在下方搜索空间中定义

# ---- 搜索空间 ----
DROPOUTS=(0.5 0.6 0.7)
LRS=(3e-4 5e-4 7e-4)
DSTATES=(32 64)
ISM_DEPTHS=(1 2 3)
SUB_LOSS_LAMBDAS=(0.1 0.3)
# ---- 如需缩减搜索, 注释掉不需要的行即可 ----

TOTAL=$(( ${#DROPOUTS[@]} * ${#LRS[@]} * ${#DSTATES[@]} * ${#ISM_DEPTHS[@]} * ${#SUB_LOSS_LAMBDAS[@]} ))
echo "============================================"
echo " Grid Search: dropout×lr×d_state×ism_depth×sub_loss"
echo "   ${#DROPOUTS[@]}×${#LRS[@]}×${#DSTATES[@]}×${#ISM_DEPTHS[@]}×${#SUB_LOSS_LAMBDAS[@]} = ${TOTAL} runs"
echo " Fixed: RoBERTa | Data2Vec | OpenFace3 | ism_full_frame"
echo "============================================"

i=0
for dropout in "${DROPOUTS[@]}"; do
  for lr in "${LRS[@]}"; do
    for d_state in "${DSTATES[@]}"; do
      for ism_depth in "${ISM_DEPTHS[@]}"; do
        for sub_loss_lambda in "${SUB_LOSS_LAMBDAS[@]}"; do
          i=$((i+1))
          tag="gs9_do${dropout}_lr${lr}_ds${d_state}_id${ism_depth}_sl${sub_loss_lambda}"

      echo ""
      echo "=== [$i/$TOTAL] $tag ==="

      python train.py \
        --dataset "$DATASET" \
        --seed "$SEED" \
        --feature_A "$FEATURE_A" \
        --feature_V "$FEATURE_V" \
        --bert_pretrained "$BERT" \
        --ism_full_frame "$ISM_FULL" \
        --multi_task "$MULTI_TASK" \
        --sub_loss_lambda "$sub_loss_lambda" \
        --dropout "$dropout" \
        --lr "$lr" \
        --d_state "$d_state" \
        --ism_depth "$ism_depth" \
        --batch_size "$BATCH_SIZE" \
        --key_eval "$KEY_EVAL" \
        --exp_tag "$tag"

      echo "=== [$i/$TOTAL] $tag DONE ==="
        done
      done
    done
  done
done

echo ""
echo "============================================"
echo " Grid Search Complete: ${TOTAL} runs"
echo " Results: results/MOSI_gs_v9_*_test.json"
echo "============================================"
