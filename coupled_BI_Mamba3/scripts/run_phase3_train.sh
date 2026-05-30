#!/bin/bash
set -e
cd "$(dirname "$0")/.."

DATASET="MOSI"
SEEDS=(42 1111 2024)
EPOCHS=120
BATCH_SIZE=32
GPU=0

FEATURE_PKL="./features/mosi_audio_data2vec.pkl"

echo ""
echo ">>> [实验 B] Phase 3: Data2Vec 离线编码 (严格控制变量版) <<<"

for SEED in "${SEEDS[@]}"; do
    echo "--- Seed ${SEED} ---"
    CUDA_VISIBLE_DEVICES=${GPU} python train.py \
        --dataset ${DATASET} \
        --seed ${SEED} \
        --epochs ${EPOCHS} \
        --batch_size ${BATCH_SIZE} \
        --bert_pretrained bert-base-uncased \
        --feature_A "${FEATURE_PKL}" \
        --skip_audio_ism false \
        --ism_depth 2 \
        --lr 3e-4 \
        --bert_lr 2e-5 \
        --dropout 0.3 \
        --v_self_ratio 0.3 \
        --sub_loss_lambda 0.3 \
        --warmup_ratio 0.15 \
        --grad_clip 0.3 \
        --key_eval Acc2 \
        --secondary_metric MAE \
        --exp_tag phase3_data2vec_offline_bert \
        2>&1 | tee logs/phase3_data2vec_offline_bert_seed${SEED}.log
    echo "  完成 seed=${SEED}"
done
