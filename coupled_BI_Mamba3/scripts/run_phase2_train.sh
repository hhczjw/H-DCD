#!/bin/bash
# =============================================================================
# Phase 2 v2 训练脚本: 文本编码器升级验证 (BERT-base → RoBERTa-base)
# =============================================================================
#
# 实验设计:
#   实验 A (Baseline): BERT-base-uncased — 通过 CLI 覆盖
#   实验 B (Phase 2):  RoBERTa-base       — 显式指定, 确保覆盖 config.json
#
# 参数优化 (基于 v1 日志分析):
#   - main_lr: 5e-4 → 3e-4  (降低学习率, 应对 seed 1111 崩溃问题)
#   - bert_lr: 1e-5 → 2e-5   (RoBERTa 微调需要稍高学习率)
#   - dropout: 0.25 → 0.3     (对齐 CAGMamba, 增强正则化)
#   - patience: 8 → 12        (MAE 早停允许更长训练)
#   - warmup_ratio: 0.1 → 0.15 (更长的 warmup, 提升初期稳定性)
#   - KeyEval: MAE (主) + Acc7 (辅助)  — 两者互补
#   - secondary_metric: Acc7  (Acc7 与 MAE 更互补)
#   - grad_clip: 0.5 → 0.3    (收紧梯度裁剪, 防止尖峰)
#
# 用法:
#   chmod +x scripts/run_phase2_train.sh
#   bash scripts/run_phase2_train.sh
# =============================================================================

set -e
cd "$(dirname "$0")/.."

DATASET="MOSI"
SEEDS=(42 1111 2024)
EPOCHS=120
BATCH_SIZE=32
GPU=0

echo "============================================================"
echo " Phase 2 v2: 文本编码器升级 (参数优化版)"
echo " 数据集: ${DATASET} | Seeds: ${SEEDS[*]} | Epochs: ${EPOCHS}"
echo " main_lr=3e-4 | bert_lr=2e-5 | dropout=0.3"
echo " KeyEval=MAE | sec=Acc7 | patience=12 | warmup=15%"
echo "============================================================"

# =============================================================================
# 实验 A: Baseline — BERT-base-uncased
# =============================================================================
echo ""
echo ">>> [实验 A] Baseline: BERT-base-uncased <<<"

for SEED in "${SEEDS[@]}"; do
    echo "--- Seed ${SEED} ---"
    CUDA_VISIBLE_DEVICES=${GPU} python train.py \
        --dataset ${DATASET} \
        --seed ${SEED} \
        --epochs ${EPOCHS} \
        --batch_size ${BATCH_SIZE} \
        --bert_pretrained bert-base-uncased \
        --lr 3e-4 \
        --bert_lr 2e-5 \
        --dropout 0.3 \
        --ism_depth 1 \
        --v_self_ratio 0.3 \
        --sub_loss_lambda 0.3 \
        --warmup_ratio 0.15 \
        --grad_clip 0.3 \
        --key_eval MAE \
        --secondary_metric Acc7 \
        --exp_tag phase2v2_baseline_bert \
        2>&1 | tee logs/phase2v2_baseline_bert_seed${SEED}.log
    echo "  完成 seed=${SEED}"
done

# =============================================================================
# 实验 B: Phase 2 — RoBERTa-base (显式指定)
# =============================================================================
echo ""
echo ">>> [实验 B] Phase 2: RoBERTa-base <<<"

for SEED in "${SEEDS[@]}"; do
    echo "--- Seed ${SEED} ---"
    CUDA_VISIBLE_DEVICES=${GPU} python train.py \
        --dataset ${DATASET} \
        --seed ${SEED} \
        --epochs ${EPOCHS} \
        --batch_size ${BATCH_SIZE} \
        --bert_pretrained roberta-base \
        --lr 3e-4 \
        --bert_lr 2e-5 \
        --dropout 0.3 \
        --ism_depth 1 \
        --v_self_ratio 0.3 \
        --sub_loss_lambda 0.3 \
        --warmup_ratio 0.15 \
        --grad_clip 0.3 \
        --key_eval MAE \
        --secondary_metric Acc7 \
        --exp_tag phase2v2_roberta \
        2>&1 | tee logs/phase2v2_roberta_seed${SEED}.log
    echo "  完成 seed=${SEED}"
done

# =============================================================================
# 结果汇总
# =============================================================================
echo ""
echo "============================================================"
echo " 训练完成! 对比结果:"
echo "============================================================"
echo ""
printf "%-35s | %-8s | %-8s | %-8s | %-8s\n" "实验" "MAE" "Acc2" "Acc7" "Corr"
printf "%-35s-+-%-8s-+-%-8s-+-%-8s-+-%-8s\n" "-----------------------------------" "--------" "--------" "--------" "--------"

for SEED in "${SEEDS[@]}"; do
    for EXP in phase2v2_baseline_bert phase2v2_roberta; do
        F="results/${DATASET}_${EXP}_seed${SEED}_test.json"
        if [ -f "$F" ]; then
            VALS=$(python3 -c "
import json
with open('${F}') as f:
    d = json.load(f)
# Find the best entry (primary or secondary that has best MAE)
best = None
for k,v in d.items():
    if best is None or v.get('MAE',999) < best.get('MAE',999):
        best = v
if best:
    print(f\"{best.get('MAE','?'):.4f}|{best.get('Acc2','?'):.4f}|{best.get('Acc7','?'):.4f}|{best.get('Corr','?'):.4f}\")
")
            printf "%-35s | %-8s | %-8s | %-8s | %-8s\n" "${EXP}_seed${SEED}" $(echo $VALS | tr '|' ' ')
        else
            printf "%-35s | %-8s | %-8s | %-8s | %-8s\n" "${EXP}_seed${SEED}" "N/A" "N/A" "N/A" "N/A"
        fi
    done
done

echo ""
echo "文件位置:"
echo "  checkpoint: checkpoints/${DATASET}_phase2v2_*_best_*.pt"
echo "  results:    results/${DATASET}_phase2v2_*_test.json"
echo "  logs:       logs/phase2v2_*.log"
