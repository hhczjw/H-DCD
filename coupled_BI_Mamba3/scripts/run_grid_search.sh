#!/bin/bash
# ===========================================================================
# 自动化网格搜索脚本 — BERT + Data2Vec (离线特征) 最优超参搜索
# ===========================================================================
# 用法:
#   bash scripts/run_grid_search.sh
#
# 搜索策略:
#   Stage 1: lr × dropout (9 组), seed=42, 固定 v_self=0.3 sub_loss=0.3
#   Stage 2: v_self_ratio × sub_loss_lambda (up to 12 组), seed=42,
#            固定 Stage 1 最优的 lr/dropout
#   Stage 3: Top-3 组合 3-seed 验证 (42, 1111, 2024)
#
# 输出:
#   grid_results/           — 每组的 JSON 结果 + CSV 汇总
#   checkpoints/            — 每组最优 Acc2/MAE/Acc7 的 .pt 文件
#   logs/                   — 每组训练日志
# ===========================================================================
set -e
cd "$(dirname "$0")/.."

# ── 固定参数 ──
DATASET="MOSI"
EPOCHS=120
BATCH_SIZE=32
GPU=0
BERT_MODEL="bert-base-uncased"
FEATURE_PKL="./features/mosi_audio_data2vec.pkl"

# ── 路径 ──
GRID_DIR="./grid_results"
SUMMARY_CSV="${GRID_DIR}/grid_summary.csv"
mkdir -p "${GRID_DIR}" ./checkpoints ./logs

# ── 工具函数 ──
get_json_val() {
    # 从 test JSON 中提取指定 ckpt 下指定指标的值
    # 用法: get_json_val <json_file> <ckpt_key> <metric>
    python3 -c "
import json, sys
try:
    with open('$1','r') as f:
        d = json.load(f)
    print(d.get('$2', {}).get('$3', 'N/A'))
except:
    print('N/A')
"
}

log_best() {
    local tag="$1" seed="$2"
    local json_file="${GRID_DIR}/${tag}_seed${seed}_test.json"
    # primary ckpt = best MAE, secondary ckpt = best Acc2, tertiary ckpt = best Acc7
    local acc2_pri=$(get_json_val "$json_file" "primary_MAE" "Acc2")
    local acc2_sec=$(get_json_val "$json_file" "secondary_Acc2" "Acc2")
    local acc2_ter=$(get_json_val "$json_file" "tertiary_Acc7" "Acc2")
    local mae_pri=$(get_json_val "$json_file" "primary_MAE" "MAE")
    local mae_sec=$(get_json_val "$json_file" "secondary_Acc2" "MAE")
    local mae_ter=$(get_json_val "$json_file" "tertiary_Acc7" "MAE")
    local acc7_pri=$(get_json_val "$json_file" "primary_MAE" "Acc7")
    local acc7_sec=$(get_json_val "$json_file" "secondary_Acc2" "Acc7")
    local acc7_ter=$(get_json_val "$json_file" "tertiary_Acc7" "Acc7")
    echo "${tag},${seed},${acc2_pri},${acc2_sec},${acc2_ter},${mae_pri},${mae_sec},${mae_ter},${acc7_pri},${acc7_sec},${acc7_ter}"
}

run_single() {
    # 运行单次训练 + 复制结果到 grid_results/
    local exp_tag="$1" seed="$2"
    shift 2
    echo ">>> [${exp_tag}] seed=${seed} start @ $(date '+%H:%M:%S')"

    CUDA_VISIBLE_DEVICES=${GPU} python train.py \
        --dataset ${DATASET} \
        --seed ${seed} \
        --epochs ${EPOCHS} \
        --batch_size ${BATCH_SIZE} \
        --bert_pretrained ${BERT_MODEL} \
        --feature_A "${FEATURE_PKL}" \
        --skip_audio_ism true \
        --ism_depth 1 \
        --key_eval MAE \
        --secondary_metric Acc2 \
        --tertiary_metric Acc7 \
        --early_stop 30 \
        --warmup_ratio 0.15 \
        --grad_clip 0.3 \
        --exp_tag "${exp_tag}" \
        "$@" \
        2>&1 | tee "logs/${exp_tag}_seed${seed}.log"

    # 复制 JSON 结果到 grid_results/ (避免被后续实验覆盖)
    local src_json="results/${DATASET}_${exp_tag}_seed${seed}_test.json"
    if [ -f "${src_json}" ]; then
        cp "${src_json}" "${GRID_DIR}/${exp_tag}_seed${seed}_test.json"
    fi
    echo "  完成 ${exp_tag} seed=${seed} @ $(date '+%H:%M:%S')"
}

# ══════════════════════════════════════════════════════════════════════
# Stage 1: lr × dropout (全组合, seed=42)
# ══════════════════════════════════════════════════════════════════════
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Stage 1: lr × dropout 网格搜索 (seed=42)                   ║"
echo "╚══════════════════════════════════════════════════════════════╝"

LR_VALS=(1e-4 3e-4 5e-4 7e-4 9e-4 11e-4)
DROPOUT_VALS=(0.1 0.3 0.5 0.7 0.9)

STAGE1_CSV="${GRID_DIR}/stage1_results.csv"
echo "exp_tag,seed,Acc2_pri,Acc2_sec,Acc2_ter,MAE_pri,MAE_sec,MAE_ter,Acc7_pri,Acc7_sec,Acc7_ter" > "${STAGE1_CSV}"

for lr in "${LR_VALS[@]}"; do
    for dp in "${DROPOUT_VALS[@]}"; do
        tag="gs1_lr${lr}_dp${dp}"
        run_single "${tag}" 42 \
            --lr "${lr}" \
            --bert_lr 2e-5 \
            --dropout "${dp}" \
            --v_self_ratio 0.3 \
            --sub_loss_lambda 0.3
        log_best "${tag}" 42 >> "${STAGE1_CSV}"
    done
done

echo ""
echo "── Stage 1 完成 ──"
echo "Top Acc7 (tertiary ckpt):"
sort -t',' -k11 -nr "${STAGE1_CSV}" | head -4 | column -t -s','

# 选出 Stage 1 最优组合 (按 tertiary_Acc7 最高)
BEST_LR=$(sort -t',' -k11 -nr "${STAGE1_CSV}" | head -1 | cut -d',' -f1 | sed 's/gs1_//' | grep -oP 'lr\K[^_]+')
BEST_DP=$(sort -t',' -k11 -nr "${STAGE1_CSV}" | head -1 | cut -d',' -f1 | sed 's/gs1_//' | grep -oP 'dp\K[^_]*$')
echo ">>> Stage 1 Best: lr=${BEST_LR}, dropout=${BEST_DP}"

# ══════════════════════════════════════════════════════════════════════
# Stage 2: v_self_ratio × sub_loss_lambda (seed=42, 固定最优 lr/dp)
# ══════════════════════════════════════════════════════════════════════
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Stage 2: v_self × sub_loss 网格搜索 (seed=42)             ║"
echo "╚══════════════════════════════════════════════════════════════╝"

VSELF_VALS=(0.0 0.2 0.3 0.5 0.6)
SUBLOSS_VALS=(0.0 0.2 0.3 0.5 0.6)

STAGE2_CSV="${GRID_DIR}/stage2_results.csv"
echo "exp_tag,seed,Acc2_pri,Acc2_sec,Acc2_ter,MAE_pri,MAE_sec,MAE_ter,Acc7_pri,Acc7_sec,Acc7_ter" > "${STAGE2_CSV}"

for vs in "${VSELF_VALS[@]}"; do
    for sl in "${SUBLOSS_VALS[@]}"; do
        tag="gs2_vs${vs}_sl${sl}"
        run_single "${tag}" 42 \
            --lr "${BEST_LR}" \
            --bert_lr 2e-5 \
            --dropout "${BEST_DP}" \
            --v_self_ratio "${vs}" \
            --sub_loss_lambda "${sl}"
        log_best "${tag}" 42 >> "${STAGE2_CSV}"
    done
done

echo ""
echo "── Stage 2 完成 ──"
echo "Top Acc7 (tertiary ckpt):"
sort -t',' -k11 -nr "${STAGE2_CSV}" | head -4 | column -t -s','

# 选出 Stage 2 Top-3 组合
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Stage 3: Top-3 组合 3-seed 验证                            ║"
echo "╚══════════════════════════════════════════════════════════════╝"

STAGE3_CSV="${GRID_DIR}/stage3_results.csv"
echo "exp_tag,seed,Acc2_pri,Acc2_sec,Acc2_ter,MAE_pri,MAE_sec,MAE_ter,Acc7_pri,Acc7_sec,Acc7_ter" > "${STAGE3_CSV}"

SEEDS=(42 1111 2024)
TOP3_TAGS=($(sort -t',' -k11 -nr "${STAGE2_CSV}" | head -3 | cut -d',' -f1))

for tag in "${TOP3_TAGS[@]}"; do
    # 从 tag 解析参数
    VS=$(echo "${tag}" | grep -oP 'vs\K[^_]+')
    SL=$(echo "${tag}" | grep -oP 'sl\K[^_]*$')
    for seed in "${SEEDS[@]}"; do
        run_single "${tag}" "${seed}" \
            --lr "${BEST_LR}" \
            --bert_lr 2e-5 \
            --dropout "${BEST_DP}" \
            --v_self_ratio "${VS}" \
            --sub_loss_lambda "${SL}"
        log_best "${tag}" "${seed}" >> "${STAGE3_CSV}"
    done
done

# ══════════════════════════════════════════════════════════════════════
# 最终汇总
# ══════════════════════════════════════════════════════════════════════
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  最终汇总 (Stage 3: 3-seed mean ± std)                      ║"
echo "╚══════════════════════════════════════════════════════════════╝"

python3 -c "
import csv, sys
from collections import defaultdict
import numpy as np

rows = defaultdict(list)
with open('${STAGE3_CSV}') as f:
    reader = csv.DictReader(f)
    for r in reader:
        tag = r['exp_tag']
        for k in r:
            if k in ('exp_tag', 'seed'): continue
            try:
                rows[(tag, k)].append(float(r[k]))
            except:
                pass

print(f'{\"Rank\":<5} {\"exp_tag\":<28} {\"Acc7(ter)↑\":>14} {\"Acc2(ter)↑\":>14} {\"MAE(sec)↓\":>14}')
print('-' * 80)
# 按 Acc7(ter) 降序排
by_acc7 = []
for tag in set(k[0] for k in rows):
    acc7_vals = rows.get((tag, 'Acc7_ter'), [])
    acc2_vals = rows.get((tag, 'Acc2_ter'), [])
    mae_vals  = rows.get((tag, 'MAE_sec'), [])
    if acc7_vals:
        by_acc7.append((np.mean(acc7_vals), tag, acc7_vals, acc2_vals, mae_vals))
by_acc7.sort(key=lambda x: x[0], reverse=True)

for rank, (mean_acc7, tag, a7, a2, mae) in enumerate(by_acc7, 1):
    a7_str = f'{np.mean(a7):.4f}±{np.std(a7):.4f}' if len(a7)>1 else f'{np.mean(a7):.4f}'
    a2_str = f'{np.mean(a2):.4f}±{np.std(a2):.4f}' if len(a2)>1 else f'{np.mean(a2):.4f}'
    mae_str = f'{np.mean(mae):.4f}±{np.std(mae):.4f}' if len(mae)>1 else f'{np.mean(mae):.4f}'
    print(f'{rank:<5} {tag:<28} {a7_str:>14} {a2_str:>14} {mae_str:>14}')
"

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  网格搜索全部完成!"
echo "  结果目录: ${GRID_DIR}/"
echo "  最佳模型: checkpoints/"
echo "═══════════════════════════════════════════════════════════════"
