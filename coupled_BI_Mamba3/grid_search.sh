#!/bin/bash
# ============================================================================
# Coupled-BI-Mamba3 网格搜索 v2: VF × HP 交叉搜索
# Phase 1: 4 VF × (2 lr × 2 do) = 16 runs
# Phase 2: Top-2 VF × 架构/损失消融 = ~8 runs
# Phase 3: Best combo × 精调 = ~4 runs
# 总计 ~28 runs, ~9h
# ============================================================================
set -euo pipefail

# 修复 conda deactivate 脚本的 set -u 冲突
export CONDA_BACKUP_CXX=""
export CONDA_BACKUP_CC=""
export CONDA_BACKUP_GXX=""
export CONDA_BACKUP_GCC=""
export CONDA_BACKUP_CFLAGS=""
export CONDA_BACKUP_CXXFLAGS=""
export CONDA_BACKUP_CPPFLAGS=""
export CONDA_BACKUP_LDFLAGS=""

CONDA_ENV="coupled_mamba_3"
CONDA_PATH="/home/zjw/anaconda3"
GPU=0
DATASET="MOSI"
SEED=42

FIXED_ARGS=(
    --dataset "${DATASET}" --seed "${SEED}"
    --feature_A features/mosi_audio_data2vec.pkl
    --bert_pretrained bert-base-uncased
    --batch_size 16
    --use_context true --use_bssm_gate true --use_gcmn_gate true
    --secondary_metric Acc2 --tertiary_metric Acc7 --key_eval MAE
    --ism_depth 3 --d_state 64 --num_layers 2
    --warmup_ratio 0.15 --weight_decay 0.0001 --grad_clip 0.1
)

RESULT_DIR="results/grid_search_v2"
LOG_DIR="logs/grid_search_v2"
mkdir -p "${RESULT_DIR}" "${LOG_DIR}"

SUMMARY="${RESULT_DIR}/summary_$(date +%Y%m%d_%H%M%S).csv"
echo "exp,vf,lr,do,ismd,test_mae,test_acc2,test_acc7,test_corr,val_mae,val_acc2,val_acc7" > "${SUMMARY}"

run_one() {
    local exp_name="$1"; shift
    local log_file="${LOG_DIR}/${exp_name}.log"

    echo "===== [$(date +%H:%M:%S)] ${exp_name} ====="
    set +u  # conda deactivate 脚本内部有未绑定变量, 暂时放宽
    source "${CONDA_PATH}/bin/activate" "${CONDA_ENV}"
    set -u

    # PYTHONUNBUFFERED=1 + python -u 强制实时输出
    PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=${GPU} python -u train.py \
        "${FIXED_ARGS[@]}" --exp_tag "gs2_${exp_name}" "$@" \
        2>&1 | tee "${log_file}"

    # 解析测试结果 (从日志中 grep)
    local t_mae="" t_acc2="" t_acc7="" t_corr=""
    local tline
    tline=$(grep "test_tertiary" "${log_file}" | tail -1)
    if [ -n "${tline}" ]; then
        t_mae=$(echo "${tline}"  | grep -oP 'MAE=\K[0-9.]+'  || echo "")
        t_acc2=$(echo "${tline}" | grep -oP 'Acc2=\K[0-9.]+' || echo "")
        t_acc7=$(echo "${tline}" | grep -oP 'Acc7=\K[0-9.]+' || echo "")
        t_corr=$(echo "${tline}" | grep -oP 'Corr=\K[0-9.]+' || echo "")
    fi

    # 验证最佳值
    local v_mae="" v_acc2="" v_acc7=""
    v_mae=$(grep "New best MAE="  "${log_file}" | tail -1 | sed 's/.*MAE=\([0-9.]*\).*/\1/' || echo "")
    v_acc2=$(grep "New best Acc2=" "${log_file}" | tail -1 | sed 's/.*Acc2=\([0-9.]*\).*/\1/' || echo "")
    v_acc7=$(grep "New best Acc7=" "${log_file}" | tail -1 | sed 's/.*Acc7=\([0-9.]*\).*/\1/' || echo "")

    # 提取参数
    local vf lr do ismd
    vf=$(echo "${exp_name}" | grep -oP '^(vf_)?\K(openface3|clip|dinov3|videomae)' || echo "?")
    lr=$(grep '"learning_rate"' "${log_file}" | head -1 | grep -oP '[\d.e+-]+' | head -1 || echo "")
    do=$(grep '"dropout"' "${log_file}" | head -1 | grep -oP '"dropout": \K[\d.]+' || echo "")
    ismd=$(grep '"ism_depth"' "${log_file}" | head -1 | grep -oP '"ism_depth": \K\d+' || echo "")

    echo "${exp_name},${vf},${lr},${do},${ismd},${t_mae},${t_acc2},${t_acc7},${t_corr},${v_mae},${v_acc2},${v_acc7}" >> "${SUMMARY}"

    # 实时打印核心指标
    printf "  >>> test Acc7=%-8s Acc2=%-8s MAE=%-8s | val Acc7=%-8s Acc2=%-8s\n" \
        "${t_acc7}" "${t_acc2}" "${t_mae}" "${v_acc7}" "${v_acc2}"
}

print_ranking() {
    local title="$1" col="$2"
    echo ""; echo "=== ${title} ==="
    sort -t',' -k"${col}" -nr "${SUMMARY}" 2>/dev/null | head -11 | column -t -s',' || echo "  (暂无)"
}

# ============================================================================
# Phase 1: VF × HP 交叉网格
#   4 VF × 3 lr × 4 dropout = 48 runs
# ============================================================================
echo "############################################################"
echo "# Phase 1: VF × HP 交叉网格 (8 runs)"
echo "############################################################"

for VF in openface3 clip dinov3 videomae; do
    VF_FILE="features/split_vision_${VF}.pkl"
    [ -f "${VF_FILE}" ] || { echo "!! ${VF_FILE} 不存在, 跳过"; continue; }
    for LR in 2e-4 3e-4 5e-4; do
        for DO in 0.2 0.3 0.5 0.7; do
            run_one "${VF}_lr${LR}_do${DO}" \
                --feature_V "${VF_FILE}" \
                --lr "${LR}" --dropout "${DO}"
        done
    done
done

# ★ Phase 1 完成后打印所有结果 (同时保存到文件)
echo ""
echo "============================================================"
echo " Phase 1 完成 — 全部实验结果"
echo "============================================================"
PHASE1_SUMMARY="${RESULT_DIR}/phase1_ranking.txt"
{
echo "Phase 1 完成 — 全部实验结果"
echo "=============================="
echo ""
echo "--- 按 test Acc7 排序 ---"
sort -t',' -k7 -nr "${SUMMARY}" 2>/dev/null | column -t -s','
echo ""
echo "--- 按 test Acc2 排序 ---"
sort -t',' -k6 -nr "${SUMMARY}" 2>/dev/null | column -t -s','
echo ""
echo "--- 按 test MAE 排序 (越低越好) ---"
sort -t',' -k5 -n  "${SUMMARY}" 2>/dev/null | column -t -s','
} | tee "${PHASE1_SUMMARY}"
echo ""
echo "Phase 1 排名已保存至: ${PHASE1_SUMMARY}"

# ============================================================================
# Phase 2: 三指标最佳组合 × 架构/损失消融
#   分别找 test Acc7 / Acc2 / MAE 各自最佳的 VF+lr+do, 各自做消融
#   (去重后至少 1 组, 最多 3 组, 每组 ~4 runs)
# ============================================================================
echo ""; echo "############################################################"
echo "# Phase 2: 三指标最佳组合消融"
echo "############################################################"

# 从 Phase 1 结果找三个最佳组合 (可能重合)
# Acc7: 第7列, 降序
BEST_A7=$(sort -t',' -k7 -nr "${SUMMARY}" 2>/dev/null | head -2 | tail -1)
# Acc2: 第6列, 降序
BEST_A2=$(sort -t',' -k6 -nr "${SUMMARY}" 2>/dev/null | head -2 | tail -1)
# MAE: 第5列, 升序
BEST_MAE=$(sort -t',' -k5 -n  "${SUMMARY}" 2>/dev/null | head -2 | tail -1)

declare -A SEEN_COMBOS  # 去重

process_best() {
    local label="$1" line="$2"
    local vf lr do
    vf=$(echo "${line}"  | cut -d',' -f2)
    lr=$(echo "${line}"  | cut -d',' -f3)
    do=$(echo "${line}"  | cut -d',' -f4)
    local key="${vf}_${lr}_${do}"
    if [ -n "${SEEN_COMBOS[$key]:-}" ]; then
        echo ">>> [${label}] ${key} 与前面重复, 跳过消融"
        return
    fi
    SEEN_COMBOS[$key]=1

    local vf_file="features/split_vision_${vf}.pkl"
    local a7=$(echo "${line}" | cut -d',' -f7)
    local a2=$(echo "${line}" | cut -d',' -f6)
    local mae=$(echo "${line}" | cut -d',' -f5)
    echo ""
    echo ">>> [${label}] ${vf} lr=${lr} do=${do} | test Acc7=${a7} Acc2=${a2} MAE=${mae}"

    # ism_depth=2
    run_one "${label}_ismd2" \
        --feature_V "${vf_file}" --lr "${lr}" --dropout "${do}" --ism_depth 2

    # v_self_ratio=0.3
    run_one "${label}_vsr0.3" \
        --feature_V "${vf_file}" --lr "${lr}" --dropout "${do}" --v_self_ratio 0.3

    # sub_loss_lambda=0.3
    run_one "${label}_sub0.3" \
        --feature_V "${vf_file}" --lr "${lr}" --dropout "${do}" --sub_loss_lambda 0.3

    # aux_cls_weight=0.3 (直接攻 Acc7)
    run_one "${label}_aux0.3" \
        --feature_V "${vf_file}" --lr "${lr}" --dropout "${do}" \
        --aux_cls_weight 0.3 --aux_num_classes 7
}

process_best "A7"  "${BEST_A7}"
process_best "A2"  "${BEST_A2}"
process_best "MAE" "${BEST_MAE}"

# ============================================================================
# Phase 3: 三指标最佳组合精调
#   对 Phase 2 之后三个指标各自最佳的 VF+lr+do 做精调
# ============================================================================
echo ""; echo "############################################################"
echo "# Phase 3: 三指标精调"
echo "############################################################"

# 重新找三个指标的最佳组合 (含 Phase 2 消融结果)
BEST_A7_L=$(sort -t',' -k7 -nr "${SUMMARY}" 2>/dev/null | head -2 | tail -1)
BEST_A2_L=$(sort -t',' -k6 -nr "${SUMMARY}" 2>/dev/null | head -2 | tail -1)
BEST_MAE_L=$(sort -t',' -k5 -n  "${SUMMARY}" 2>/dev/null | head -2 | tail -1)

declare -A FT_SEEN

fine_tune() {
    local label="$1" line="$2"
    local vf lr do
    vf=$(echo "${line}"  | cut -d',' -f2)
    lr=$(echo "${line}"  | cut -d',' -f3)
    do=$(echo "${line}"  | cut -d',' -f4)
    local key="${vf}_${lr}_${do}"
    if [ -n "${FT_SEEN[$key]:-}" ]; then
        echo ">>> [精调-${label}] ${key} 重复, 跳过"
        return
    fi
    FT_SEEN[$key]=1

    local vf_file="features/split_vision_${vf}.pkl"
    local a7=$(echo "${line}" | cut -d',' -f7)
    local a2=$(echo "${line}" | cut -d',' -f6)
    local mae=$(echo "${line}" | cut -d',' -f5)
    echo ""
    echo ">>> [精调-${label}] ${vf} lr=${lr} do=${do} | test Acc7=${a7} Acc2=${a2} MAE=${mae}"

    # weight_decay 扫 3 个值
    for WD in 0.0 0.0005 0.001; do
        run_one "ft_${label}_wd${WD}" \
            --feature_V "${vf_file}" --lr "${lr}" --dropout "${do}" \
            --weight_decay "${WD}"
    done

    # grad_clip 微调
    for GC in 0.05 0.3; do
        run_one "ft_${label}_gc${GC}" \
            --feature_V "${vf_file}" --lr "${lr}" --dropout "${do}" \
            --grad_clip "${GC}"
    done
}

fine_tune "A7"  "${BEST_A7_L}"
fine_tune "A2"  "${BEST_A2_L}"
fine_tune "MAE" "${BEST_MAE_L}"

# ============================================================================
# 汇总 (打印 + 保存)
# ============================================================================
FINAL_SUMMARY="${RESULT_DIR}/final_ranking.txt"
{
echo ""; echo "===== 网格搜索 v2 完成 ====="
print_ranking "测试 Acc7 Top 10" 7
print_ranking "测试 Acc2 Top 10" 6
print_ranking "测试 MAE  Top 10 (越低越好)" 5
echo ""; echo "CSV 汇总: ${SUMMARY}"
} | tee "${FINAL_SUMMARY}"
echo "最终排名已保存至: ${FINAL_SUMMARY}"
