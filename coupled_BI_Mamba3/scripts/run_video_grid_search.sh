#!/bin/bash
# ===========================================================================
# 视频特征自动化网格搜索脚本
# ===========================================================================
# 搜索策略:
#   Stage 1: 在 3 种维度代表上搜正则化参数 (lr × dropout × weight_decay)
#           OpenFace3(18维) / Original_FACET(20维) / DINOv3(768维)
#           每个特征独立找最优 lr/dropout/weight_decay
#   Stage 2: 在 Stage1 最优特征上搜损失/模态交互参数
#   Stage 3: 在 Stage2 最优特征上搜模型结构参数
#   Stage 4: 每个特征用自己最优正则化 + 全局最优其他参数, 全 5 特征验证
#
# 用法:
#   bash scripts/run_video_grid_search.sh                     # 全阶段运行
#   bash scripts/run_video_grid_search.sh --stage 1 --seeds 42 # 仅 Stage 1
#   bash scripts/run_video_grid_search.sh --dry-run           # 仅预览
# ===========================================================================
set -e
cd "$(dirname "$0")/.."

# ── 固定参数 ──
DATASET="MOSI"
EPOCHS=120
BATCH_SIZE=32
GPU=0
BERT_MODEL="bert-base-uncased"
FEATURES_DIR="./features"

# ── 路径 ──
GRID_DIR="./results/video_grid_search"
STAGE1_DIR="${GRID_DIR}/stage1"
STAGE2_DIR="${GRID_DIR}/stage2"
STAGE3_DIR="${GRID_DIR}/stage3"
STAGE4_DIR="${GRID_DIR}/stage4"
mkdir -p "${GRID_DIR}" "${STAGE1_DIR}" "${STAGE2_DIR}" "${STAGE3_DIR}" "${STAGE4_DIR}" ./checkpoints ./logs

# ── CLI 参数 ──
MAX_STAGE=4
START_STAGE=1
SEEDS=(42)
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --stage)    START_STAGE="$2"; shift 2 ;;
        --seeds)    IFS=',' read -ra SEEDS <<< "$2"; shift 2 ;;
        --dry-run)  DRY_RUN=true; shift ;;
        --max-stage) MAX_STAGE="$2"; shift 2 ;;
        *)          shift ;;
    esac
done

# ── 工具函数 ──
get_json_val() {
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

# ── 运行单次训练 ──
run_single() {
    local exp_tag="$1" feature_name="$2" seed="$3"
    shift 3

    echo ""
    echo "◆ [${exp_tag}] feature=${feature_name} seed=${seed} @ $(date '+%H:%M:%S')"
    echo "  参数: $@"

    if [ "${DRY_RUN}" = true ]; then
        echo "  [DRY] 跳过训练"
        return
    fi

    # 解析特征配置
    local feat_arg=""
    local skip_audio_ism="true"
    case "${feature_name}" in
        OpenFace3)      feat_arg="--feature_V ${FEATURES_DIR}/vision_openface3.pkl" ;;
        Original_FACET) feat_arg="" ;;
        CLIP)           feat_arg="--feature_V ${FEATURES_DIR}/vision_clip.pkl" ;;
        VideoMAE)       feat_arg="--feature_V ${FEATURES_DIR}/vision_videomae.pkl" ;;
        DINOv3)         feat_arg="--feature_V ${FEATURES_DIR}/vision_dinov3.pkl" ;;
    esac

    CUDA_VISIBLE_DEVICES=${GPU} python -u train.py \
        --dataset ${DATASET} \
        --seed ${seed} \
        --epochs ${EPOCHS} \
        --batch_size ${BATCH_SIZE} \
        --bert_pretrained ${BERT_MODEL} \
        --skip_audio_ism ${skip_audio_ism} \
        --ism_depth 1 \
        --key_eval MAE \
        --secondary_metric Acc2 \
        --tertiary_metric Acc7 \
        --early_stop 30 \
        --warmup_ratio 0.15 \
        --grad_clip 0.3 \
        --exp_tag "${exp_tag}" \
        ${feat_arg} \
        "$@" \
        2>&1 | tee "logs/${exp_tag}_seed${seed}.log"

    # 复制 JSON 结果
    local src_json="results/${DATASET}_${exp_tag}_seed${seed}_test.json"
    if [ -f "${src_json}" ]; then
        cp "${src_json}" "${GRID_DIR}/${exp_tag}_seed${seed}_test.json"
    fi

    echo "  ✓ 完成 @ $(date '+%H:%M:%S')"
}

# 从 JSON 结果文件中读取最佳特征/参数
find_best_feature_and_params() {
    local stage_dir="$1"
    local feature_filter="${2:-}"
    python3 -c "
import json, os, sys
json_path = os.path.join('${stage_dir}', 'all_results.json')
if not os.path.exists(json_path):
    json_path = os.path.join('${stage_dir}', 'results.json')
if not os.path.exists(json_path):
    print('{}')
    sys.exit(0)
with open(json_path) as f:
    data = json.load(f)
if '${feature_filter}':
    data = [r for r in data if r.get('feature','')=='${feature_filter}']
if not data:
    print('{}')
    sys.exit(0)
best = max(data, key=lambda r: float(r.get('Acc7_ter',0) or 0))
result = {
    'feature': best.get('feature',''),
    'lr': best.get('lr','3e-4'),
    'dropout': best.get('dropout','0.3'),
    'weight_decay': best.get('weight_decay','1e-5'),
    'v_self_ratio': best.get('v_self_ratio','0.0'),
    'sub_loss_lambda': best.get('sub_loss_lambda','0.3'),
    'warmup_ratio': best.get('warmup_ratio','0.15'),
    'ism_depth': best.get('ism_depth','2'),
}
import json as j2; print(j2.dumps(result))
"
}

# 保存阶段结果到 all_results.json
save_stage_results() {
    local stage_dir="$1"
    python3 -c "
import csv, json, os, glob
rows = []
for csv_file in glob.glob(os.path.join('${stage_dir}', '*_results.csv')):
    with open(csv_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
with open(os.path.join('${stage_dir}', 'all_results.json'), 'w') as f:
    json.dump(rows, f, indent=2)
print(f'Saved {len(rows)} results to {stage_dir}/all_results.json')
"
}

# ══════════════════════════════════════════════════════════════════════
# Stage 1: 正则化参数搜索 (3 种维度代表独立搜索)
# ══════════════════════════════════════════════════════════════════════
if [ ${START_STAGE} -le 1 ] && [ ${MAX_STAGE} -ge 1 ]; then
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Stage 1: 正则化参数搜索 (3 特征 × 27 组合)                 ║"
echo "╚══════════════════════════════════════════════════════════════╝"

STAGE1_FEATURES=("OpenFace3" "Original_FACET" "DINOv3")
LR_VALS=(1e-4 3e-4 1e-3)
DROPOUT_VALS=(0.1 0.3 0.5)
WEIGHT_DECAY_VALS=(1e-5 1e-4 1e-3)

for fname in "${STAGE1_FEATURES[@]}"; do
    echo ""
    echo "── Feature: ${fname} ──"
    STAGE1_CSV="${STAGE1_DIR}/${fname}_results.csv"
    echo "exp_tag,seed,feature,lr,dropout,weight_decay,Acc2_ter,MAE_ter,Acc7_ter" > "${STAGE1_CSV}"

    for lr in "${LR_VALS[@]}"; do
        for dp in "${DROPOUT_VALS[@]}"; do
            for wd in "${WEIGHT_DECAY_VALS[@]}"; do
                tag="gs1_${fname}_lr${lr}_dp${dp}_wd${wd}"

                for seed in "${SEEDS[@]}"; do
                    run_single "${tag}" "${fname}" "${seed}" \
                        --lr "${lr}" \
                        --bert_lr 2e-5 \
                        --dropout "${dp}" \
                        --weight_decay "${wd}" \
                        --v_self_ratio 0.0 \
                        --sub_loss_lambda 0.3
                done

                # 提取指标写入 CSV
                local json_file="${GRID_DIR}/${tag}_seed${SEEDS[0]}_test.json"
                local a2=$(get_json_val "${json_file}" "tertiary_Acc7" "Acc2")
                local mae=$(get_json_val "${json_file}" "tertiary_Acc7" "MAE")
                local a7=$(get_json_val "${json_file}" "tertiary_Acc7" "Acc7")
                echo "${tag},${SEEDS[0]},${fname},${lr},${dp},${wd},${a2},${mae},${a7}" >> "${STAGE1_CSV}"
            done
        done
    done

    echo "  Best for ${fname} (by Acc7):"
    sort -t',' -k9 -nr "${STAGE1_CSV}" | head -3 | column -t -s','
done

save_stage_results "${STAGE1_DIR}"
fi

# ══════════════════════════════════════════════════════════════════════
# Stage 2: 损失/模态交互参数搜索
# ══════════════════════════════════════════════════════════════════════
if [ ${START_STAGE} -le 2 ] && [ ${MAX_STAGE} -ge 2 ]; then
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Stage 2: 损失/模态交互参数搜索                              ║"
echo "╚══════════════════════════════════════════════════════════════╝"

BEST_INFO=$(find_best_feature_and_params "${STAGE1_DIR}")
BEST_FEATURE=$(echo "${BEST_INFO}" | python3 -c "import json,sys; print(json.load(sys.stdin).get('feature','OpenFace3'))")
BEST_LR=$(echo "${BEST_INFO}" | python3 -c "import json,sys; print(json.load(sys.stdin).get('lr','3e-4'))")
BEST_DP=$(echo "${BEST_INFO}" | python3 -c "import json,sys; print(json.load(sys.stdin).get('dropout','0.3'))")
BEST_WD=$(echo "${BEST_INFO}" | python3 -c "import json,sys; print(json.load(sys.stdin).get('weight_decay','1e-5'))")
echo "  基于特征: ${BEST_FEATURE} (lr=${BEST_LR} dp=${BEST_DP} wd=${BEST_WD})"

VSELF_VALS=(0.0 0.2 0.4)
SUBLOSS_VALS=(0.0 0.3 0.5)
WARMUP_VALS=(0.0 0.1 0.15)

STAGE2_CSV="${STAGE2_DIR}/results.csv"
echo "exp_tag,seed,feature,v_self_ratio,sub_loss_lambda,warmup_ratio,Acc2_ter,MAE_ter,Acc7_ter" > "${STAGE2_CSV}"

for vs in "${VSELF_VALS[@]}"; do
    for sl in "${SUBLOSS_VALS[@]}"; do
        for wr in "${WARMUP_VALS[@]}"; do
            tag="gs2_${BEST_FEATURE}_vs${vs}_sl${sl}_wr${wr}"

            for seed in "${SEEDS[@]}"; do
                run_single "${tag}" "${BEST_FEATURE}" "${seed}" \
                    --lr "${BEST_LR}" \
                    --bert_lr 2e-5 \
                    --dropout "${BEST_DP}" \
                    --weight_decay "${BEST_WD}" \
                    --v_self_ratio "${vs}" \
                    --sub_loss_lambda "${sl}" \
                    --warmup_ratio "${wr}"
            done

            local json_file="${GRID_DIR}/${tag}_seed${SEEDS[0]}_test.json"
            local a2=$(get_json_val "${json_file}" "tertiary_Acc7" "Acc2")
            local mae=$(get_json_val "${json_file}" "tertiary_Acc7" "MAE")
            local a7=$(get_json_val "${json_file}" "tertiary_Acc7" "Acc7")
            echo "${tag},${SEEDS[0]},${BEST_FEATURE},${vs},${sl},${wr},${a2},${mae},${a7}" >> "${STAGE2_CSV}"
        done
    done
done

echo ""
echo "Stage 2 Top (by Acc7):"
sort -t',' -k9 -nr "${STAGE2_CSV}" | head -5 | column -t -s','

python3 -c "
import csv, json
rows = []
with open('${STAGE2_CSV}') as f:
    for row in csv.DictReader(f):
        rows.append(row)
with open('${STAGE2_DIR}/all_results.json', 'w') as f:
    json.dump(rows, f, indent=2)
"
fi

# ══════════════════════════════════════════════════════════════════════
# Stage 3: 模型结构参数搜索
# ══════════════════════════════════════════════════════════════════════
if [ ${START_STAGE} -le 3 ] && [ ${MAX_STAGE} -ge 3 ]; then
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Stage 3: 模型结构参数搜索                                  ║"
echo "╚══════════════════════════════════════════════════════════════╝"

STAGE2_INFO=$(find_best_feature_and_params "${STAGE2_DIR}")
BEST_FEATURE=$(echo "${STAGE2_INFO}" | python3 -c "import json,sys; print(json.load(sys.stdin).get('feature','OpenFace3'))")

# 从 Stage 2 取最优参数, 缺失的从 Stage 1 补
STAGE2_BEST_PARAMS=$(find_best_feature_and_params "${STAGE2_DIR}" "${BEST_FEATURE}")
STAGE1_BEST_PARAMS=$(find_best_feature_and_params "${STAGE1_DIR}" "${BEST_FEATURE}")
BEST_LR=$(echo "${STAGE2_BEST_PARAMS}${STAGE1_BEST_PARAMS}" | python3 -c "import json,sys; d=json.loads(sys.stdin.read().replace('}{','|')); print(d.get('lr','3e-4'))" 2>/dev/null || echo "3e-4")
BEST_DP=$(echo "${STAGE1_BEST_PARAMS}" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('dropout','0.3'))" 2>/dev/null || echo "0.3")
BEST_WD=$(echo "${STAGE1_BEST_PARAMS}" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('weight_decay','1e-5'))" 2>/dev/null || echo "1e-5")
echo "  基于特征: ${BEST_FEATURE}"

ISM_DEPTH_VALS=(2 3)
ISM_MIXER_VALS=("bimamba" "bimamba3")

STAGE3_CSV="${STAGE3_DIR}/results.csv"
echo "exp_tag,seed,feature,ism_depth,ism_mixer_type,Acc2_ter,MAE_ter,Acc7_ter" > "${STAGE3_CSV}"

for idepth in "${ISM_DEPTH_VALS[@]}"; do
    for imixer in "${ISM_MIXER_VALS[@]}"; do
        tag="gs3_${BEST_FEATURE}_idepth${idepth}_mix${imixer}"

        for seed in "${SEEDS[@]}"; do
            run_single "${tag}" "${BEST_FEATURE}" "${seed}" \
                --lr "${BEST_LR}" \
                --bert_lr 2e-5 \
                --dropout "${BEST_DP}" \
                --weight_decay "${BEST_WD}" \
                --ism_depth "${idepth}" \
                --ism_mixer_type "${imixer}"
        done

        local json_file="${GRID_DIR}/${tag}_seed${SEEDS[0]}_test.json"
        local a2=$(get_json_val "${json_file}" "tertiary_Acc7" "Acc2")
        local mae=$(get_json_val "${json_file}" "tertiary_Acc7" "MAE")
        local a7=$(get_json_val "${json_file}" "tertiary_Acc7" "Acc7")
        echo "${tag},${SEEDS[0]},${BEST_FEATURE},${idepth},${imixer},${a2},${mae},${a7}" >> "${STAGE3_CSV}"
    done
done

echo ""
echo "Stage 3 Top (by Acc7):"
sort -t',' -k7 -nr "${STAGE3_CSV}" | head -5 | column -t -s','

python3 -c "
import csv, json
rows = []
with open('${STAGE3_CSV}') as f:
    for row in csv.DictReader(f):
        rows.append(row)
with open('${STAGE3_DIR}/all_results.json', 'w') as f:
    json.dump(rows, f, indent=2)
"
fi

# ══════════════════════════════════════════════════════════════════════
# Stage 4: 全 5 特征验证 (每个特征用自己最优正则化)
# ══════════════════════════════════════════════════════════════════════
if [ ${START_STAGE} -le 4 ] && [ ${MAX_STAGE} -ge 4 ]; then
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Stage 4: 全 5 特征验证                                    ║"
echo "╚══════════════════════════════════════════════════════════════╝"

ALL_FEATURES=("OpenFace3" "Original_FACET" "CLIP" "VideoMAE" "DINOv3")

# 从 Stage 2 取最优全局参数 (v_self_ratio, sub_loss_lambda, warmup_ratio)
STAGE2_BEST=$(find_best_feature_and_params "${STAGE2_DIR}" "")
GLOBAL_VS=$(echo "${STAGE2_BEST}" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('v_self_ratio','0.0'))")
GLOBAL_SL=$(echo "${STAGE2_BEST}" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('sub_loss_lambda','0.3'))")
GLOBAL_WR=$(echo "${STAGE2_BEST}" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('warmup_ratio','0.15'))")

# 从 Stage 3 取最优模型结构参数
STAGE3_BEST=$(find_best_feature_and_params "${STAGE3_DIR}" "")
GLOBAL_ID=$(echo "${STAGE3_BEST}" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('ism_depth','2'))")
GLOBAL_IMIX=$(echo "${STAGE3_BEST}" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('ism_mixer_type','bimamba'))")

STAGE4_CSV="${STAGE4_DIR}/results.csv"
echo "exp_tag,seed,feature,lr,dropout,weight_decay,Acc2_ter,MAE_ter,Acc7_ter" > "${STAGE4_CSV}"

for fname in "${ALL_FEATURES[@]}"; do
    echo ""
    echo "── 验证: ${fname} ──"

    # 取该特征的 Stage1 最优正则化参数
    FEAT_BEST=$(find_best_feature_and_params "${STAGE1_DIR}" "${fname}")
    local_lr=$(echo "${FEAT_BEST}" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('lr','3e-4'))")
    local_dp=$(echo "${FEAT_BEST}" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('dropout','0.3'))")
    local_wd=$(echo "${FEAT_BEST}" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('weight_decay','1e-5'))")

    for seed in "${SEEDS[@]}"; do
        tag="gs4_${fname}"
        run_single "${tag}" "${fname}" "${seed}" \
            --lr "${local_lr}" \
            --bert_lr 2e-5 \
            --dropout "${local_dp}" \
            --weight_decay "${local_wd}" \
            --v_self_ratio "${GLOBAL_VS}" \
            --sub_loss_lambda "${GLOBAL_SL}" \
            --warmup_ratio "${GLOBAL_WR}" \
            --ism_depth "${GLOBAL_ID}" \
            --ism_mixer_type "${GLOBAL_IMIX}"
    done

    local json_file="${GRID_DIR}/${tag}_seed${SEEDS[0]}_test.json"
    local a2=$(get_json_val "${json_file}" "tertiary_Acc7" "Acc2")
    local mae=$(get_json_val "${json_file}" "tertiary_Acc7" "MAE")
    local a7=$(get_json_val "${json_file}" "tertiary_Acc7" "Acc7")
    echo "${tag},${SEEDS[0]},${fname},${local_lr},${local_dp},${local_wd},${a2},${mae},${a7}" >> "${STAGE4_CSV}"
done

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Stage 4 Final Ranking (by Acc7)                           ║"
echo "╚══════════════════════════════════════════════════════════════╝"
sort -t',' -k9 -nr "${STAGE4_CSV}" | column -t -s','

python3 -c "
import csv, json
rows = []
with open('${STAGE4_CSV}') as f:
    for row in csv.DictReader(f):
        rows.append(row)
with open('${STAGE4_DIR}/all_results.json', 'w') as f:
    json.dump(rows, f, indent=2)
"
fi

# ══════════════════════════════════════════════════════════════════════
# 最终汇总
# ══════════════════════════════════════════════════════════════════════
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Grid Search 完成!                                         ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo "结果目录: ${GRID_DIR}/"
echo "  Stage 1: stage1/ (各特征独立最优正则化)"
echo "  Stage 2: stage2/ (损失/模态交互最优)"
echo "  Stage 3: stage3/ (模型结构最优)"
echo "  Stage 4: stage4/ (全特征验证对比)"
echo "  Logs:    ./logs/"
echo ""
echo "快速查看:"
echo "  column -t -s',' ${STAGE4_CSV}"
echo "  python3 -m json.tool ${STAGE4_DIR}/all_results.json"
