#!/bin/bash
# ============================================================
# P0 重跑队列 (2026-09-18)
# ------------------------------------------------------------
# 目标: 重跑 3 组顶配 × 3 seed, 收集第四 ckpt (best F1_has0) —— P0-1
#       全部跑完后自动执行二分类阈值校准 (3-seed 集成) —— P0-2
#
# 任务清单 (历史网格结论):
#   组 A (p0A): do0.5 / lr4e-4 / id2 / sl0.1   → 历史 best Acc7 (0.5350) & best has0 (0.8848)
#   组 B (p0B): do0.6 / lr4e-4 / id1 / sl0.3   → 历史 best Acc2 (0.9024)
#   组 D (p0D): do0.6 / lr5e-4 / id2 / sl0.1   → 导师引用综合配置 (0.8979/0.5277)
#
# 硬件: GPU0 常驻占用 ~1.1GB+碎片, 实测该配置 optimizer.step 会 OOM;
#       全部任务在 GPU1 串行执行 (单任务 ~1.5h, 总计 ~10-13h, 自动早停可缩短)。
#
# 启动方式 (持久化, 脱离终端):
#   setsid nohup bash scripts/run_p0_rerun.sh > logs/p0_rerun_master.out 2>&1 < /dev/null &
#
# 监控:
#   tail -f logs/p0_rerun_master.out
#   完成标志: logs/p0_rerun.done 出现
# ============================================================
set -u
cd "$(dirname "$0")/.."
# conda 激活需要容忍未绑定变量 (deactivate 钩子在 set -u 下会报错)
set +u
source /home/zjw/anaconda3/etc/profile.d/conda.sh
conda activate coupled_mamba_3
set -u
export PYTHONPATH="$PWD/mamba:$PWD"
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs checkpoints results

FEAT_A="features/mosi_audio_data2vec_full.pkl"
FEAT_V="features/split_vision_openface3.pkl"

# tag|seed|dropout|lr|ism_depth|sub_loss_lambda
JOBS=(
"p0A|42|0.5|4e-4|2|0.1"
"p0A|1111|0.5|4e-4|2|0.1"
"p0A|2024|0.5|4e-4|2|0.1"
"p0B|42|0.6|4e-4|1|0.3"
"p0B|1111|0.6|4e-4|1|0.3"
"p0B|2024|0.6|4e-4|1|0.3"
"p0D|42|0.6|5e-4|2|0.1"
"p0D|1111|0.6|5e-4|2|0.1"
"p0D|2024|0.6|5e-4|2|0.1"
)

run_one () {
  local job="$1"
  local tag seed do lr id sl
  IFS='|' read -r tag seed do lr id sl <<< "$job"

  local attempt
  for attempt in 1 2; do
    echo ""
    echo "=== [$(date '+%F %T')] START $tag seed=$seed (attempt $attempt) ==="
    python -u train.py --dataset MOSI --seed "$seed" \
      --feature_A "$FEAT_A" --feature_V "$FEAT_V" \
      --bert_pretrained roberta-large --ism_full_frame true --multi_task true \
      --dropout "$do" --lr "$lr" --d_state 64 --ism_depth "$id" \
      --sub_loss_lambda "$sl" --batch_size 4 --key_eval MAE \
      --exp_tag "$tag" 2>&1 \
      | grep -E "Early stop monitor|New best|Testing with|test_|Test results|early stop at|Early stop at|out of memory|Traceback"
    # 产出校验: 四号 ckpt 必须存在
    if [ -f "checkpoints/MOSI_${tag}_seed${seed}_best_F1_has0.pt" ]; then
      echo "=== [$(date '+%F %T')] DONE  $tag seed=$seed ==="
      return 0
    fi
    echo "!!! [$(date '+%F %T')] $tag seed=$seed 未产出四号 ckpt, 准备重试"
    sleep 20
  done
  echo "!!! [$(date '+%F %T')] $tag seed=$seed 两次尝试均失败, 跳过 (稍后可从日志排查)"
  return 1
}

FAILED=()
echo "########## P0 rerun start $(date '+%F %T') | ${#JOBS[@]} jobs on GPU1 ##########"
for job in "${JOBS[@]}"; do
  run_one "$job" || FAILED+=("$job")
done

echo ""
echo "########## 训练阶段结束 $(date '+%F %T') | 失败: ${#FAILED[@]} 个 ##########"
if [ ${#FAILED[@]} -gt 0 ]; then
  printf '失败任务: %s\n' "${FAILED[@]}"
fi

# ---- P0-2: 二分类阈值校准 (3-seed 集成; F1_has0 ckpt 与 Acc2 ckpt 各一套) ----
echo ""
echo "########## 开始二分类阈值校准 (P0-2) $(date '+%F %T') ##########"
for cfg in p0A p0B p0D; do
  for metric in F1_has0 Acc2; do
    CKPTS=""
    for seed in 42 1111 2024; do
      f="checkpoints/MOSI_${cfg}_seed${seed}_best_${metric}.pt"
      [ -f "$f" ] && CKPTS="$CKPTS $f"
    done
    if [ -n "$CKPTS" ]; then
      echo ""
      echo ">>> 校准 $cfg [$metric] 集成"
      python -u scripts/calibrate_binary_threshold.py --ckpt $CKPTS 2>&1 \
        | grep -E ">>>|\[valid|\[test|最优|⚠|结果已保存|Traceback"
    fi
  done
done

echo ""
echo "########## P0 rerun ALL DONE $(date '+%F %T') ##########"
touch logs/p0_rerun.done
