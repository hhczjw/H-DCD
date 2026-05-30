#!/usr/bin/env python3
"""
视频特征消融实验 —— 对比 5 种视觉特征在 MOSI 上的效果
=============================================================
特征类型:
  1. Original_FACET (基线)  — 原始 20 维 FACET 特征
  2. CLIP (ViT-B-32)        — 512 维跨模态语义特征
  3. VideoMAE               — 768 维时空掩码自编码特征
  4. OpenFace 3.0           — 18  维面部行为特征
  5. DINOv3 (ViT-B/16)      — 768 维自监督视觉特征

用法:
  python scripts/run_video_ablation.py [--seeds 42,123,456] [--dry_run]

输出:
  results/ablation_video/summary.json     — 各实验完整指标
  results/ablation_video/comparison.csv   — CSV 对比表
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

# ============================================================
# 全局配置
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = "/media/zjw/951FB31A9E1EB7E0/dateSet/MSA-DataSets"
ORIGINAL_PKL = os.path.join(DATA_ROOT, "CMU-MOSI/Processed/unaligned_50.pkl")
FEATURES_DIR = PROJECT_ROOT / "features"
RESULTS_ROOT = PROJECT_ROOT / "results" / "ablation_video"

# ★ 5 种视觉特征配置
FEATURE_CONFIGS = {
    "Original_FACET": {
        "path": None,           # None = 使用原始特征, 不传 --feature_V
        "dim": 20,
        "desc": "原始FACET (基线)",
    },
    "CLIP": {
        "path": str(FEATURES_DIR / "vision_clip.pkl"),
        "dim": 512,
        "desc": "CLIP ViT-B-32",
    },
    "VideoMAE": {
        "path": str(FEATURES_DIR / "vision_videomae.pkl"),
        "dim": 768,
        "desc": "VideoMAE Base",
    },
    "OpenFace3": {
        "path": str(FEATURES_DIR / "vision_openface3.pkl"),
        "dim": 18,
        "desc": "OpenFace 3.0 (AU+Gaze+Emotion)",
    },
    "DINOv3": {
        "path": str(FEATURES_DIR / "vision_dinov3.pkl"),
        "dim": 768,
        "desc": "DINOv3 ViT-B/16",
    },
}

# 默认多种子
DEFAULT_SEEDS = [42, 123, 456]


# ============================================================
# 1) 将 flat pkl {vid_id: array} 转换为 split-aware pkl
# ============================================================
def _load_flat_features(flat_path: str) -> dict:
    """加载我们的 flat 格式 pkl, 返回 {vid_id: np.ndarray(50, D)}"""
    with open(flat_path, "rb") as f:
        data = pickle.load(f)
    return data


def make_split_pkl(
    flat_path: str,
    output_path: str,
    original_path: str = ORIGINAL_PKL,
) -> str:
    """
    将 flat pkl 转换为 dataloader 所需的 split 格式:
    {
        "train": {
            "vision": np.ndarray (N_train, 50, D),
            "vision_ids": ["id1", "id2", ...],
        },
        "valid": {...},
        "test": {...},
    }
    """
    if os.path.isfile(output_path):
        return output_path  # 已存在

    print(f"  [prepare] 转换: {os.path.basename(flat_path)} → {os.path.basename(output_path)}")

    with open(original_path, "rb") as f:
        orig = pickle.load(f)
    flat = _load_flat_features(flat_path)
    dim = next(iter(flat.values())).shape[-1]

    result = {}
    for split_name in ["train", "valid", "test"]:
        orig_ids = [str(i).replace("$_$", "_") for i in orig[split_name]["id"]]
        vision_list = []
        for sid in orig_ids:
            arr = flat.get(sid)
            if arr is None:
                arr = np.zeros((50, dim), dtype=np.float32)
            else:
                arr = np.asarray(arr, dtype=np.float32)
            vision_list.append(arr)
        result[split_name] = {
            "vision": np.stack(vision_list).astype(np.float32),
            "vision_ids": [str(i).replace("$_$", "_") for i in orig[split_name]["id"]],
        }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(result, f)
    print(f"  [prepare] 完成: {output_path}")
    return output_path


# ============================================================
# 2) 运行单次训练 + 解析结果 + 记录训练过程
# ============================================================
def _find_latest_log(log_dir: str, pattern: str = "MSA_MOSI") -> str | None:
    """找到最近创建的训练日志文件"""
    logs = sorted(
        [f for f in os.listdir(log_dir) if f.startswith(pattern) and f.endswith(".log")],
        key=lambda f: os.path.getmtime(os.path.join(log_dir, f)),
        reverse=True,
    )
    return os.path.join(log_dir, logs[0]) if logs else None


def _parse_training_log(log_path: str) -> list[dict]:
    """
    从 train.py 日志中提取逐 epoch 的训练/验证指标。
    日志格式:
      [Train] Epoch 1 | loss=0.6234 | lr=[...]
      [valid] Acc2=0.8611 | MAE=0.8319 | Corr=0.7409 | Acc7=0.3406 | F1=0.8274 | Loss=0.6275
    """
    epochs = []
    current_epoch = None
    train_re = re.compile(r"\[Train\]\s*Epoch\s+(\d+)\s*\|")
    val_re = re.compile(r"\[(valid)\]")
    metric_re = re.compile(r"(\w+(?:_has0)?)\s*=\s*([0-9.e+\-]+)")

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            # 匹配训练行
            m_train = train_re.search(line)
            if m_train:
                epoch_num = int(m_train.group(1))
                if current_epoch is None or current_epoch.get("epoch") != epoch_num:
                    if current_epoch is not None:
                        epochs.append(current_epoch)
                    current_epoch = {"epoch": epoch_num}
                # 提取训练 loss
                for k, v in metric_re.findall(line):
                    current_epoch[f"train_{k}"] = float(v)
                continue

            # 匹配验证行
            m_val = val_re.search(line)
            if m_val and current_epoch is not None:
                for k, v in metric_re.findall(line):
                    current_epoch[k] = float(v)

    if current_epoch is not None:
        epochs.append(current_epoch)
    return epochs


def _save_training_curves(epochs: list[dict], output_path: str) -> None:
    """将逐 epoch 指标保存为 CSV"""
    if not epochs:
        return
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    all_keys = sorted(set().union(*[d.keys() for d in epochs]))
    # epoch 放第一列
    if "epoch" in all_keys:
        all_keys.remove("epoch")
    fieldnames = ["epoch"] + all_keys
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for ep in epochs:
            writer.writerow({k: ep.get(k, "") for k in fieldnames})
    print(f"  [curves] 训练过程已保存: {output_path} ({len(epochs)} epochs)")


def run_train(feature_name: str, cfg: dict, seed: int, dry_run: bool = False) -> dict | None:
    tag = feature_name
    exp_tag = feature_name  # train.py 内部自动追加 _seed{seed}, 这里不要重复

    cmd = [
        sys.executable, "train.py",
        "--dataset", "MOSI",
        "--seed", str(seed),
        "--exp_tag", exp_tag,
        "--epochs", "120",
        "--batch_size", "32",
        "--lr", "3e-4",
        "--dropout", "0.5",
        "--early_stop", "30",
        "--key_eval", "MAE",
        "--secondary_metric", "Acc2",
        "--tertiary_metric", "Acc7",
        "--bert_pretrained", "bert-base-uncased",
        "--v_self_ratio", "0.3",
        "--sub_loss_lambda",'0.5',
        "--weight_decay", "1e-4",
    ]

    if cfg["path"] is not None:
        split_path = os.path.join(FEATURES_DIR, f"split_{os.path.basename(cfg['path'])}")
        make_split_pkl(cfg["path"], split_path)
        cmd += ["--feature_V", split_path]

    cmd_str = " ".join(cmd)

    if dry_run:
        print(f"  [DRY RUN] {cmd_str}")
        return None

    print(f"\n{'='*60}")
    print(f">>> 训练: {feature_name} (seed={seed})")
    print(f"  CMD: {cmd_str}")
    print(f"{'='*60}")

    # 记录训练开始前的日志文件, 用于定位本次训练的日志
    log_dir = os.path.join(PROJECT_ROOT, "logs")
    before_logs = set(os.listdir(log_dir)) if os.path.isdir(log_dir) else set()

    t0 = time.time()
    result = subprocess.run(
        cmd, cwd=str(PROJECT_ROOT),
        capture_output=True, text=True,
        env={**os.environ, "PYTHONHOME": "", "PYTHONPATH": str(PROJECT_ROOT)},
    )
    elapsed = time.time() - t0
    print(f"  耗时: {elapsed:.0f}s, 退出码: {result.returncode}")

    for line in result.stdout.split("\n")[-30:]:
        if any(kw in line for kw in ["Acc2", "MAE", "Acc7", "F1", "Corr", "Test", "best"]):
            print(f"  | {line.strip()}")

    if result.returncode != 0:
        print(f"  [ERROR] stderr 最后 20 行:")
        for line in result.stderr.split("\n")[-20:]:
            print(f"  | {line.strip()}")
        return None

    # ---- 解析训练过程 (逐 epoch 曲线) ----
    after_logs = set(os.listdir(log_dir)) if os.path.isdir(log_dir) else set()
    new_logs = after_logs - before_logs
    if new_logs:
        log_file = os.path.join(log_dir, sorted(new_logs)[-1])
        epochs = _parse_training_log(log_file)
        curve_dir = os.path.join(RESULTS_ROOT, "curves")
        curve_path = os.path.join(curve_dir, f"{feature_name}_seed{seed}_epochs.csv")
        _save_training_curves(epochs, curve_path)

    # ---- 解析最终 test 结果 ----
    results_json = os.path.join(PROJECT_ROOT, "results", f"MOSI_{feature_name}_seed{seed}_test.json")
    if not os.path.isfile(results_json):
        print(f"  [WARN] 未找到: {results_json}")
        return None

    with open(results_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


# ============================================================
# 3) 解析单次实验的最佳指标
# ============================================================
def extract_best_metrics(test_json: dict, feature_name: str, seed: int) -> dict:
    primary_key = None
    for k in test_json:
        if k.startswith("primary_"):
            primary_key = k
            break
    if primary_key is None:
        primary_key = list(test_json.keys())[0]

    best = test_json[primary_key].copy()

    best_acc7 = best.get("Acc7", 0.0)
    best_mae = best.get("MAE", 999.0)
    for k, v in test_json.items():
        if "Acc7" in v and v["Acc7"] > best_acc7:
            best_acc7 = v["Acc7"]
        if "MAE" in v and v["MAE"] < best_mae:
            best_mae = v["MAE"]
    best["best_Acc7"] = best_acc7
    best["best_MAE"] = best_mae
    best["feature"] = feature_name
    best["seed"] = seed
    return best


# ============================================================
# 4) 汇总并打印对比表
# ============================================================
def summarize(all_metrics: list[dict]) -> None:
    groups = defaultdict(list)
    for m in all_metrics:
        groups[m["feature"]].append(m)

    METRICS = ["Acc7", "Acc2", "Acc5", "F1", "MAE", "Corr"]
    DIR = {"Acc7": "↑", "Acc2": "↑", "Acc5": "↑", "F1": "↑", "MAE": "↓", "Corr": "↑"}

    print("\n" + "=" * 75)
    print("                    视频特征消融实验结果")
    print("=" * 75)
    header = f"{'Feature':<18}"
    for m in METRICS:
        header += f"  {m}({DIR[m]})     "
    print(header)
    print("-" * 75)

    rows = []
    for fname, ml in sorted(groups.items()):
        line = f"{fname:<18}"
        row = {"Feature": fname}
        for m in METRICS:
            vals = [x.get(m, float('nan')) for x in ml if m in x]
            vals = [v for v in vals if np.isfinite(v)]
            if vals:
                mean_v = np.mean(vals)
                std_v = np.std(vals)
                line += f"  {mean_v:.4f}±{std_v:.3f}"
                row[f"{m}_mean"] = round(float(mean_v), 4)
                row[f"{m}_std"] = round(float(std_v), 4)
            else:
                line += "  N/A         "
                row[f"{m}_mean"] = None
                row[f"{m}_std"] = None
        print(line)
        rows.append(row)

    os.makedirs(RESULTS_ROOT, exist_ok=True)

    csv_path = os.path.join(RESULTS_ROOT, "comparison.csv")
    csv_header = ["Feature"] + [f"{m}_{s}" for m in METRICS for s in ("mean", "std")]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_header)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV → {csv_path}")

    json_path = os.path.join(RESULTS_ROOT, "summary.json")
    all_data = {
        "configs": {k: {"dim": v["dim"], "desc": v["desc"]} for k, v in FEATURE_CONFIGS.items()},
        "per_seed": all_metrics,
        "aggregated": rows,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False)
    print(f"JSON → {json_path}")


# ============================================================
# 5) 主入口
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="视频特征消融实验")
    parser.add_argument("--seeds", type=str, default="42,123,456",
                        help="逗号分隔的随机种子 (默认: 42,123,456)")
    parser.add_argument("--dry_run", action="store_true",
                        help="仅打印命令, 不实际执行")
    parser.add_argument("--feature", type=str, default=None,
                        help="仅测试某个特征 (如: DINOv3)")
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",")]

    print("=" * 60)
    print("  视频特征消融实验")
    print(f"  数据集: CMU-MOSI  |  种子: {seeds}  |  Dry run: {args.dry_run}")
    print("=" * 60)

    all_metrics = []

    for fname, cfg in FEATURE_CONFIGS.items():
        if args.feature and fname != args.feature:
            continue
        print(f"\n{'#'*60}")
        print(f"# 特征: {fname} — {cfg['desc']} (dim={cfg['dim']})")
        print(f"{'#'*60}")

        for seed in seeds:
            test_json = run_train(fname, cfg, seed, dry_run=args.dry_run)
            if test_json is not None:
                metrics = extract_best_metrics(test_json, fname, seed)
                all_metrics.append(metrics)

    if not args.dry_run and all_metrics:
        summarize(all_metrics)
    elif args.dry_run:
        print("\n[Dry run 完成 — 请去掉 --dry_run 实际执行]")
    else:
        print("\n[警告] 没有收集到任何实验结果, 请检查训练日志]")


if __name__ == "__main__":
    main()
