#!/usr/bin/env python3
"""
网格搜索结果解析器
==================
读取 grid_results/ 目录下所有 *_test.json,
按 exp_tag 聚合, 输出完整排名表 (支持多 seed 均值).

用法:
    python scripts/parse_grid_results.py                          # 解析全部
    python scripts/parse_grid_results.py --dir grid_results       # 指定目录
    python scripts/parse_grid_results.py --tag gs2                # 仅解析匹配 tag 的实验
    python scripts/parse_grid_results.py --top 5                  # 仅显示 Top-5
    python scripts/parse_grid_results.py --markdown               # 输出 Markdown 表格
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="网格搜索结果解析")
    p.add_argument("--dir", type=str, default="grid_results",
                   help="结果目录 (默认 grid_results/)")
    p.add_argument("--tag", type=str, default=None,
                   help="仅解析匹配此子串的实验 (e.g. gs2)")
    p.add_argument("--top", type=int, default=0,
                   help="仅显示 Top-N (按 Acc7_ter 排序), 0=全部")
    p.add_argument("--sort_by", type=str, default="Acc7_ter",
                   choices=["Acc7_ter", "Acc2_ter", "MAE_sec", "Acc7_pri", "Acc2_pri", "MAE_pri"],
                   help="排序指标")
    p.add_argument("--markdown", action="store_true",
                   help="以 Markdown 表格输出")
    p.add_argument("--csv", type=str, default=None,
                   help="保存为 CSV 文件")
    return p.parse_args()


# ckpt 键 → 短前缀映射
CKPT_PREFIX = {
    "primary_Acc2":   "pri",
    "secondary_MAE":  "sec",
    "tertiary_Acc7":  "ter",
    "primary_MAE":    "pri",
    "primary_Acc7":   "pri",
    "secondary_Acc2": "sec",
    "secondary_Acc7": "sec",
    "tertiary_MAE":   "ter",
    "tertiary_Acc2":  "ter",
}


def extract_metrics(json_path: str) -> dict:
    """从 test JSON 中提取 triple-ckpt 所有关键指标.

    返回扁平化键名: pri_MAE, pri_Acc2, pri_Acc7, sec_MAE, sec_Acc2,
    sec_Acc7, ter_MAE, ter_Acc2, ter_Acc7 等.
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    metrics = {}
    for ckpt_key in data:
        prefix = CKPT_PREFIX.get(ckpt_key, "unk")
        for m in ["MAE", "Corr", "Acc2", "F1", "Acc5", "Acc7", "Loss"]:
            if m in data[ckpt_key]:
                metrics[f"{prefix}_{m}"] = data[ckpt_key][m]
    return metrics


def main():
    args = parse_args()
    results_dir = Path(args.dir)
    if not results_dir.is_dir():
        print(f"Error: 目录不存在: {results_dir}")
        sys.exit(1)

    # 收集所有 JSON
    json_files = sorted(results_dir.glob("*_test.json"))
    if args.tag:
        json_files = [f for f in json_files if args.tag in f.name]

    if not json_files:
        print(f"未找到匹配的 JSON 文件 (dir={results_dir}, tag={args.tag})")
        sys.exit(1)

    print(f"找到 {len(json_files)} 个 JSON 文件\n")

    # 按 exp_tag 分组
    groups = defaultdict(list)
    for jf in json_files:
        # 文件名格式: MOSI_{exp_tag}_seed{seed}_test.json
        name = jf.stem.replace("_test", "")
        # 提取 exp_tag
        parts = name.split("_seed")
        if len(parts) < 2:
            continue
        exp_tag = parts[0].replace("MOSI_", "")
        seed = parts[1]
        metrics = extract_metrics(str(jf))
        metrics["_seed"] = seed
        metrics["_file"] = jf.name
        groups[exp_tag].append(metrics)

    # 聚合
    summary = []
    for tag, runs in groups.items():
        n_seeds = len(runs)
        row = {"exp_tag": tag, "n_seeds": n_seeds}

        # 收集所有指标
        all_metrics = defaultdict(list)
        for run in runs:
            for k, v in run.items():
                if not k.startswith("_"):
                    all_metrics[k].append(float(v))

        for k, vals in all_metrics.items():
            if len(vals) == 1:
                row[k] = f"{vals[0]:.4f}"
            else:
                row[k] = f"{np.mean(vals):.4f}±{np.std(vals):.4f}"
            row[f"{k}_mean"] = np.mean(vals)  # 用于排序

        summary.append(row)

    # 排序键映射: 显示名 → 内部键
    SORT_MAP = {
        "Acc7_ter": "ter_Acc7", "Acc2_ter": "ter_Acc2", "MAE_sec": "sec_MAE",
        "Acc7_pri": "pri_Acc7", "Acc2_pri": "pri_Acc2", "MAE_pri": "pri_MAE",
    }
    sort_key = f"{SORT_MAP.get(args.sort_by, args.sort_by)}_mean"
    summary.sort(key=lambda r: r.get(sort_key, -1e9),
                 reverse=("MAE" not in SORT_MAP.get(args.sort_by, args.sort_by)))

    # 截断
    if args.top > 0:
        summary = summary[:args.top]

    # ── 输出 ──
    # 选择要显示的指标列: (内部键, 显示名, 宽度)
    display_cols = [
        ("exp_tag",    "实验标签",    30),
        ("n_seeds",    "Seeds",       5),
        ("ter_Acc7",   "Acc7(ter)↑",  14),
        ("ter_Acc2",   "Acc2(ter)↑",  14),
        ("sec_MAE",    "MAE(sec)↓",   14),
        ("pri_Acc7",   "Acc7(pri)↑",  14),
        ("pri_Acc2",   "Acc2(pri)↑",  14),
        ("pri_MAE",    "MAE(pri)↓",   14),
        ("ter_F1",     "F1(ter)↑",    14),
    ]

    if args.markdown:
        # Markdown 表格
        header = "| " + " | ".join(h[1] for h in display_cols) + " |"
        sep = "|" + "|".join([":" + "-" * (h[2] - 2) + ":" for h in display_cols]) + "|"
        print(header)
        print(sep)
        for row in summary:
            cells = []
            for key, _, width in display_cols:
                val = row.get(key, "-")
                if key == "exp_tag":
                    cells.append(f"{str(val):<{width}}")
                else:
                    cells.append(f"{str(val):>{width}}")
            print("| " + " | ".join(cells) + " |")
    else:
        # 终端表格
        header = " ".join(f"{h[1]:>{h[2]}}" for h in display_cols)
        print(header)
        print("-" * len(header))
        for rank, row in enumerate(summary, 1):
            cells = []
            for key, _, width in display_cols:
                val = row.get(key, "-")
                if key == "exp_tag":
                    val = f"{rank}. {val}"
                    cells.append(f"{str(val):<{width}}")
                else:
                    cells.append(f"{str(val):>{width}}")
            print(" ".join(cells))

    # 高亮最优
    print(f"\n{'='*70}")
    best_acc7 = max(summary, key=lambda r: r.get("ter_Acc7_mean", -1))
    best_acc2 = max(summary, key=lambda r: r.get("ter_Acc2_mean", -1))
    best_mae  = min(summary, key=lambda r: r.get("sec_MAE_mean", 1e9))
    print(f"🏆 Best Acc7: {best_acc7['exp_tag']} = {best_acc7.get('ter_Acc7', 'N/A')}")
    print(f"🏆 Best Acc2: {best_acc2['exp_tag']} = {best_acc2.get('ter_Acc2', 'N/A')}")
    print(f"🏆 Best MAE:  {best_mae['exp_tag']}  = {best_mae.get('sec_MAE', 'N/A')}")

    # CSV 导出
    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([h[0] for h in display_cols])
            for row in summary:
                writer.writerow([
                    row.get(key, row.get(key.split("|")[0], ""))
                    for key, _, _ in display_cols
                ])
        print(f"CSV saved: {args.csv}")

    print(f"\n总计 {len(summary)} 组实验, {len(json_files)} 次训练")


if __name__ == "__main__":
    main()
