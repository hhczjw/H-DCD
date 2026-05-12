"""
聚合 Run C 多 seed test 结果, 输出 mean/std/min/max 表格 + Markdown 报告.

用法:
    python scripts/agg_run_C.py --pattern "MOSI_C_alpha03_seed*_test.json"
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics as st
from pathlib import Path

METRICS = ["MAE", "Corr", "Acc2", "Acc2_has0", "Acc5", "Acc7", "F1", "F1_has0"]


def load_results(pattern: str, results_dir: str = "results"):
    files = sorted(glob.glob(os.path.join(results_dir, pattern)))
    if not files:
        raise FileNotFoundError(f"No files match: {results_dir}/{pattern}")
    rows = []
    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            d = json.load(f)
        rows.append({"file": os.path.basename(fp), "data": d})
    return rows


def discover_ckpt_keys(rows):
    """自动发现 test JSON 中的 ckpt key (e.g. primary_Acc2, secondary_MAE, primary_MAE).
    取所有 run 共有的 key.
    """
    common = None
    for r in rows:
        keys = set(r["data"].keys())
        common = keys if common is None else (common & keys)
    return sorted(common or [])


def agg_metric(rows, ckpt_key: str, metric: str):
    vals = []
    for r in rows:
        if metric in r["data"][ckpt_key]:
            vals.append(r["data"][ckpt_key][metric])
    if not vals:
        return None
    if len(vals) >= 2:
        mean = st.mean(vals)
        std = st.stdev(vals)
    else:
        mean, std = vals[0], 0.0
    return {"mean": mean, "std": std, "min": min(vals), "max": max(vals), "vals": vals}


def fmt_row(metric, primary, secondary):
    p = f"{primary['mean']*100:.2f} ± {primary['std']*100:.2f}"
    s = f"{secondary['mean']*100:.2f} ± {secondary['std']*100:.2f}"
    return f"| {metric:<5} | {p:<18} | {s:<18} |"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", required=True)
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--out", default=None, help="markdown 报告输出路径 (默认 results/<pattern>.summary.md)")
    args = ap.parse_args()

    rows = load_results(args.pattern, args.results_dir)
    n = len(rows)
    print(f"\n[agg] Loaded {n} runs:")
    for r in rows:
        print(f"  - {r['file']}")

    ckpt_keys = discover_ckpt_keys(rows)
    print(f"\n[agg] Detected ckpt keys: {ckpt_keys}")

    summary = {"n_runs": n}
    for ck in ckpt_keys:
        print(f"\n=== {ck} ===")
        print(f"{'Metric':<10} | {'mean':<10} | {'std':<10} | {'min':<10} | {'max':<10}")
        print("-" * 60)
        summary[ck] = {}
        for m in METRICS:
            agg = agg_metric(rows, ck, m)
            if agg is None:
                continue
            summary[ck][m] = agg
            print(f"{m:<10} | {agg['mean']:<10.4f} | {agg['std']:<10.4f} | {agg['min']:<10.4f} | {agg['max']:<10.4f}")

    out_path = args.out or os.path.join(
        args.results_dir,
        args.pattern.replace("*", "all").replace("_test.json", ".summary.json"),
    )
    Path(os.path.dirname(out_path) or ".").mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n[agg] Summary saved: {out_path}")

    # 通过门槛检查 (优先 secondary_MAE, 否则 primary_MAE, 否则第一个 ckpt)
    eval_key = None
    for k in ("secondary_MAE", "primary_MAE", "primary_Acc2"):
        if k in summary and summary[k]:
            eval_key = k
            break
    if eval_key is None and ckpt_keys:
        eval_key = ckpt_keys[0]
    if eval_key:
        print(f"\n=== Pass-Gate Check ({eval_key}) ===")
        s = summary[eval_key]
        if "Acc7" in s:
            print(f"  Acc7 mean >= 0.42 ? {s['Acc7']['mean']:.4f} -> {'PASS' if s['Acc7']['mean'] >= 0.42 else 'FAIL'}")
            print(f"  Acc7 std  <= 4%   ? {s['Acc7']['std']*100:.2f}% -> {'PASS' if s['Acc7']['std'] <= 0.04 else 'FAIL'}")
        if "Acc2" in s:
            print(f"  Acc2 mean >= 0.84 ? {s['Acc2']['mean']:.4f} -> {'PASS' if s['Acc2']['mean'] >= 0.84 else 'FAIL'}")


if __name__ == "__main__":
    main()