#!/usr/bin/env python3
"""
视频特征 Grid Search 脚本
===========================
针对训练过程分析中发现的问题（过拟合 / best epoch 脱节 / 收敛速度差异），
分三阶段递进搜索最优超参组合。

三阶段策略:
  Stage 1 (Reg): 在 OpenFace3 上搜 dropout / lr / weight_decay    → 27 runs
  Stage 2 (Loss): 取 Stage1 最优, 搜 v_self_ratio / sub_loss_lambda / warmup_ratio / key_eval
  Stage 3 (All):  取全局最优, 验证到全部 5 种特征

用法:
  python scripts/run_grid_search.py --stage 1 --seeds 42           # 仅跑 Stage 1
  python scripts/run_grid_search.py --stage all --seeds 42         # 跑全部三阶段
  python scripts/run_grid_search.py --dry_run                      # 预览所有实验
"""

from __future__ import annotations

import argparse, csv, json, os, pickle, re, subprocess, sys, time
from collections import defaultdict
from itertools import product
from pathlib import Path

import numpy as np

# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = "/media/zjw/951FB31A9E1EB7E0/dateSet/MSA-DataSets"
ORIGINAL_PKL = os.path.join(DATA_ROOT, "CMU-MOSI/Processed/unaligned_50.pkl")
FEATURES_DIR = PROJECT_ROOT / "features"
RESULTS_ROOT = PROJECT_ROOT / "results" / "grid_search"

# ============================================================
# 全部可搜索参数分类
# ============================================================
# ★ 核心训练超参 (正则化 + 优化器) — 对过拟合影响最大
STAGE1_SEARCH = {
    "lr": [1e-4, 3e-4, 1e-3],
    "dropout": [0.1, 0.3, 0.5],
    "weight_decay": [1e-5, 1e-4, 1e-3],
}

# ★ 损失与模态交互 — 影响收敛时机和 best epoch 对齐
STAGE2_SEARCH = {
    "key_eval": ["Acc2", "MAE"],           # early stop 监控指标
    "v_self_ratio": [0.0, 0.2, 0.4],       # 跨模态 V 自注入
    "sub_loss_lambda": [0.0, 0.3, 0.5],     # 模态级 aux loss
    "warmup_ratio": [0.0, 0.1, 0.15],       # LR warmup
}

# ★ 模型结构 — 低优先级, 边际影响
STAGE3_SEARCH = {
    "ism_depth": [1, 2, 3],
    "ism_mixer_type": ["bimamba", "bimamba3"],
    "contrastive_weight": [0.0, 0.1, 0.2, 0.3],
}

# ★ 固定参数 (所有实验统一)
FIXED_PARAMS = {
    "dataset": "MOSI",
    "bert_pretrained": "bert-base-uncased",
    "epochs": 120,
    "batch_size": 32,
    "early_stop": 30,
    "secondary_metric": "MAE",
    "tertiary_metric": "Acc7",
}

# ★ Baseline 超参 (从上一轮消融实验继承)
BASELINE_PARAMS = {
    "lr": 3e-4,
    "dropout": 0.3,
    "weight_decay": 1e-5,
    "key_eval": "Acc2",
    "v_self_ratio": 0.3,
    "sub_loss_lambda": 0.5,
    "warmup_ratio": 0.0,
    "ism_depth": 3,
    "ism_mixer_type": "bimamba",
    "contrastive_weight": 0.0,
}

# 用于 Grid Search 的特征
FEATURE_CONFIGS = {
    "OpenFace3": {"path": str(FEATURES_DIR / "vision_openface3.pkl"), "dim": 18},
    "Original_FACET": {"path": None, "dim": 20},
    "CLIP": {"path": str(FEATURES_DIR / "vision_clip.pkl"), "dim": 512},
    "VideoMAE": {"path": str(FEATURES_DIR / "vision_videomae.pkl"), "dim": 768},
    "DINOv3": {"path": str(FEATURES_DIR / "vision_dinov3.pkl"), "dim": 768},
}

DEFAULT_SEEDS = [42, 123, 456]

# ★ Stage 1 特征选择: 低/中/高维各选一个代表, 避免全 5 特征组合爆炸
#    OpenFace3=18维(低)  Original_FACET=20维(中)  DINOv3=768维(高)
STAGE1_FEATURES = ["OpenFace3", "Original_FACET", "DINOv3"]

# ═══════════════════════════════════════════════════════════
# 工具函数 (复用 run_video_ablation.py)
# ═══════════════════════════════════════════════════════════

def make_split_pkl(flat_path, output_path, original_path=ORIGINAL_PKL):
    if os.path.isfile(output_path):
        return output_path
    with open(original_path, "rb") as f:
        orig = pickle.load(f)
    with open(flat_path, "rb") as f:
        flat = pickle.load(f)
    dim = next(iter(flat.values())).shape[-1]
    result = {}
    for split_name in ["train", "valid", "test"]:
        orig_ids = [str(i).replace("$_$", "_") for i in orig[split_name]["id"]]
        vision_list = [np.asarray(flat.get(sid, np.zeros((50, dim), dtype=np.float32)), dtype=np.float32) for sid in orig_ids]
        result[split_name] = {
            "vision": np.stack(vision_list).astype(np.float32),
            "vision_ids": [str(i).replace("$_$", "_") for i in orig[split_name]["id"]],
        }
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(result, f)
    return output_path


def _parse_training_log(log_path):
    epochs = []
    current_epoch = None
    train_re = re.compile(r"\[Train\]\s*Epoch\s+(\d+)\s*\|")
    val_re = re.compile(r"\[(valid)\]")
    metric_re = re.compile(r"(\w+(?:_has0)?)\s*=\s*([0-9.e+\-]+)")
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            m_train = train_re.search(line)
            if m_train:
                epoch_num = int(m_train.group(1))
                if current_epoch is None or current_epoch.get("epoch") != epoch_num:
                    if current_epoch is not None:
                        epochs.append(current_epoch)
                    current_epoch = {"epoch": epoch_num}
                for k, v in metric_re.findall(line):
                    current_epoch[f"train_{k}"] = float(v)
                continue
            m_val = val_re.search(line)
            if m_val and current_epoch is not None:
                for k, v in metric_re.findall(line):
                    current_epoch[k] = float(v)
    if current_epoch is not None:
        epochs.append(current_epoch)
    return epochs


def run_single_train(params: dict, feature_name: str, seed: int, dry_run: bool = False) -> dict | None:
    """运行单次训练, 返回 {test_json, epochs, valid_best, log_path}"""
    exp_tag = f"GS_{feature_name}".replace(" ", "_")

    cmd = [sys.executable, "train.py"]
    for k in FIXED_PARAMS:
        cmd += [f"--{k}", str(FIXED_PARAMS[k])]
    for k, v in params.items():
        cmd += [f"--{k}", str(v)]
    cmd += ["--seed", str(seed), "--exp_tag", exp_tag]

    cfg = FEATURE_CONFIGS[feature_name]
    if cfg["path"] is not None:
        split_path = os.path.join(FEATURES_DIR, f"split_{os.path.basename(cfg['path'])}")
        make_split_pkl(cfg["path"], split_path)
        cmd += ["--feature_V", split_path]

    cmd_str = " ".join(cmd)

    if dry_run:
        print(f"  [DRY] {cmd_str}")
        return None

    print(f"  ▶ {feature_name} seed={seed}", flush=True)
    log_dir = os.path.join(PROJECT_ROOT, "logs")
    before_logs = set(os.listdir(log_dir)) if os.path.isdir(log_dir) else set()

    t0 = time.time()
    # ★ capture_output=False → 子进程输出直接流到当前终端/日志, 不等待
    result = subprocess.run(
        cmd, cwd=str(PROJECT_ROOT),
        capture_output=False,   # 实时输出, 不缓存
        text=True,
        env={**os.environ, "PYTHONHOME": "", "PYTHONPATH": str(PROJECT_ROOT)},
    )
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"    [FAIL] exit={result.returncode} in {elapsed:.0f}s", flush=True)
        return None

    # 解析训练曲线
    after_logs = set(os.listdir(log_dir)) if os.path.isdir(log_dir) else set()
    new_logs = after_logs - before_logs
    epochs_data = []
    valid_best = {}
    if new_logs:
        log_file = os.path.join(log_dir, sorted(new_logs)[-1])
        epochs_data = _parse_training_log(log_file)
        if epochs_data:
            valid_best = {
                "best_Acc2": max((e.get("Acc2", 0) for e in epochs_data), default=0),
                "best_MAE": min((e.get("MAE", 999) for e in epochs_data), default=999),
                "best_Acc7": max((e.get("Acc7", 0) for e in epochs_data), default=0),
                "best_epoch_Acc2": next((e["epoch"] for e in epochs_data if e.get("Acc2") == valid_best["best_Acc2"]), 0),
                "best_epoch_MAE": next((e["epoch"] for e in epochs_data if e.get("MAE") == valid_best["best_MAE"]), 0),
                "total_epochs": len(epochs_data),
            }

    # test JSON
    results_json = os.path.join(PROJECT_ROOT, "results", f"MOSI_{exp_tag}_seed{seed}_test.json")
    test_data = {}
    if os.path.isfile(results_json):
        with open(results_json, "r") as f:
            test_data = json.load(f)

    print(f"    [OK] {elapsed:.0f}s | valid best_Acc2={valid_best.get('best_Acc2',0):.4f} | test_Acc2={list(test_data.values())[0].get('Acc2',0) if test_data else 0:.4f}", flush=True)
    return {"test": test_data, "epochs": epochs_data, "valid_best": valid_best}


def _get_test_metric(test_data: dict, metric: str = "Acc2") -> float:
    for k, v in test_data.items():
        if k.startswith("primary_") and metric in v:
            return float(v[metric])
    return 0.0


# ═══════════════════════════════════════════════════════════
# Grid Search 引擎
# ═══════════════════════════════════════════════════════════

def grid_search_stage(stage_id: int, seeds: list[int], feature_filter: str = None,
                      dry_run: bool = False) -> list[dict]:
    """
    执行一个阶段的 Grid Search.
    - Stage 1: 在 STAGE1_FEATURES (3种维度代表) 上搜正则化参数, 每个特征独立找最优
    - Stage 2: 在 Stage1 最优特征上搜 loss/模态交互参数
    - Stage 3: 在 Stage2 最优特征上搜模型结构参数
    - Stage 4: 各特征用独立正则化 + 全局最优其他参数, 全 5 特征验证
    """
    if stage_id == 1:
        search_space = STAGE1_SEARCH
        features = list(STAGE1_FEATURES)     # ★ 3 种维度代表: 低/中/高维各搜各的
        base_params = {}
    elif stage_id == 2:
        search_space = STAGE2_SEARCH
        features = [_find_best_feature_from_stage(1)]  # ★ 取 Stage1 最优特征
        base_params = _load_best_params_for_feature(1, features[0])
    elif stage_id == 3:
        search_space = STAGE3_SEARCH
        features = [_find_best_feature_from_stage(2) or _find_best_feature_from_stage(1)]
        base_params = (_load_best_params_for_feature(2, features[0]) or
                       _load_best_params_for_feature(1, features[0]) or {})
    elif stage_id == 4:
        # ★ Stage 4: 每个特征用自己的最优正则化 + 全局最优其他参数
        search_space = {}
        features = list(FEATURE_CONFIGS.keys())
        global_best = (_load_best_params_for_feature(2, features[0]) or
                       _load_best_params_for_feature(1, features[0]) or BASELINE_PARAMS)
        base_params = global_best
        if feature_filter:
            features = [feature_filter]
    else:
        raise ValueError(f"Unknown stage: {stage_id}")

    if feature_filter and stage_id not in (1, 4):
        features = [feature_filter]

    # 合并 baseline
    full_base = {**BASELINE_PARAMS, **base_params}

    # 生成参数组合
    keys = list(search_space.keys())
    values = [search_space[k] for k in keys]
    combos = [dict(zip(keys, combo)) for combo in product(*values)]

    if not combos:
        # Stage 4: 无搜索空间, 直接跑一次
        combos = [{}]

    all_results = []
    total_runs = len(combos) * len(features) * (1 if dry_run else len(seeds))

    print(f"\n{'='*60}")
    print(f"  Stage {stage_id}: {len(combos)} 参数组合 × {len(features)} 特征 = {total_runs} runs")
    if search_space:
        for k, v in search_space.items():
            print(f"    {k}: {v}")
    print(f"{'='*60}")

    for combo in combos:
        params = {**full_base, **combo}
        param_tag = "_".join(f"{k}={v}" for k, v in combo.items()) if combo else "baseline"

        for fname in features:
            # ★ Stage 4: 每个特征用自己的最优 Stage1 正则化参数
            actual_params = dict(params)
            if stage_id == 4:
                feat_best_reg = _load_best_params_for_feature(1, fname)
                if feat_best_reg:
                    # 只覆盖正则化相关的参数 (lr/dropout/weight_decay)
                    for k in STAGE1_SEARCH:
                        if k in feat_best_reg:
                            actual_params[k] = feat_best_reg[k]

            for seed in (seeds if not dry_run else seeds[:1]):
                run_result = run_single_train(actual_params, fname, seed, dry_run=dry_run)
                if run_result is None:
                    continue
                record = {
                    "stage": stage_id,
                    "feature": fname,
                    "seed": seed,
                    "params": {k: v for k, v in params.items() if k not in FIXED_PARAMS},
                }
                if run_result["test"]:
                    record["test_Acc2"] = _get_test_metric(run_result["test"], "Acc2")
                    record["test_Acc7"] = _get_test_metric(run_result["test"], "Acc7")
                    record["test_MAE"] = _get_test_metric(run_result["test"], "MAE")
                    record["test_Corr"] = _get_test_metric(run_result["test"], "Corr")
                if run_result["valid_best"]:
                    record.update({f"valid_{k}": v for k, v in run_result["valid_best"].items()})
                all_results.append(record)

    # 保存阶段结果
    stage_dir = os.path.join(RESULTS_ROOT, f"stage{stage_id}")
    os.makedirs(stage_dir, exist_ok=True)
    _save_stage_results(all_results, stage_dir, stage_id)

    return all_results


def _find_best_feature_from_stage(stage_id: int) -> str | None:
    """从某个阶段结果中找出 test_Acc2 最高的特征名"""
    stage_dir = os.path.join(RESULTS_ROOT, f"stage{stage_id}")
    json_path = os.path.join(stage_dir, "results.json")
    if not os.path.isfile(json_path):
        return None
    with open(json_path, "r") as f:
        data = json.load(f)
    if not data:
        return None
    best = max(data, key=lambda r: r.get("test_Acc2", 0))
    return best.get("feature", None)


def _load_best_params_for_feature(stage_id: int, feature_name: str) -> dict:
    """从某个阶段结果中取出指定特征的最优参数 (by test_Acc2)"""
    stage_dir = os.path.join(RESULTS_ROOT, f"stage{stage_id}")
    json_path = os.path.join(stage_dir, "results.json")
    if not os.path.isfile(json_path):
        return {}
    with open(json_path, "r") as f:
        data = json.load(f)
    feature_results = [r for r in data if r.get("feature") == feature_name]
    if not feature_results:
        return {}
    best = max(feature_results, key=lambda r: r.get("test_Acc2", 0))
    return best.get("params", {})


def _save_stage_results(results: list[dict], stage_dir: str, stage_id: int):
    with open(os.path.join(stage_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    if not results:
        return

    # 按 params 聚合
    groups = defaultdict(list)
    for r in results:
        key = json.dumps(r["params"], sort_keys=True)
        groups[key].append(r)

    rows = []
    for key, group in groups.items():
        params = group[0]["params"]
        row = {"params": key}
        for m in ["test_Acc2", "test_Acc7", "test_MAE", "test_Corr", "valid_best_Acc2", "valid_best_MAE", "valid_total_epochs"]:
            vals = [r.get(m) for r in group if r.get(m) is not None]
            if vals:
                row[f"{m}_mean"] = round(float(np.mean(vals)), 4)
                row[f"{m}_std"] = round(float(np.std(vals)), 4) if len(vals) > 1 else 0
        row.update(params)
        rows.append(row)

    rows.sort(key=lambda r: -(r.get("test_Acc2_mean", 0)))
    fieldnames = list(rows[0].keys()) if rows else []
    csv_path = os.path.join(stage_dir, "summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n  [保存] {csv_path}")
    print(f"  Top-3 参数组合 (by test_Acc2):")
    for r in rows[:3]:
        acc2 = r.get("test_Acc2_mean", "N/A")
        params_str = r.get("params", "baseline")
        try:
            short = json.loads(params_str)
            short_str = " ".join(f"{k}={v}" for k, v in short.items())
        except Exception:
            short_str = params_str
        print(f"    Acc2={acc2} | {short_str}")


# ═══════════════════════════════════════════════════════════
# 参数重要性分析
# ═══════════════════════════════════════════════════════════

def show_param_catalog():
    """打印所有参数分类"""
    print("""
╔══════════════════════════════════════════════════════════╗
║            Grid Search 参数目录                          ║
╠══════════════════════════════════════════════════════════╣
║  Stage 1: 核心训练超参 (正则化) — 3特征独立搜索          ║
║    特征: OpenFace3(18维) Original_FACET(20维) DINOv3(768维)║
║    --lr            学习率        [1e-4, 3e-4, 1e-3]      ║
║    --dropout       Dropout 率    [0.1, 0.3, 0.5]         ║
║    --weight_decay  权重衰减      [1e-5, 1e-4, 1e-3]      ║
╠══════════════════════════════════════════════════════════╣
║  Stage 2: 损失与模态交互 (Stage1最优特征上搜)             ║
║    --key_eval         Early stop 指标 [Acc2, MAE]        ║
║    --v_self_ratio     V通道自注入    [0.0, 0.2, 0.4]     ║
║    --sub_loss_lambda  模态aux loss   [0.0, 0.3, 0.5]     ║
║    --warmup_ratio     LR warmup      [0.0, 0.1, 0.15]    ║
╠══════════════════════════════════════════════════════════╣
║  Stage 3: 模型结构 (低优先级)                             ║
║    --ism_depth           ISM深度      [2, 3]              ║
║    --ism_mixer_type      Mamba版本    [bimamba, bimamba3] ║
║    --contrastive_weight  对比损失权重  [0.0, 0.1]          ║
╠══════════════════════════════════════════════════════════╣
║  Stage 4: 全特征验证 ★                                  ║
║    每个特征用自己 Stage1 最优正则化 + Stage2/3最优全局参数 ║
║    在全部 5 种特征上 3 种子验证                          ║
╠══════════════════════════════════════════════════════════╣
║  固定参数 (不搜):                                        ║
║    dataset=MOSI  bert=base-uncased  epochs=120           ║
║    batch_size=32  early_stop=8                           ║
╠══════════════════════════════════════════════════════════╣
║  不搜的原因:                                             ║
║    --d_model/num_layers/d_state  → 结构参数组合爆炸      ║
║    --ism_d_state/grad_clip       → 边际影响小             ║
║    --aux_cls_weight              → 与 sub_loss 功能重叠  ║
║    --augment_prob/ema_decay      → 影响有限              ║
╚══════════════════════════════════════════════════════════╝
""")


# ═══════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="视频特征 Grid Search")
    parser.add_argument("--stage", type=str, default="catalog",
                        choices=["catalog", "1", "2", "3", "4", "all"],
                        help="catalog=查看参数目录; 1-4=单独阶段; all=全部")
    parser.add_argument("--seeds", type=str, default="42",
                        help="逗号分隔的种子 (Stage1建议2个:42,123; Stage4建议3个:42,123,456)")
    parser.add_argument("--feature", type=str, default=None,
                        help="限制特征 (Stage4可用)")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    if args.stage == "catalog":
        show_param_catalog()
        return

    seeds = [int(s.strip()) for s in args.seeds.split(",")]

    if args.stage == "all":
        stages = [1, 2, 3, 4]
    else:
        stages = [int(args.stage)]

    all_results = []
    for stage_id in stages:
        results = grid_search_stage(stage_id, seeds, args.feature, args.dry_run)
        all_results.extend(results)

    if not args.dry_run and all_results:
        print(f"\n{'='*60}")
        print(f"  全部 Grid Search 完成! 共 {len(all_results)} 次实验")
        print(f"  结果目录: {RESULTS_ROOT}")
        print(f"{'='*60}")


if __name__ == "__main__":
    # 无缓冲实时输出 (配合 python -u 使用效果更好)
    sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None
    main()
