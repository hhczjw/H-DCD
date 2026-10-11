#!/usr/bin/env python3
"""
二分类阈值校准脚本 (P0-2)
=========================

背景 (2026-09-18 诊断):
  - 官方 has0 口径判定规则: pred >= 0 -> 正类; 零点样本算正类。
  - 本项目零点样本错分率约 40~60% (非零点仅 ~10%), 导致 Acc2_has0/F1_has0
    系统性落后 ~1.8 分。
  - 本脚本在 **验证集** 上搜索最优二分类边界 tau* (判定规则 pred >= tau),
    分别以 val F1_has0 / val Acc2_has0 为目标, 再把同一 tau* 应用到测试集。

⚠️ 协议要求 (必须遵守, 否则测试集泄漏):
  - tau 只能在校验集上搜索, 测试集只做推理;
  - 论文/汇报中必须披露 "threshold tuned on validation set"。

用法:
  python scripts/calibrate_binary_threshold.py \
      --ckpt checkpoints/MOSI_xxx_seed42_best_F1_has0.pt \
      --ckpt checkpoints/MOSI_xxx_seed42_best_Acc2.pt

  多 ckpt 时: 逐个输出, 最后追加 "预测平均集成" 的校准结果。

输出:
  - 终端打印 校准前后对比表 (τ=0 官方口径 vs τ* 校准口径)
  - results/<stem>_binary_calib.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from argparse import Namespace

import numpy as np
import torch
from sklearn.metrics import f1_score

# 让脚本可从项目根目录或 scripts/ 下运行
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from dataset.data_loader import MMDataLoader      # noqa: E402
from models.classifier import MSAClassifier       # noqa: E402


# ----------------------------------------------------------------------
# 模型构建 (与 train.py 完全一致的参数映射)
# ----------------------------------------------------------------------
def build_model(model_args: Namespace) -> MSAClassifier:
    # 外部特征文件 → 自动更新 feature_dims (与 train.py 相同)
    for attr, idx, key in (("feature_A", 1, "audio"), ("feature_V", 2, "vision")):
        path = getattr(model_args, attr, "")
        if not path or not os.path.isfile(path):
            continue
        import pickle
        with open(path, "rb") as f:
            ext = pickle.load(f)
        first = next(iter(ext.keys()))
        sample = ext[first]
        if key in sample:
            model_args.feature_dims[idx] = int(np.asarray(sample[key]).shape[-1])

    text_dim, audio_dim, video_dim = model_args.feature_dims
    ism_seq_len = int(getattr(model_args, "ism_seq_len", model_args.seq_lens[0]))
    model = MSAClassifier(
        text_input_dim=text_dim,
        audio_input_dim=audio_dim,
        video_input_dim=video_dim,
        d_model=model_args.d_model,
        num_layers=model_args.num_layers,
        num_classes=model_args.num_classes,
        task_type=model_args.task_type,
        pool_type=model_args.pool_type,
        dropout=model_args.dropout,
        use_bert=model_args.use_bert,
        bert_pretrained=getattr(model_args, "bert_pretrained", "bert-base-uncased"),
        bert_finetune=getattr(model_args, "bert_finetune", True),
        ism_depth=int(getattr(model_args, "ism_depth", 1)),
        ism_seq_len=ism_seq_len,
        ism_d_state=int(getattr(model_args, "ism_d_state", 16)),
        ism_mixer_type=getattr(model_args, "ism_mixer_type", "bimamba"),
        ism_bimamba3_headdim=int(getattr(model_args, "ism_bimamba3_headdim", 64)),
        ism_bimamba3_ngroups=int(getattr(model_args, "ism_bimamba3_ngroups", 1)),
        ism_bimamba3_rope_fraction=float(getattr(model_args, "ism_bimamba3_rope_fraction", 0.5)),
        ism_bimamba3_chunk_size=int(getattr(model_args, "ism_bimamba3_chunk_size", 64)),
        ism_bimamba3_is_mimo=bool(getattr(model_args, "ism_bimamba3_is_mimo", False)),
        ism_bimamba3_mimo_rank=int(getattr(model_args, "ism_bimamba3_mimo_rank", 4)),
        ism_bimamba3_is_outproj_norm=bool(getattr(model_args, "ism_bimamba3_is_outproj_norm", False)),
        ism_bimamba3_fusion=getattr(model_args, "ism_bimamba3_fusion", "add_divide2"),
        ism_bimamba3_share_mimo=bool(getattr(model_args, "ism_bimamba3_share_mimo", True)),
        use_bssm_gate=bool(getattr(model_args, "use_bssm_gate", False)),
        bssm_gate_expand=int(getattr(model_args, "bssm_gate_expand", 2)),
        use_gcmn_gate=bool(getattr(model_args, "use_gcmn_gate", False)),
        bidirectional=bool(getattr(model_args, "bidirectional", False)),
        d_state=model_args.d_state,
        expand=model_args.expand,
        headdim=model_args.headdim,
        ngroups=model_args.ngroups,
        rope_fraction=model_args.rope_fraction,
        is_mimo=model_args.is_mimo,
        mimo_rank=model_args.mimo_rank,
        chunk_size=model_args.chunk_size,
        is_outproj_norm=model_args.is_outproj_norm,
        v_self_ratio=float(getattr(model_args, "v_self_ratio", 0.0) or 0.0),
        use_pairwise_mamba=getattr(model_args, "use_pairwise_mamba", False),
        multi_task=bool(getattr(model_args, "multi_task", False)),
        use_pretrained_audio=bool(getattr(model_args, "use_pretrained_audio", False)),
        audio_pretrained=getattr(model_args, "audio_pretrained",
                                 "facebook/data2vec-audio-base-960h"),
        audio_finetune=bool(getattr(model_args, "audio_finetune", True)),
        skip_audio_ism=bool(getattr(model_args, "skip_audio_ism", False)),
    )
    if getattr(model_args, "ism_full_frame", False):
        model._ism_full_frame = True
    return model


# ----------------------------------------------------------------------
# 推理
# ----------------------------------------------------------------------
def predict(model: MSAClassifier, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in loader:
            kw = dict(
                text=batch["text"].to(device),
                audio=batch["audio"].to(device),
                video=batch["vision"].to(device),
            )
            if "audio_lengths" in batch:
                kw["audio_lengths"] = batch["audio_lengths"].to(device)
            if "vision_lengths" in batch:
                kw["vision_lengths"] = batch["vision_lengths"].to(device)
            if "context_text" in batch:
                kw["context_text"] = batch["context_text"].to(device)
                kw["context_audio"] = batch["context_audio"].to(device)
                kw["context_video"] = batch["context_video"].to(device)
            out = model(**kw)
            logits = out["logits"] if isinstance(out, dict) else out
            preds.append(logits.squeeze(-1).detach().cpu().numpy())
            labels.append(batch["labels"]["M"].cpu().numpy())
    return np.concatenate(preds).reshape(-1), np.concatenate(labels).reshape(-1)


# ----------------------------------------------------------------------
# 指标与扫描
# ----------------------------------------------------------------------
def metrics_at(preds: np.ndarray, labels: np.ndarray, tau: float) -> tuple[float, float]:
    """has0 口径: 正类 = labels >= 0; 判定 = preds >= tau."""
    y = (labels >= 0).astype(int)
    p = (preds >= tau).astype(int)
    acc2_has0 = float((y == p).mean())
    f1_has0 = float(f1_score(y, p, average="weighted", zero_division=0))
    return acc2_has0, f1_has0


def sweep(preds: np.ndarray, labels: np.ndarray, tau_min: float, tau_max: float, tau_step: float):
    taus = np.arange(tau_min, tau_max + 1e-9, tau_step)
    rows = []
    for t in taus:
        acc, f1 = metrics_at(preds, labels, float(t))
        rows.append((float(t), acc, f1))
    return rows


def pick_tau(rows, key: str):
    """key: 'f1' 或 'acc'; 并列时取 |τ| 最小 (更保守)."""
    idx = 1 if key == "acc" else 2
    best = max(rows, key=lambda r: (r[idx], -abs(r[0])))
    return best[0], best[idx]


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="二分类阈值校准 (验证集搜索, 测试集应用)")
    ap.add_argument("--ckpt", type=str, nargs="+", action="append", required=True,
                    help="checkpoint 路径; 支持 `--ckpt a.pt b.pt` 或重复 `--ckpt a.pt --ckpt b.pt`")
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--tau-min", type=float, default=-1.0)
    ap.add_argument("--tau-max", type=float, default=1.0)
    ap.add_argument("--tau-step", type=float, default=0.02)
    ap.add_argument("--save-json", type=str, default=None,
                    help="结果 JSON 路径; 默认 results/<第一个ckpt名>_binary_calib.json")
    cli = ap.parse_args()
    # 展平: 兼容 `--ckpt a b` 与重复 `--ckpt a --ckpt b` 两种写法
    cli.ckpt = [p for group in cli.ckpt for p in group]

    device = torch.device(cli.device if torch.cuda.is_available() else "cpu")

    print("=" * 72)
    print("⚠️  协议提醒: tau 仅在验证集搜索, 测试集只做推理; 汇报/论文需披露本协议。")
    print("=" * 72)

    # 用第一个 ckpt 的 args 构建数据管线 (特征路径/序列长度等)
    ref = torch.load(cli.ckpt[0], map_location="cpu", weights_only=False)
    model_args = Namespace(**ref["args"])
    loaders = MMDataLoader(model_args, num_workers=cli.num_workers)
    valid_loader, test_loader = loaders["valid"], loaders["test"]

    all_results = {}
    val_vecs, test_vecs = {}, {}

    for ckpt_path in cli.ckpt:
        name = os.path.basename(ckpt_path).replace(".pt", "")
        print(f"\n>>> 加载 {ckpt_path}")
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = build_model(Namespace(**state["args"])).to(device)
        model.load_state_dict(state["model"])
        model.eval()

        val_p, val_y = predict(model, valid_loader, device)
        test_p, test_y = predict(model, test_loader, device)
        val_vecs[name], test_vecs[name] = val_p, test_p
        del model
        torch.cuda.empty_cache()

        rows = sweep(val_p, val_y, cli.tau_min, cli.tau_max, cli.tau_step)
        tau_f1, v_f1 = pick_tau(rows, "f1")
        tau_acc, v_acc = pick_tau(rows, "acc")
        if abs(tau_f1) > 0.5 or abs(tau_acc) > 0.5:
            print("  ⚠️ τ* 漂移很大 (|τ|>0.5), 校准可能不稳定: 请核对验证集尾部样本分布, "
                  "必要时放弃校准并用 τ=0 原始口径上报。")

        base = metrics_at(test_p, test_y, 0.0)
        t_f1 = metrics_at(test_p, test_y, tau_f1)
        t_acc = metrics_at(test_p, test_y, tau_acc)

        print(f"  [valid ] τ=0 基线:  Acc2_has0={[m for m in rows if abs(m[0])<1e-9][0][1]:.4f}  F1_has0={[m for m in rows if abs(m[0])<1e-9][0][2]:.4f}")
        print(f"  最优 τ(F1) ={tau_f1:+.2f} (val F1_has0={v_f1:.4f}) | 最优 τ(Acc)={tau_acc:+.2f} (val Acc2_has0={v_acc:.4f})")
        print(f"  [test  ] τ=0:      Acc2_has0={base[0]:.4f}  F1_has0={base[1]:.4f}")
        print(f"  [test  ] τ(F1):    Acc2_has0={t_f1[0]:.4f}  F1_has0={t_f1[1]:.4f}   "
              f"(ΔF1={t_f1[1]-base[1]:+.4f}, ΔAcc={t_f1[0]-base[0]:+.4f})")
        print(f"  [test  ] τ(Acc):   Acc2_has0={t_acc[0]:.4f}  F1_has0={t_acc[1]:.4f}   "
              f"(ΔF1={t_acc[1]-base[1]:+.4f}, ΔAcc={t_acc[0]-base[0]:+.4f})")

        all_results[name] = {
            "tau_f1": tau_f1, "tau_acc": tau_acc,
            "valid_f1_has0_at_tau_f1": v_f1, "valid_acc2_has0_at_tau_acc": v_acc,
            "test_tau0": {"Acc2_has0": base[0], "F1_has0": base[1]},
            "test_tau_f1": {"Acc2_has0": t_f1[0], "F1_has0": t_f1[1]},
            "test_tau_acc": {"Acc2_has0": t_acc[0], "F1_has0": t_acc[1]},
        }

    # ---- 多 ckpt 预测平均集成 ----
    if len(cli.ckpt) > 1:
        print("\n>>> 预测平均集成 (所有 ckpt)")
        ens_val = np.mean(np.stack(list(val_vecs.values())), axis=0)
        ens_test = np.mean(np.stack(list(test_vecs.values())), axis=0)
        rows = sweep(ens_val, val_y, cli.tau_min, cli.tau_max, cli.tau_step)
        tau_f1, v_f1 = pick_tau(rows, "f1")
        tau_acc, v_acc = pick_tau(rows, "acc")
        if abs(tau_f1) > 0.5 or abs(tau_acc) > 0.5:
            print("  ⚠️ τ* 漂移很大 (|τ|>0.5), 校准可能不稳定。")
        base = metrics_at(ens_test, test_y, 0.0)
        t_f1 = metrics_at(ens_test, test_y, tau_f1)
        t_acc = metrics_at(ens_test, test_y, tau_acc)
        print(f"  [valid ] τ=0:      F1_has0={[m for m in rows if abs(m[0])<1e-9][0][2]:.4f}")
        print(f"  最优 τ(F1)={tau_f1:+.2f} (val F1_has0={v_f1:.4f}) | 最优 τ(Acc)={tau_acc:+.2f} (val Acc2_has0={v_acc:.4f})")
        print(f"  [test  ] τ=0:      Acc2_has0={base[0]:.4f}  F1_has0={base[1]:.4f}")
        print(f"  [test  ] τ(F1):    Acc2_has0={t_f1[0]:.4f}  F1_has0={t_f1[1]:.4f}   "
              f"(ΔF1={t_f1[1]-base[1]:+.4f}, ΔAcc={t_f1[0]-base[0]:+.4f})")
        all_results["ensemble_avg"] = {
            "n_ckpts": len(cli.ckpt),
            "tau_f1": tau_f1, "tau_acc": tau_acc,
            "valid_f1_has0_at_tau_f1": v_f1, "valid_acc2_has0_at_tau_acc": v_acc,
            "test_tau0": {"Acc2_has0": base[0], "F1_has0": base[1]},
            "test_tau_f1": {"Acc2_has0": t_f1[0], "F1_has0": t_f1[1]},
            "test_tau_acc": {"Acc2_has0": t_acc[0], "F1_has0": t_acc[1]},
        }

    # ---- 保存 ----
    stem = os.path.basename(cli.ckpt[0]).replace(".pt", "")
    save_path = cli.save_json or os.path.join(_ROOT, "results", f"{stem}_binary_calib.json")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump({
            "protocol": "tau tuned on validation set, applied to test set (disclose in paper)",
            "ckpts": cli.ckpt,
            "tau_grid": {"min": cli.tau_min, "max": cli.tau_max, "step": cli.tau_step},
            "results": all_results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {save_path}")


if __name__ == "__main__":
    main()
