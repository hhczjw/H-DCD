"""
多 ckpt 集成推理 (logits 平均).

用法:
    python scripts/ensemble.py \
        --dataset MOSI \
        --ckpts checkpoints/MOSI_D_alpha03_ema_seed42_seed42_best_MAE.pt \
                checkpoints/MOSI_D_alpha03_ema_seed2024_seed2024_best_MAE.pt \
                checkpoints/MOSI_D_alpha03_ema_seed0_seed0_best_MAE.pt \
        --tag D_ensemble3

行为:
    1. 用每个 ckpt 内嵌的 args 重建模型, load 权重
    2. 在 test set 上各 forward 一次, 得到 N 组 (preds_i,)
    3. 取 logits 平均 (回归就是 score 平均) 作为最终预测
    4. 跑标准 eval_regression
"""
from __future__ import annotations

import os
import sys

# 把项目根目录 (scripts/ 的上一级) 加入 sys.path, 保证能 import 项目模块
_PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)
import _path_setup  # noqa: F401  必须早于 mamba_ssm/项目模块的 import

import argparse
import json
from types import SimpleNamespace

import numpy as np
import torch

from configs import load_config
from dataset import MMDataLoader
from models import MSAClassifier
from utils.metrics import eval_regression


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="MOSI")
    p.add_argument("--ckpts", type=str, nargs="+", required=True)
    p.add_argument("--tag", type=str, default="ensemble")
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def build_model_from_args(a):
    """根据 ckpt 内嵌 args 重建 MSAClassifier."""
    g = lambda k, d=None: getattr(a, k, d) if hasattr(a, k) else (a[k] if isinstance(a, dict) and k in a else d)
    text_dim, audio_dim, video_dim = g("feature_dims")
    ism_seq_len = int(g("ism_seq_len", g("seq_lens", [50])[0]))
    return MSAClassifier(
        text_input_dim=text_dim,
        audio_input_dim=audio_dim,
        video_input_dim=video_dim,
        d_model=g("d_model"),
        num_layers=g("num_layers"),
        num_classes=g("num_classes"),
        task_type=g("task_type"),
        pool_type=g("pool_type", "attention"),
        dropout=g("dropout", 0.3),
        use_bert=g("use_bert", True),
        bert_pretrained=g("bert_pretrained", "bert-base-uncased"),
        bert_finetune=g("bert_finetune", True),
        ism_depth=int(g("ism_depth", 2)),
        ism_seq_len=ism_seq_len,
        ism_d_state=int(g("ism_d_state", 64)),
        ism_mixer_type=g("ism_mixer_type", "bimamba"),
        ism_bimamba3_headdim=int(g("ism_bimamba3_headdim", 32)),
        ism_bimamba3_ngroups=int(g("ism_bimamba3_ngroups", 1)),
        ism_bimamba3_rope_fraction=float(g("ism_bimamba3_rope_fraction", 0.5)),
        ism_bimamba3_chunk_size=int(g("ism_bimamba3_chunk_size", 64)),
        ism_bimamba3_is_mimo=bool(g("ism_bimamba3_is_mimo", False)),
        ism_bimamba3_mimo_rank=int(g("ism_bimamba3_mimo_rank", 4)),
        ism_bimamba3_is_outproj_norm=bool(g("ism_bimamba3_is_outproj_norm", False)),
        ism_bimamba3_fusion=g("ism_bimamba3_fusion", "add_divide2"),
        ism_bimamba3_share_mimo=bool(g("ism_bimamba3_share_mimo", True)),
        d_state=g("d_state"),
        expand=g("expand"),
        headdim=g("headdim"),
        ngroups=g("ngroups"),
        rope_fraction=g("rope_fraction"),
        is_mimo=g("is_mimo"),
        mimo_rank=g("mimo_rank"),
        chunk_size=g("chunk_size"),
        is_outproj_norm=g("is_outproj_norm", False),
        aux_num_classes=int(g("aux_num_classes", 0)),
    )


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    all_p, all_t = [], []
    for batch in loader:
        text = batch["text"]
        audio = batch["audio"]
        vision = batch["vision"]
        if isinstance(text, dict):
            text = {k: v.to(device, non_blocking=True) for k, v in text.items()}
        else:
            text = text.to(device, non_blocking=True)
        audio = audio.to(device, non_blocking=True)
        vision = vision.to(device, non_blocking=True)
        out = model(text=text, audio=audio, video=vision)
        if isinstance(out, dict):
            logits = out["logits"]
        else:
            logits = out
        all_p.append(logits.squeeze(-1).cpu().numpy())
        all_t.append(batch["labels"]["M"].cpu().numpy())
    preds = np.concatenate(all_p, axis=0)
    truths = np.concatenate(all_t, axis=0)
    return preds, truths


def main():
    cli = parse_args()
    device = torch.device(cli.device if torch.cuda.is_available() else "cpu")

    args = load_config(cli.dataset)
    loaders = MMDataLoader(args, num_workers=int(args.num_workers))

    all_preds = []
    truths_ref = None
    for i, ckpt_path in enumerate(cli.ckpts):
        print(f"\n[{i+1}/{len(cli.ckpts)}] Loading: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        ckpt_args = ckpt.get("args", vars(args))
        if not isinstance(ckpt_args, dict):
            ckpt_args = vars(ckpt_args)
        # 用 ckpt 内嵌 args 重建模型 (保证结构一致)
        a_ns = SimpleNamespace(**ckpt_args)
        model = build_model_from_args(a_ns).to(device)
        model.load_state_dict(ckpt["model"])
        if ckpt.get("is_ema", False):
            print("  >> ckpt 是 EMA 权重")
        preds, truths = predict(model, loaders["test"], device)
        m_single = eval_regression(preds, truths)
        print(f"  single Acc2={m_single['Acc2']:.4f} Acc7={m_single['Acc7']:.4f} MAE={m_single['MAE']:.4f}")
        all_preds.append(preds)
        if truths_ref is None:
            truths_ref = truths
        del model
        torch.cuda.empty_cache()

    # logits/score 平均
    preds_ens = np.mean(np.stack(all_preds, axis=0), axis=0)
    metrics = eval_regression(preds_ens, truths_ref)

    print("\n" + "=" * 60)
    print(f"=== Ensemble of {len(cli.ckpts)} ckpts ===")
    print("=" * 60)
    for k, v in metrics.items():
        print(f"  {k:<10} = {v:.4f}")

    out_path = os.path.join(args.results_dir, f"{cli.dataset}_{cli.tag}_test.json")
    os.makedirs(args.results_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"ensemble": metrics, "n_ckpts": len(cli.ckpts), "ckpts": cli.ckpts}, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()