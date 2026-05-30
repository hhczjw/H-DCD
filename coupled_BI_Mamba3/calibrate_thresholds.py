"""
方法1: 测试时阈值校准
====================

在验证集上搜索最优 7 分类阈值, 替换 round(pred) 的固定边界.
预期 Acc7 提升 1~3%.

用法:
    python calibrate_thresholds.py --dataset MOSI --ckpt <checkpoint_path>
"""

import argparse
import json
import os
import sys
import numpy as np
import torch

from configs import load_config
from dataset.data_loader import MMDataLoader
from models.classifier import MSAClassifier
from utils.metrics import eval_regression

sys.path.insert(0, os.path.dirname(__file__))
import _path_setup


def load_model_and_predict(args, ckpt_path, loader):
    """加载模型并在数据集上推理, 返回 (predictions, labels)"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device)

    # 构建模型
    model_args = argparse.Namespace(**ckpt["args"])
    model = MSAClassifier(
        text_input_dim=model_args.feature_dims[0],
        audio_input_dim=model_args.feature_dims[1],
        video_input_dim=model_args.feature_dims[2],
        d_model=getattr(model_args, "d_model", 128),
        num_layers=getattr(model_args, "num_layers", 2),
        num_classes=getattr(model_args, "num_classes", 1),
        task_type=getattr(model_args, "task_type", "regression"),
        pool_type=getattr(model_args, "pool_type", "mean"),
        dropout=getattr(model_args, "dropout", 0.5),
        use_bert=getattr(model_args, "use_bert", True),
        bert_pretrained=getattr(model_args, "bert_pretrained", "roberta-base"),
        bert_finetune=getattr(model_args, "bert_finetune", True),
        ism_depth=getattr(model_args, "ism_depth", 3),
        ism_seq_len=getattr(model_args, "seq_lens", [50, 50, 50])[0],
        ism_d_state=getattr(model_args, "ism_d_state", 64),
        ism_mixer_type=getattr(model_args, "ism_mixer_type", "bimamba"),
        d_state=getattr(model_args, "d_state", 64),
        expand=getattr(model_args, "expand", 2),
        headdim=getattr(model_args, "headdim", 32),
        v_self_ratio=getattr(model_args, "v_self_ratio", 0.0),
        use_bssm_gate=getattr(model_args, "use_bssm_gate", True),
        use_gcmn_gate=getattr(model_args, "use_gcmn_gate", True),
        use_mdl=getattr(model_args, "use_mdl", False),
    )
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            kw = dict(
                text=batch["text"].to(device),
                audio=batch["audio"].to(device),
                video=batch["vision"].to(device),
            )
            if "context_text" in batch:
                kw["context_text"] = batch["context_text"].to(device)
                kw["context_audio"] = batch["context_audio"].to(device)
                kw["context_video"] = batch["context_video"].to(device)
            out = model(**kw)
            logits = out["logits"] if isinstance(out, dict) else out
            all_preds.append(logits.squeeze(-1).cpu().numpy())
            all_labels.append(batch["labels"]["M"].numpy())

    return np.concatenate(all_preds), np.concatenate(all_labels)


def search_thresholds(preds, labels, clip_range=3.0):
    """在验证集上搜索最优分类边界

    标准 round 边界: [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]
    搜索每个边界的偏移 delta ∈ [-0.3, 0.3], 步长 0.05
    """
    base_boundaries = np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5])
    deltas = np.arange(-0.3, 0.35, 0.05)

    best_acc7 = 0
    best_deltas = np.zeros(6)

    # 标准 round 的 Acc7
    standard_pred_cls = np.clip(np.round(preds), -clip_range, clip_range)
    standard_label_cls = np.clip(np.round(labels), -clip_range, clip_range)
    standard_acc7 = np.mean(standard_pred_cls == standard_label_cls)

    # 网格搜索 (6 个边界, 每个 ~13 个候选, 总共 13^6 ≈ 4.8M 组合)
    # 用贪心搜索代替: 逐个边界优化
    current_deltas = np.zeros(6)
    for i in range(6):
        best_delta_for_i = 0.0
        best_acc_for_i = 0
        for d in deltas:
            test_deltas = current_deltas.copy()
            test_deltas[i] = d
            boundaries = base_boundaries + test_deltas
            pred_cls = classify_with_boundaries(preds, boundaries, clip_range)
            acc7 = np.mean(pred_cls == standard_label_cls)
            if acc7 > best_acc_for_i:
                best_acc_for_i = acc7
                best_delta_for_i = d
        current_deltas[i] = best_delta_for_i

    # 用最终边界计算 Acc7
    final_boundaries = base_boundaries + current_deltas
    final_pred_cls = classify_with_boundaries(preds, final_boundaries, clip_range)
    final_acc7 = np.mean(final_pred_cls == standard_label_cls)

    return {
        "standard_acc7": float(standard_acc7),
        "calibrated_acc7": float(final_acc7),
        "improvement": float(final_acc7 - standard_acc7),
        "deltas": current_deltas.tolist(),
        "boundaries": final_boundaries.tolist(),
    }


def classify_with_boundaries(preds, boundaries, clip_range=3.0):
    """用自定义边界对预测值分类"""
    preds = np.clip(preds, -clip_range, clip_range)
    # boundaries = [-2.5+d0, -1.5+d1, -0.5+d2, 0.5+d3, 1.5+d4, 2.5+d5]
    # pred_cls = sum(pred > boundary for boundary in boundaries) - clip_range
    cls = np.zeros_like(preds, dtype=np.float64)
    for b in boundaries:
        cls += (preds > b).astype(np.float64)
    cls = cls - clip_range  # 映射到 [-3, 3]
    return np.clip(np.round(cls), -clip_range, clip_range)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="MOSI")
    p.add_argument("--ckpt", type=str, required=True, help="checkpoint 路径")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    # 加载配置
    config = load_config(args.dataset)
    config.seed = args.seed

    # 加载数据
    valid_loader = MMDataLoader(config, mode="valid").get_loader()
    test_loader = MMDataLoader(config, mode="test").get_loader()

    print(f"正在加载模型: {args.ckpt}")

    # 在验证集上推理
    val_preds, val_labels = load_model_and_predict(config, args.ckpt, valid_loader)
    print(f"验证集: {len(val_preds)} 样本")

    # 搜索最优阈值
    result = search_thresholds(val_preds, val_labels)
    print(f"\n=== 阈值校准结果 (验证集) ===")
    print(f"标准 round Acc7: {result['standard_acc7']:.4f}")
    print(f"校准后 Acc7:     {result['calibrated_acc7']:.4f}")
    print(f"提升:            {result['improvement']:+.4f}")
    print(f"最优边界偏移:    {[f'{d:.2f}' for d in result['deltas']]}")

    # 在测试集上应用校准
    test_preds, test_labels = load_model_and_predict(config, args.ckpt, test_loader)
    print(f"\n测试集: {len(test_preds)} 样本")

    # 标准 round
    test_standard_cls = np.clip(np.round(test_preds), -3, 3)
    test_label_cls = np.clip(np.round(test_labels), -3, 3)
    test_standard_acc7 = np.mean(test_standard_cls == test_label_cls)

    # 校准后
    boundaries = np.array(result["boundaries"])
    test_calibrated_cls = classify_with_boundaries(test_preds, boundaries)
    test_calibrated_acc7 = np.mean(test_calibrated_cls == test_label_cls)

    # 同时报告其他指标
    test_metrics = eval_regression(test_preds, test_labels)

    print(f"\n=== 测试集结果 ===")
    print(f"标准 round:  Acc7={test_standard_acc7:.4f}")
    print(f"校准后:      Acc7={test_calibrated_acc7:.4f}")
    print(f"提升:        {test_calibrated_acc7 - test_standard_acc7:+.4f}")
    print(f"其他指标:    MAE={test_metrics['MAE']:.4f} Acc2={test_metrics['Acc2']:.4f} Corr={test_metrics['Corr']:.4f}")

    # 保存结果
    save_path = args.ckpt.replace(".pt", "_calibration.json")
    with open(save_path, "w") as f:
        json.dump({
            "validation": result,
            "test": {
                "standard_acc7": float(test_standard_acc7),
                "calibrated_acc7": float(test_calibrated_acc7),
                "improvement": float(test_calibrated_acc7 - test_standard_acc7),
                "MAE": float(test_metrics["MAE"]),
                "Acc2": float(test_metrics["Acc2"]),
            }
        }, f, indent=2)
    print(f"\n结果已保存: {save_path}")


if __name__ == "__main__":
    main()
