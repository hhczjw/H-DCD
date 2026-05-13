"""
评估指标:
    - eval_regression:  MOSI/MOSEI (MAE, Corr, Acc-2, Acc-5, Acc-7, F1)
    - eval_classification:  IEMOCAP/MELD (Accuracy, Weighted F1, Macro F1)
"""
from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def _multiclass_acc(preds: np.ndarray, truths: np.ndarray) -> float:
    return float(np.sum(np.round(preds) == np.round(truths))) / float(len(truths))


def eval_regression(preds: np.ndarray, truths: np.ndarray) -> Dict[str, float]:
    """
    MOSEI / MOSI 标准指标.
    preds, truths: (N,) float
    """
    preds = np.asarray(preds).reshape(-1)
    truths = np.asarray(truths).reshape(-1)

    mae = float(np.mean(np.abs(preds - truths)))
    # 审查4 修复: 早期 epoch / 退化模型 preds 或 truths 方差为 0 时 corrcoef → NaN
    if len(preds) > 1:
        # 任一序列方差≈0 直接报 0 (相关无意义)
        p_std = float(np.std(preds))
        t_std = float(np.std(truths))
        if p_std < 1e-8 or t_std < 1e-8:
            corr = 0.0
        else:
            c = np.corrcoef(preds, truths)[0, 1]
            corr = float(c) if np.isfinite(c) else 0.0
    else:
        corr = 0.0

    # Acc-7: [-3,3] 离散
    preds_a7 = np.clip(preds, a_min=-3.0, a_max=3.0)
    truths_a7 = np.clip(truths, a_min=-3.0, a_max=3.0)
    acc7 = _multiclass_acc(preds_a7, truths_a7)

    # Acc-5
    preds_a5 = np.clip(preds, a_min=-2.0, a_max=2.0)
    truths_a5 = np.clip(truths, a_min=-2.0, a_max=2.0)
    acc5 = _multiclass_acc(preds_a5, truths_a5)

    # ===== Acc2 / F1 双口径 =====
    # (1) Non0 (Self-MM/MISA/MMIM 标准): 排除 label==0 的中性样本
    non_zeros = np.array([i for i, e in enumerate(truths) if e != 0])
    if len(non_zeros) > 0:
        bp = (preds[non_zeros] > 0).astype(int)
        bt = (truths[non_zeros] > 0).astype(int)
        acc2_non0 = float(accuracy_score(bt, bp))
        f1_non0 = float(f1_score(bt, bp, average="weighted"))
    else:
        acc2_non0, f1_non0 = 0.0, 0.0

    # (2) Has0 (TFN/MMIM 报点): 包含全部样本, 规则 pred>=0 -> 正
    bp_h = (preds >= 0).astype(int)
    bt_h = (truths >= 0).astype(int)
    acc2_has0 = float(accuracy_score(bt_h, bp_h))
    f1_has0 = float(f1_score(bt_h, bp_h, average="weighted"))

    return {
        "MAE": mae, "Corr": corr,
        "Acc2": acc2_non0, "F1": f1_non0,           # 主口径 (向后兼容, KeyEval 仍能用)
        "Acc2_has0": acc2_has0, "F1_has0": f1_has0, # 辅助口径
        "Acc5": acc5, "Acc7": acc7,
    }


def eval_classification(preds: np.ndarray, truths: np.ndarray) -> Dict[str, float]:
    """
    IEMOCAP / MELD.
    preds: (N, C) logits 或 (N,) 类别;  truths: (N,) int
    """
    preds = np.asarray(preds)
    truths = np.asarray(truths).reshape(-1).astype(int)
    if preds.ndim == 2:
        preds = preds.argmax(axis=-1)
    acc = float(accuracy_score(truths, preds))
    f1_w = float(f1_score(truths, preds, average="weighted", zero_division=0))
    f1_m = float(f1_score(truths, preds, average="macro", zero_division=0))
    return {"Acc": acc, "F1_weighted": f1_w, "F1_macro": f1_m, "F1": f1_w}