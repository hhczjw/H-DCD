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
    corr = float(np.corrcoef(preds, truths)[0, 1]) if len(preds) > 1 else 0.0

    # Acc-7: [-3,3] 离散
    preds_a7 = np.clip(preds, a_min=-3.0, a_max=3.0)
    truths_a7 = np.clip(truths, a_min=-3.0, a_max=3.0)
    acc7 = _multiclass_acc(preds_a7, truths_a7)

    # Acc-5
    preds_a5 = np.clip(preds, a_min=-2.0, a_max=2.0)
    truths_a5 = np.clip(truths, a_min=-2.0, a_max=2.0)
    acc5 = _multiclass_acc(preds_a5, truths_a5)

    # Binary (>=0 / <0), 排除 0
    non_zeros = np.array([i for i, e in enumerate(truths) if e != 0])
    if len(non_zeros) > 0:
        binary_preds = (preds[non_zeros] > 0).astype(int)
        binary_truths = (truths[non_zeros] > 0).astype(int)
        acc2 = float(accuracy_score(binary_truths, binary_preds))
        f1 = float(f1_score(binary_truths, binary_preds, average="weighted"))
    else:
        acc2, f1 = 0.0, 0.0

    return {"MAE": mae, "Corr": corr, "Acc2": acc2, "Acc5": acc5, "Acc7": acc7, "F1": f1}


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