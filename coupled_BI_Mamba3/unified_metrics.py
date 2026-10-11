#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
统一指标口径（唯一真源 / single source of truth）
=================================================

所有复现模型的指标必须与此处定义一致。本文件由 `utils/metrics.py::eval_regression`
抽出并固定，作为跨仓库重算的基准实现。

口径定义（CMU-MOSI / CMU-MOSEI 回归任务）
------------------------------------------
  MAE, Corr, Acc-5, Acc-7      在**全测试集**上计算
  Acc-2 / F1  含 0  (has0)     全样本，判正规则  pred >= 0
  Acc-2 / F1  不含 0 (non0)    剔除 label == 0，判正规则  pred > 0

  Acc-5 : clip 到 [-2, 2] 后 round 比较
  Acc-7 : clip 到 [-3, 3] 后 round 比较

与各 release 原口径的差异（重算原因）
--------------------------------------
  EUAR    `compute()` 把 MAE/Corr 算在 label != 0 子集上      -> 改全集
  EBMC    MAE/Corr 算在非零子集；has0 用 pred > 0            -> 改全集 / >= 0
  DLF     双口径都算了，但输出 dict 只留 non0                -> 补 has0/Acc5/Corr
  AGS-SMoE 只算 non0 Acc2 与 Acc7，缺 MAE/Corr/has0/Acc5      -> 补齐

已一致（无需改动）：GLoMo / MLCL / MIMRL / MSE-Adapter / DEVA / MSAmba / MMRest / PSA-MF
不可比（任务不同）：CMC(3类分类) / MCR(分类Acc) / GraphSmile(7类分类)

用法
----
    from unified_metrics import eval_regression
    m = eval_regression(preds, truths)   # preds/truths: (N,) float
"""
from typing import Dict

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def _multiclass_acc(preds: np.ndarray, truths: np.ndarray) -> float:
    """round 后逐元素相等的比例（与各 release 的 multiclass_acc 一致）。"""
    return float(np.sum(np.round(preds) == np.round(truths)) / float(len(truths)))


def eval_regression(preds: np.ndarray, truths: np.ndarray) -> Dict[str, float]:
    """
    MOSI / MOSEI 标准指标（统一口径）。

    参数
    ----
    preds  : (N,) float, 模型预测的情感分
    truths : (N,) float, 真实情感分，取值域 [-3, 3]

    返回
    ----
    dict，键固定为：
        MAE, Corr,
        Acc2      (= Acc2_non0, 主口径，向后兼容),
        F1        (= F1_non0),
        Acc2_has0, F1_has0,
        Acc5, Acc7
    """
    preds = np.asarray(preds, dtype=np.float64).reshape(-1)
    truths = np.asarray(truths, dtype=np.float64).reshape(-1)
    assert preds.shape == truths.shape, f"shape mismatch: {preds.shape} vs {truths.shape}"

    # ---- 全测试集指标 ----
    mae = float(np.mean(np.abs(preds - truths)))

    # 退化防护：任一序列方差≈0 时 corr 无意义，报 0（避免 NaN）
    if len(preds) > 1 and float(np.std(preds)) > 1e-8 and float(np.std(truths)) > 1e-8:
        c = np.corrcoef(preds, truths)[0, 1]
        corr = float(c) if np.isfinite(c) else 0.0
    else:
        corr = 0.0

    acc7 = _multiclass_acc(np.clip(preds, -3.0, 3.0), np.clip(truths, -3.0, 3.0))
    acc5 = _multiclass_acc(np.clip(preds, -2.0, 2.0), np.clip(truths, -2.0, 2.0))

    # ---- Acc-2 / F1 双口径 ----
    # 不含 0：剔除 label == 0 的中性样本，pred > 0 判正
    non_zeros = np.flatnonzero(truths != 0)
    if non_zeros.size > 0:
        bp = (preds[non_zeros] > 0).astype(int)
        bt = (truths[non_zeros] > 0).astype(int)
        acc2_non0 = float(accuracy_score(bt, bp))
        f1_non0 = float(f1_score(bt, bp, average="weighted"))
    else:
        acc2_non0, f1_non0 = 0.0, 0.0

    # 含 0：全样本，pred >= 0 判正
    bp_h = (preds >= 0).astype(int)
    bt_h = (truths >= 0).astype(int)
    acc2_has0 = float(accuracy_score(bt_h, bp_h))
    f1_has0 = float(f1_score(bt_h, bp_h, average="weighted"))

    return {
        "MAE": mae,
        "Corr": corr,
        "Acc2": acc2_non0,
        "F1": f1_non0,
        "Acc2_has0": acc2_has0,
        "F1_has0": f1_has0,
        "Acc5": acc5,
        "Acc7": acc7,
    }


# 便于日志打印的固定顺序
DISPLAY_ORDER = ["MAE", "Corr", "Acc2", "F1", "Acc2_has0", "F1_has0", "Acc5", "Acc7"]
DISPLAY_LABEL = {
    "MAE": "MAE", "Corr": "Corr",
    "Acc2": "Acc2(non0)", "F1": "F1(non0)",
    "Acc2_has0": "Acc2(has0)", "F1_has0": "F1(has0)",
    "Acc5": "Acc5", "Acc7": "Acc7",
}


def format_metrics(m: Dict[str, float]) -> str:
    """统一的单行打印格式。"""
    return " | ".join(f"{DISPLAY_LABEL[k]}={m[k]:.4f}" for k in DISPLAY_ORDER)


if __name__ == "__main__":
    # 自检：构造可验证的小样本
    t = np.array([-3.0, -1.0, 0.0, 0.0, 1.0, 3.0])
    p = np.array([-3.0, -1.0, -0.5, 0.5, 1.0, 3.0])
    m = eval_regression(p, t)
    print(format_metrics(m))
    # non0: 去掉两个 0 -> [-3,-1,1,3] vs [-3,-1,1,3] 全对 -> 1.0
    assert abs(m["Acc2"] - 1.0) < 1e-9, m
    # has0: 全 6 个，-0.5 判负 vs 真值 0 判正错，0.5 判正 vs 真值 0 判正对 -> 5/6
    assert abs(m["Acc2_has0"] - 5 / 6) < 1e-9, m
    # MAE = (0+0+0.5+0.5+0+0)/6 = 1/6
    assert abs(m["MAE"] - 1 / 6) < 1e-9, m
    print("self-check OK")
