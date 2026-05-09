"""
任务损失:
    - 回归: L1Loss (MOSI/MOSEI/SIMS 主流做法, 与 KeyEval='Loss' 配套)
    - 分类: CrossEntropyLoss (IEMOCAP/MELD)
    - 多任务: 加权和 (SIMS T/A/V)
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


def build_loss(task_type: str) -> nn.Module:
    if task_type == "regression":
        return nn.L1Loss()
    elif task_type == "classification":
        return nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unknown task_type: {task_type}")


class MultiTaskLoss(nn.Module):
    """
    SIMS 多任务损失: M (主) + T/A/V (辅).
    模型需为每个 tag 各产出 logits/score, 这里假设模型 forward 返回 dict:
        {"M": (B, num_classes), "T": (B,1), "A": (B,1), "V": (B,1)}
    或由调用方拆分。
    """

    def __init__(self, task_weights: Dict[str, float], task_type: str = "regression"):
        super().__init__()
        self.task_weights = task_weights
        self.base_loss = build_loss(task_type)

    def forward(
        self,
        preds: Dict[str, torch.Tensor],
        labels: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        total = 0.0
        for tag, w in self.task_weights.items():
            if tag in preds and tag in labels:
                p = preds[tag].squeeze(-1) if preds[tag].ndim > 1 and preds[tag].size(-1) == 1 else preds[tag]
                total = total + w * self.base_loss(p, labels[tag])
        return total