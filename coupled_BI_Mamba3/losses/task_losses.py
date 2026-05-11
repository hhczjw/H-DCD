"""
任务损失:
    - 回归: SmoothL1Loss (比 L1 更鲁棒, 对异常值不敏感)
    - 分类: CrossEntropyLoss (IEMOCAP/MELD)
    - 多任务: 加权和 (SIMS T/A/V)
    - InfoNCE: 对比损失 (模态对齐, 在 trainer.py 中调用)

改进:
    - L1Loss → SmoothL1Loss (beta=0.5), 减少异常值对梯度的影响
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_loss(task_type: str) -> nn.Module:
    if task_type == "regression":
        return nn.SmoothL1Loss(beta=0.5)
    elif task_type == "classification":
        return nn.CrossEntropyLoss(label_smoothing=0.1)
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


class InfoNCELoss(nn.Module):
    """
    InfoNCE 对比损失: 让同一样本的不同模态表征相互靠近.
    输入: z1, z2 shape (B, D), 对称计算.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        logits = z1 @ z2.t() / self.temperature
        labels = torch.arange(z1.size(0), device=z1.device)
        loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)) / 2
        return loss