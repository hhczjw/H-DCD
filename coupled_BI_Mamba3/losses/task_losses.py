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


class RegressionWithDiscreteCE(nn.Module):
    """
    复合损失: 回归 SmoothL1 + α · 离散 CE (7 类辅助分类) + β · Σ sub_loss

    设计:
        - 回归输出 (B, 1) 直接用 SmoothL1 拟合连续标签
        - 7 类辅助 logits (B, 7) 用 CE 监督 round(clip(label, -3, 3)) + 3
        - **sub_loss**: 模态级回归头 sub_T/A/V 各产 (B,1), 各与同一 label 做 SmoothL1
          loss_sub = SmoothL1(sub_T) + SmoothL1(sub_A) + SmoothL1(sub_V)
        - 总损失 = SmoothL1(main) + α · CE(aux) + β · loss_sub
        - α=0 关闭离散 CE; β=0 关闭 sub_loss; 二者均 0 退化为纯回归

    设计动机 (针对 MOSI):
        - 离散 CE: 在特征空间相邻类间产生 margin → 直接提升 Acc5/Acc7
        - sub_loss (对齐 MSAmba): 强制每个模态独立可预测 → 单模态特征更鲁棒
                                  + 充当深监督 → ISM CLS 学到判别性表征

    使用方式 (model 必须返回 dict {"logits", ?"aux_logits", ?"sub_T/A/V"}):
        loss_fn = RegressionWithDiscreteCE(alpha=0.3, sub_loss_lambda=0.3)
        loss = loss_fn(out["logits"], out.get("aux_logits"), labels,
                       sub_outputs=(out.get("sub_T"), out.get("sub_A"), out.get("sub_V")))
    """

    def __init__(
        self,
        alpha: float = 0.3,
        num_aux_classes: int = 7,
        clip_range: float = 3.0,
        label_smoothing: float = 0.05,
        regression_beta: float = 0.5,
        sub_loss_lambda: float = 0.0,
    ):
        super().__init__()
        assert num_aux_classes == 2 * int(clip_range) + 1, (
            f"num_aux_classes ({num_aux_classes}) 应等于 2*clip_range+1 "
            f"({2 * int(clip_range) + 1}); MOSI 默认 clip_range=3 → 7 类"
        )
        self.alpha = float(alpha)
        self.sub_loss_lambda = float(sub_loss_lambda)
        self.num_aux_classes = int(num_aux_classes)
        self.clip_range = float(clip_range)
        self.reg_loss = nn.SmoothL1Loss(beta=regression_beta)
        self.cls_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def _to_class_idx(self, labels: torch.Tensor) -> torch.Tensor:
        """连续标签 → 离散类别索引 [0, num_aux_classes)"""
        clipped = torch.clamp(labels, -self.clip_range, self.clip_range)
        # round 到最近整数, 然后 shift 到 [0, num_aux_classes)
        idx = torch.round(clipped).long() + int(self.clip_range)
        return idx

    def forward(
        self,
        reg_logits: torch.Tensor,   # (B,) or (B, 1)
        aux_logits: torch.Tensor,   # (B, num_aux_classes) or None
        labels: torch.Tensor,       # (B,) float
        sub_outputs=None,           # tuple/list of (sub_T, sub_A, sub_V) each (B,1) or None
    ) -> torch.Tensor:
        if reg_logits.ndim > 1 and reg_logits.size(-1) == 1:
            reg_logits = reg_logits.squeeze(-1)
        loss = self.reg_loss(reg_logits, labels)
        if self.alpha > 0.0 and aux_logits is not None:
            cls_idx = self._to_class_idx(labels)
            loss = loss + self.alpha * self.cls_loss(aux_logits, cls_idx)
        if self.sub_loss_lambda > 0.0 and sub_outputs is not None:
            loss_sub = 0.0
            n_sub = 0
            for s in sub_outputs:
                if s is None:
                    continue
                if s.ndim > 1 and s.size(-1) == 1:
                    s = s.squeeze(-1)
                loss_sub = loss_sub + self.reg_loss(s, labels)
                n_sub += 1
            if n_sub > 0:
                loss = loss + self.sub_loss_lambda * loss_sub
        return loss


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