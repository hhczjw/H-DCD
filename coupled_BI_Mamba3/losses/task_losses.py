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
        self.cls_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing, reduction='none')

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
            
            # --- Distance-Aware Ordinal Penalty ---
            ce_loss_raw = self.cls_loss(aux_logits, cls_idx)  # (B,) since reduction='none'
            with torch.no_grad():
                pred_idx = aux_logits.argmax(dim=-1)
                dist = torch.abs(pred_idx - cls_idx).float()
                # 【修改】将二次方惩罚改为柔和的线性惩罚，防止梯度爆炸吃掉ACC2
                dist_weight = 1.0 + dist * 0.1
                
            ce_loss_weighted = (ce_loss_raw * dist_weight).mean()
            loss = loss + self.alpha * ce_loss_weighted
            
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
                # 问题 5 修复: 按分支数归一, 避免实际系数被 ×n_sub 放大
                loss = loss + self.sub_loss_lambda * (loss_sub / n_sub)
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


class SupervisedContrastiveLoss(nn.Module):
    """标签感知对比损失: 标签相近的样本互拉, 标签远离的样本互推.

    与无监督 InfoNCE 不同, 正/负样本对由标签距离动态决定.
    标签距离越大, margin 越大, 推离力度越强.
    """

    def __init__(self, temperature: float = 0.07, label_margin_scale: float = 0.15):
        super().__init__()
        self.temperature = temperature
        self.margin_scale = label_margin_scale

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        features = F.normalize(features, dim=-1)
        sim = features @ features.t() / self.temperature  # (B, B)

        label_dist = (labels.unsqueeze(0) - labels.unsqueeze(1)).abs()  # (B, B)
        # 动态 margin: 标签越远, 推离力度越大
        margin = self.margin_scale * label_dist
        logits = sim - margin / self.temperature

        # 排除自身对角线
        mask = ~torch.eye(features.size(0), dtype=torch.bool, device=features.device)
        logits = logits.masked_fill(~mask, -1e9)

        # 软目标分布: 标签越近, 权重越大
        target_dist = torch.exp(-label_dist * 2.0)
        target_dist = target_dist.masked_fill(~mask, 0.0)
        target_dist = target_dist / target_dist.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        log_probs = F.log_softmax(logits, dim=-1)
        loss = -(target_dist * log_probs).sum(dim=-1).mean()
        return loss


# ============================================================================
# 方案 A: Acc7/Acc2 专项损失函数改进
# ============================================================================

class OrdinalRegressionLoss(nn.Module):
    """
    序数回归损失: 直接优化 7 类离散分类精度

    原理 (Acc7 评估: round(pred) == round(label)):
        对于 K 类序数分类, 学习 K-1 个可学习阈值 θ_k
        P(Y ≤ k) = σ(θ_k - f(x)),  其中 f(x) 是模型连续输出
        P(Y = k) = P(Y ≤ k) - P(Y ≤ k-1)

    优势:
        - 直接建模离散边界, 与 Acc7 评估方式一致
        - 保留序数关系 (类2比类1更接近类3)
        - 预测时: pred_class = argmax P(Y=k), 或直接 round(f(x))

    来源: 多篇情感分析论文中使用序数回归提升 Acc7

    Args:
        num_classes: 类别数 (MOSI 默认 7)
        clip_range: 标签裁剪范围 (MOSI 默认 3.0, 即 [-3, +3])
    """

    def __init__(self, num_classes: int = 7, clip_range: float = 3.0):
        super().__init__()
        self.num_classes = num_classes
        self.clip_range = clip_range
        # 可学习阈值: 初始化为均匀分布 [-3, -2, -1, 0, 1, 2] (6个)
        init_thresholds = torch.linspace(-clip_range, clip_range, num_classes - 1)
        self.thresholds = nn.Parameter(init_thresholds)

    def _to_class_idx(self, labels: torch.Tensor) -> torch.Tensor:
        """连续标签 → 离散类别索引 [0, num_classes)"""
        clipped = torch.clamp(labels, -self.clip_range, self.clip_range)
        idx = torch.round(clipped).long() + int(self.clip_range)
        return torch.clamp(idx, 0, self.num_classes - 1)

    def forward(self, pred: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (B,) 连续预测值
            labels: (B,) 连续标签
        Returns:
            loss: 标量
        """
        if pred.ndim > 1 and pred.size(-1) == 1:
            pred = pred.squeeze(-1)

        cls_idx = self._to_class_idx(labels)

        # 排序阈值 (确保 θ_0 < θ_1 < ... < θ_5)
        sorted_thresh = torch.cumsum(
            F.softplus(self.thresholds.to(pred.device)), dim=0
        ) - self.clip_range

        # 累积概率: P(Y ≤ k) = σ(θ_k - pred)
        # pred: (B,), sorted_thresh: (K-1,) → (B, K-1)
        cum_probs = torch.sigmoid(
            sorted_thresh.unsqueeze(0) - pred.unsqueeze(1)
        )  # (B, K-1)

        # 类别概率: P(Y=k) = P(Y≤k) - P(Y≤k-1)
        probs = torch.zeros(
            pred.size(0), self.num_classes, device=pred.device, dtype=pred.dtype
        )
        probs[:, 0] = cum_probs[:, 0]                        # P(Y=0) = P(Y≤0)
        probs[:, 1:-1] = cum_probs[:, 1:] - cum_probs[:, :-1]  # P(Y=k)
        probs[:, -1] = 1.0 - cum_probs[:, -1]                 # P(Y=K-1) = 1-P(Y≤K-2)

        # 数值稳定: clamp 防止 log(0)
        probs = probs.clamp(min=1e-8)

        # 交叉熵: -log P(Y=cls_idx)
        log_probs = torch.log(probs)
        loss = F.nll_loss(log_probs, cls_idx)

        return loss


class BoundaryAwareL1(nn.Module):
    """
    边界感知 L1 损失: 对 round 边界附近的样本加大惩罚

    原理:
        SmoothL1 优化连续误差, 但对 round 边界无感知。
        例如: pred=0.4, label=0.6 → MAE=0.2 (很好)
              但 round(0.4)=0, round(0.6)=1 → Acc7 错误

        BoundaryAwareL1 在 pred 接近 round 边界时加大惩罚,
        推动 pred 远离边界, 减少边界附近的分类错误。

    Args:
        beta: SmoothL1 的 beta 参数
        boundary_margin: 边界判定范围 (默认 0.4)
        boundary_weight: 边界附近的额外权重 (默认 3.0)
    """

    def __init__(self, beta: float = 0.5, boundary_margin: float = 0.4,
                 boundary_weight: float = 3.0):
        super().__init__()
        self.smooth_l1 = nn.SmoothL1Loss(beta=beta, reduction='none')
        self.margin = boundary_margin
        self.weight = boundary_weight

    def forward(self, pred: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        if pred.ndim > 1 and pred.size(-1) == 1:
            pred = pred.squeeze(-1)
        base_loss = self.smooth_l1(pred, label)
        # 检查 pred 是否在 round 边界附近
        dist_to_boundary = torch.abs(pred - torch.round(pred))
        is_near_boundary = (dist_to_boundary < self.margin).float()
        # 边界附近加权
        weight = 1.0 + (self.weight - 1.0) * is_near_boundary
        return (base_loss * weight).mean()


class OrdinalCompositeLoss(nn.Module):
    """
    方案 A 组合损失: 序数回归 + 边界感知 L1 + 子任务 L1

    总损失 = BoundaryAwareL1(main) + α · OrdinalLoss(main) + β · Σ L1(sub)

    与 RegressionWithDiscreteCE 的区别:
        - 用 OrdinalLoss 替代 CE (直接建模离散边界)
        - 用 BoundaryAwareL1 替代 SmoothL1 (边界感知)
        - 保留 sub_loss 深监督

    使用方式:
        loss_fn = OrdinalCompositeLoss(alpha=0.5, sub_loss_lambda=0.3)
        loss = loss_fn(pred, labels, sub_outputs=(sub_T, sub_A, sub_V))
    """

    def __init__(
        self,
        alpha: float = 0.5,
        num_classes: int = 7,
        clip_range: float = 3.0,
        beta: float = 0.5,
        boundary_margin: float = 0.4,
        boundary_weight: float = 3.0,
        sub_loss_lambda: float = 0.0,
    ):
        super().__init__()
        self.alpha = float(alpha)
        self.sub_loss_lambda = float(sub_loss_lambda)
        self.reg_loss = BoundaryAwareL1(
            beta=beta, boundary_margin=boundary_margin,
            boundary_weight=boundary_weight,
        )
        self.ordinal_loss = OrdinalRegressionLoss(
            num_classes=num_classes, clip_range=clip_range,
        )

    def forward(
        self,
        pred: torch.Tensor,
        labels: torch.Tensor,
        sub_outputs=None,
    ) -> torch.Tensor:
        if pred.ndim > 1 and pred.size(-1) == 1:
            pred = pred.squeeze(-1)

        # 主损失: 边界感知 L1
        loss = self.reg_loss(pred, labels)

        # 序数回归损失
        if self.alpha > 0.0:
            loss = loss + self.alpha * self.ordinal_loss(pred, labels)

        # 子任务损失
        if self.sub_loss_lambda > 0.0 and sub_outputs is not None:
            loss_sub = 0.0
            n_sub = 0
            for s in sub_outputs:
                if s is None:
                    continue
                if s.ndim > 1 and s.size(-1) == 1:
                    s = s.squeeze(-1)
                loss_sub = loss_sub + F.smooth_l1_loss(s, labels, beta=0.5)
                n_sub += 1
            if n_sub > 0:
                # 按分支数归一, 避免实际系数被 ×n_sub 放大
                loss = loss + self.sub_loss_lambda * (loss_sub / n_sub)

        return loss