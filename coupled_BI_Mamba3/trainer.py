"""
通用 Trainer: 训练 / 验证 / 测试 单循环, 与具体模型解耦.

改进:
    - 添加 linear warmup + cosine decay 调度
    - 添加 gradient accumulation 支持
    - 添加 InfoNCE 对比损失 (模态对齐)
"""
from __future__ import annotations

import math
import os
from typing import Any, Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from losses import build_loss, MultiTaskLoss
from utils.metrics import eval_regression, eval_classification


class Trainer:
    def __init__(self, args: Any, model: torch.nn.Module, logger):
        self.args = args
        self.model = model
        self.logger = logger
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.task_type = args.task_type
        self.is_multi_task = bool(getattr(args, "multi_task", False))

        # --- 对比损失 ---
        self.contrastive_weight = float(getattr(args, "contrastive_weight", 0.0))
        self.contrastive_temp = float(getattr(args, "contrastive_temp", 0.07))

        # --- gradient accumulation ---
        self.grad_accum_steps = int(getattr(args, "grad_accum_steps", 1))

        if self.is_multi_task:
            self.criterion = MultiTaskLoss(
                task_weights=getattr(args, "task_weights", {"M": 1.0}),
                task_type=self.task_type,
            )
        else:
            self.criterion = build_loss(self.task_type)

        # ------------------------------------------------------------------
        # 优化器: BERT 用小 lr, 其他模块用大 lr
        # ------------------------------------------------------------------
        bert_lr = float(getattr(args, "bert_learning_rate", 2e-5))
        main_lr = float(args.learning_rate)
        wd = float(args.weight_decay)
        bert_params, other_params = [], []
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if "text_encoder" in n:
                bert_params.append(p)
            else:
                other_params.append(p)
        param_groups = []
        if bert_params:
            param_groups.append({"params": bert_params, "lr": bert_lr, "weight_decay": wd})
        param_groups.append({"params": other_params, "lr": main_lr, "weight_decay": wd})
        self.optimizer = torch.optim.AdamW(param_groups)
        self.logger.info(
            f"Optimizer: AdamW | bert_lr={bert_lr} ({len(bert_params)} params) | "
            f"main_lr={main_lr} ({len(other_params)} params) | wd={wd} | "
            f"grad_accum={self.grad_accum_steps}"
        )

        # --- warmup + cosine decay 调度 ---
        total_epochs = int(getattr(args, "epochs", 30))
        warmup_ratio = float(getattr(args, "warmup_ratio", 0.1))
        train_samples = int(getattr(args, "train_samples", 1000))
        batch_size = int(getattr(args, "batch_size", 32))
        steps_per_epoch = math.ceil(train_samples / batch_size)
        total_steps = steps_per_epoch * total_epochs
        self.warmup_steps = int(total_steps * warmup_ratio)
        self.total_steps = total_steps
        self.global_step = 0

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=self._lr_lambda
        )
        self.logger.info(
            f"Scheduler: warmup {self.warmup_steps} steps + cosine decay | "
            f"total_steps={total_steps}"
        )

    def _lr_lambda(self, step: int) -> float:
        """线性 warmup + cosine decay"""
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        progress = float(step - self.warmup_steps) / float(
            max(1, self.total_steps - self.warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    # ------------------------------------------------------------------
    def _to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        out = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                out[k] = v.to(self.device, non_blocking=True)
            elif isinstance(v, dict):
                out[k] = {kk: vv.to(self.device, non_blocking=True) for kk, vv in v.items()}
            else:
                out[k] = v
        return out

    def _forward_pred(self, batch: Dict[str, Any]):
        return self.model(text=batch["text"], audio=batch["audio"], video=batch["vision"])

    def _forward_with_contrastive(self, batch: Dict[str, Any]):
        """前向 + 对比损失 (共享 _encode, 避免重复计算)"""
        model = self.model
        out_l, out_a, out_v = model._encode(
            batch["text"], batch["audio"], batch["vision"]
        )
        # 池化
        pl = model._pool(out_l, model.pool_text if model.pool_type == "attention" else None)
        pa = model._pool(out_a, model.pool_audio if model.pool_type == "attention" else None)
        pv = model._pool(out_v, model.pool_video if model.pool_type == "attention" else None)
        # 分类头
        fused = model.fusion_norm(torch.cat([pl, pa, pv], dim=-1))
        logits = model.head(fused)
        # 对比损失
        cl = (_info_nce(pl, pa, self.contrastive_temp) +
              _info_nce(pl, pv, self.contrastive_temp) +
              _info_nce(pa, pv, self.contrastive_temp)) / 3.0
        return logits, cl

    # ------------------------------------------------------------------
    def train_one_epoch(self, loader: DataLoader, epoch: int) -> float:
        self.model.train()
        total_loss, n = 0.0, 0
        self.optimizer.zero_grad()
        use_cl = self.contrastive_weight > 0

        for step_i, batch in enumerate(loader):
            batch = self._to_device(batch)

            if use_cl:
                logits, cl = self._forward_with_contrastive(batch)
            else:
                logits = self._forward_pred(batch)
                cl = torch.tensor(0.0, device=self.device)

            label = batch["labels"]["M"] if not self.is_multi_task else None

            if self.is_multi_task:
                loss = self.criterion({"M": logits.squeeze(-1)}, batch["labels"])
            elif self.task_type == "regression":
                loss = self.criterion(logits.squeeze(-1), label)
            else:
                loss = self.criterion(logits, label)

            # 对比损失
            loss = loss + self.contrastive_weight * cl
            
            # === NaN 看门狗: 出现非有限值就跳过该 step ===
            if not torch.isfinite(loss):
                self.logger.warning(
                    f"[Train] Epoch {epoch} step {step_i}: non-finite loss "
                    f"(task={...}, cl={float(cl):.4f}), skipping batch"
                )
                self.optimizer.zero_grad(set_to_none=True)
                continue


            loss = loss / self.grad_accum_steps
            loss.backward()

            if (step_i + 1) % self.grad_accum_steps == 0 or (step_i + 1) == len(loader):
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1

            bs = logits.size(0)
            total_loss += float(loss.item()) * bs * self.grad_accum_steps
            n += bs

        avg = total_loss / max(n, 1)
        lrs = [f"{g['lr']:.2e}" for g in self.optimizer.param_groups]
        self.logger.info(f"[Train] Epoch {epoch} | loss={avg:.4f} | lr={lrs}")
        return avg

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, split: str = "valid") -> Dict[str, float]:
        self.model.eval()
        all_p, all_t = [], []
        for batch in loader:
            batch = self._to_device(batch)
            logits = self._forward_pred(batch)
            if self.task_type == "regression":
                all_p.append(logits.squeeze(-1).cpu().numpy())
            else:
                all_p.append(logits.cpu().numpy())
            all_t.append(batch["labels"]["M"].cpu().numpy())
        preds = np.concatenate(all_p, axis=0)
        truths = np.concatenate(all_t, axis=0)
        metrics = eval_regression(preds, truths) if self.task_type == "regression" else eval_classification(preds, truths)
        msg = " | ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        self.logger.info(f"[{split}] {msg}")
        return metrics

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({"model": self.model.state_dict(), "args": vars(self.args)}, path)
        self.logger.info(f"Checkpoint saved: {path}")

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.logger.info(f"Checkpoint loaded: {path}")


def _info_nce(z1: torch.Tensor, z2: torch.Tensor, temp: float = 0.07) -> torch.Tensor:
    """对称 InfoNCE: z1, z2 shape (B, D)"""
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    logits = z1 @ z2.t() / temp        # (B, B)
    labels = torch.arange(z1.size(0), device=z1.device)
    loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)) / 2
    return loss