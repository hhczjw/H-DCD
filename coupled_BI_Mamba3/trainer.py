"""
通用 Trainer: 训练 / 验证 / 测试 单循环, 与具体模型解耦.
"""
from __future__ import annotations

import os
from typing import Any, Dict

import numpy as np
import torch
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

        if self.is_multi_task:
            self.criterion = MultiTaskLoss(
                task_weights=getattr(args, "task_weights", {"M": 1.0}),
                task_type=self.task_type,
            )
        else:
            self.criterion = build_loss(self.task_type)

        # ------------------------------------------------------------------
        # 优化器: BERT 用小 lr (~1e-5), 其他模块用大 lr (~1e-3 ~ 1e-4)
        # ------------------------------------------------------------------
        bert_lr = float(getattr(args, "bert_learning_rate", 1e-5))
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
            f"main_lr={main_lr} ({len(other_params)} params) | wd={wd}"
        )

        # cosine 调度 (按 epoch 衰减)
        total_epochs = int(getattr(args, "epochs", 30))
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=total_epochs, eta_min=1e-7
        )

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

    def _forward_pred(self, batch: Dict[str, Any]) -> torch.Tensor:
        # text 形如:
        #   - use_bert=True : (B, 3, L)  input_ids / mask / segment, 由模型内 BertTextEncoder 处理
        #   - use_bert=False: (B, L, D)  词向量
        # 无需在这里做任何降维 / 占位处理.
        return self.model(text=batch["text"], audio=batch["audio"], video=batch["vision"])

    # ------------------------------------------------------------------
    def train_one_epoch(self, loader: DataLoader, epoch: int) -> float:
        self.model.train()
        total_loss, n = 0.0, 0
        for batch in loader:
            batch = self._to_device(batch)
            self.optimizer.zero_grad()
            logits = self._forward_pred(batch)
            label = batch["labels"]["M"] if not self.is_multi_task else None

            if self.is_multi_task:
                loss = self.criterion({"M": logits.squeeze(-1)}, batch["labels"])
            elif self.task_type == "regression":
                loss = self.criterion(logits.squeeze(-1), label)
            else:
                loss = self.criterion(logits, label)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            bs = logits.size(0)
            total_loss += float(loss.item()) * bs
            n += bs
        avg = total_loss / max(n, 1)
        self.scheduler.step()
        lrs = [g["lr"] for g in self.optimizer.param_groups]
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