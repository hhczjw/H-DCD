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

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(args.learning_rate),
            weight_decay=float(args.weight_decay),
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
        # text 可能是 (B, 3, L)  BERT 三通道, 这里取 input_ids 通道 0 替换为零向量;
        # 简单起见, 若为 BERT 三通道, 用 channel-0 + 嵌入 (None) -> 这里假设上游已嵌入.
        text = batch["text"]
        if text.ndim == 3 and text.size(1) == 3:
            # (B, 3, L) -> 取 input_ids, 后续模型应自行做 embedding;
            # 这里给个保底: 转成 float 当作 one-hot-like (仅占位, 实际工程应在模型内嵌 BERT).
            text = text[:, 0].unsqueeze(-1).float()  # (B, L, 1)
        return self.model(text=text, audio=batch["audio"], video=batch["vision"])

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
                # 简化版: 主任务使用同一 logits, 子任务复用 (可后续扩展为多头)
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
        self.logger.info(f"[Train] Epoch {epoch} | loss={avg:.4f}")
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