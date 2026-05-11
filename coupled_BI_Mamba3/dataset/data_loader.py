"""
标准多模态情感识别 DataLoader
=================================

支持数据集:
    - MOSI / MOSEI         (回归: regression_labels)
    - SIMS / SIMS2         (多任务: regression_labels + T/A/V 子任务标签)
    - IEMOCAP / MELD       (分类: classification_labels)

参考实现:
    H-DCD/dataset/data_loader.py (作者原版)
    并扩展为更标准的 MMSA 风格接口。

改进:
    - 添加训练阶段的 audio/vision 数据增强 (特征级别)
      * 随机时间 mask: 随机遮蔽连续时间步 → 0
      * 随机特征 dropout: 按概率随机置零部分特征维度
      * 高斯噪声: 添加微小噪声提升鲁棒性

约定:
    输入 .pkl 数据布局:
        {
            "train": {
                "text": np.ndarray,            # (N, L_t, D_t) 或 (N, D_t, L_t)
                "text_bert": np.ndarray,       # (N, 3, L_t)  可选 (BERT 三通道: input_ids/mask/segment)
                "audio": np.ndarray,           # (N, L_a, D_a)
                "vision": np.ndarray,          # (N, L_v, D_v)
                "regression_labels": np.ndarray,         # (N,)  MOSI/MOSEI/SIMS
                "classification_labels": np.ndarray,     # (N,)  IEMOCAP/MELD
                "regression_labels_T/A/V": np.ndarray,   # (N,)  SIMS 多任务
                "audio_lengths": np.ndarray,             # 非对齐 audio 真实长度 (可选)
                "vision_lengths": np.ndarray,            # 非对齐 vision 真实长度 (可选)
                "id": List[str]                          # 样本 id
            },
            "valid": {...}, "test": {...}
        }

返回 (DataLoader 单 batch dict):
    {
        "text":    Tensor (B, L_t, D_t)  或 (B, 3, L_t) 当 use_bert=True
        "audio":   Tensor (B, L_a, D_a)
        "vision":  Tensor (B, L_v, D_v)
        "audio_lengths":  LongTensor (B,)   (非对齐时)
        "vision_lengths": LongTensor (B,)   (非对齐时)
        "labels":  {
            "M":   Tensor (B,),   # 主任务标签
            "T":   Tensor (B,),   # SIMS 多任务可选
            "A":   Tensor (B,),
            "V":   Tensor (B,),
        },
        "ids":     List[str],
        "index":   LongTensor (B,)
    }
"""

from __future__ import annotations

import logging
import os
import pickle
from typing import Any, Dict, List, Optional

import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger("MSA")

# 主任务键: 哪些数据集走回归, 哪些走分类
REGRESSION_DATASETS = {"mosi", "mosei", "sims", "sims2"}
CLASSIFICATION_DATASETS = {"iemocap", "meld"}
MULTI_TASK_DATASETS = {"sims", "sims2"}     # 同时拥有 T/A/V 子任务标签

__all__ = ["MMDataset", "MMDataLoader"]


class MMDataset(Dataset):
    """
    通用多模态数据集.
    """

    def __init__(self, args: Any, mode: str = "train"):
        self.args = args
        self.mode = mode  # train / valid / test
        self.dataset_name = str(getattr(args, "dataset_name", "mosi")).lower()
        if self.dataset_name not in REGRESSION_DATASETS | CLASSIFICATION_DATASETS:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")

        self.use_bert = bool(getattr(args, "use_bert", True))
        self.need_truncated = bool(getattr(args, "need_data_aligned", False)) is False  # 非对齐时才截断/补齐
        self.need_normalize = bool(getattr(args, "need_normalized", False))

        # ---- 数据增强配置 (仅训练集) ----
        self.augment_audio = bool(getattr(args, "augment_audio", False)) and mode == "train"
        self.augment_vision = bool(getattr(args, "augment_vision", False)) and mode == "train"
        self.augment_prob = float(getattr(args, "augment_prob", 0.3))

        # ---- 1) 加载 pickle ----
        data_path = args.featurePath
        if not os.path.isfile(data_path):
            raise FileNotFoundError(f"Feature file not found: {data_path}")
        logger.info(f"[{mode}] Loading dataset: {data_path}")
        with open(data_path, "rb") as f:
            data = pickle.load(f)

        if mode not in data:
            raise KeyError(f"Mode '{mode}' not in pkl. Available: {list(data.keys())}")
        split = data[mode]

        # ---- 2) 提取三模态 ----
        # 文本
        if self.use_bert and "text_bert" in split:
            self.text = np.asarray(split["text_bert"], dtype=np.float32)   # (N, 3, L)
        else:
            self.text = np.asarray(split["text"], dtype=np.float32)
            self.text = self._maybe_transpose(self.text)                    # -> (N, L, D)

        self.audio = np.asarray(split["audio"], dtype=np.float32)
        self.audio = self._maybe_transpose(self.audio)
        self.vision = np.asarray(split["vision"], dtype=np.float32)
        self.vision = self._maybe_transpose(self.vision)

        # NaN 防御 (作者原代码做法: audio NaN -> 0)
        np.nan_to_num(self.audio, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        np.nan_to_num(self.vision, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        # ---- 3) 标签 ----
        self.labels: Dict[str, np.ndarray] = {}
        if self.dataset_name in REGRESSION_DATASETS:
            if "regression_labels" not in split:
                raise KeyError(f"'regression_labels' missing for {self.dataset_name}")
            self.labels["M"] = np.asarray(split["regression_labels"], dtype=np.float32)
            if self.dataset_name in MULTI_TASK_DATASETS:
                for tag in ("T", "A", "V"):
                    key = f"regression_labels_{tag}"
                    if key in split:
                        self.labels[tag] = np.asarray(split[key], dtype=np.float32)
        else:
            # 分类
            label_key = "classification_labels" if "classification_labels" in split else "labels"
            if label_key not in split:
                raise KeyError(f"classification labels missing for {self.dataset_name}")
            self.labels["M"] = np.asarray(split[label_key], dtype=np.int64)

        # ---- 4) 非对齐序列长度 (可选) ----
        self.audio_lengths = (
            np.asarray(split["audio_lengths"], dtype=np.int64)
            if "audio_lengths" in split else None
        )
        self.vision_lengths = (
            np.asarray(split["vision_lengths"], dtype=np.int64)
            if "vision_lengths" in split else None
        )

        # ---- 5) sample id ----
        self.ids = list(split.get("id", [f"{mode}_{i}" for i in range(len(self.labels['M']))]))

        # ---- 6) 外部特征替换 (动态 ablation) ----
        self._maybe_replace_external_features()

        # ---- 7) 截断 + 归一化 ----
        if self.need_truncated:
            self._truncate(getattr(args, "seq_lens", None))
        if self.need_normalize:
            self._normalize()

        # ---- 8) 形状校验 ----
        self.n_samples = len(self.labels["M"])
        assert self.audio.shape[0] == self.n_samples, "audio N mismatch"
        assert self.vision.shape[0] == self.n_samples, "vision N mismatch"
        if self.use_bert:
            assert self.text.shape[0] == self.n_samples, "text_bert N mismatch"
        else:
            assert self.text.shape[0] == self.n_samples, "text N mismatch"

        logger.info(
            f"[{mode}] dataset={self.dataset_name}, N={self.n_samples}, "
            f"text={tuple(self.text.shape)}, audio={tuple(self.audio.shape)}, "
            f"vision={tuple(self.vision.shape)}"
        )

    # ---------- private utils ----------
    @staticmethod
    def _maybe_transpose(x: np.ndarray) -> np.ndarray:
        """
        统一为 (N, L, D). 如果检测到 (N, D, L) (即 D < L, 且与作者原代码 transpose(1,2) 一致),
        则不动手; 这里采用 H-DCD 的策略: 当 axis=1 大于 axis=2 时保持, 否则不变.
        实际场景中: BERT 分词后 L>>D, GLoVe 词向量 L<D.

        简洁策略: 强制 (N, L, D) — 如果 shape[1] < shape[2] 视为 (N, D, L), 转置.
        """
        if x.ndim != 3:
            return x
        # 经验: 通常 D < L, 若 D > L 不转
        if x.shape[1] < x.shape[2]:
            return np.transpose(x, (0, 2, 1))
        return x

    def _maybe_replace_external_features(self):
        """支持 ablation 时把某模态特征替换为外部 .npy。"""
        for tag, attr in (("T", "text"), ("A", "audio"), ("V", "vision")):
            key = f"feature_{tag}"
            path = getattr(self.args, key, "")
            if path and os.path.isfile(path):
                logger.info(f"[{self.mode}] Replace {attr} feature with: {path}")
                arr = np.load(path, allow_pickle=True)
                if isinstance(arr, np.ndarray) and arr.dtype == object:
                    # 可能是 dict 包装
                    arr = arr.item().get(self.mode, arr)
                arr = self._maybe_transpose(np.asarray(arr, dtype=np.float32))
                setattr(self, attr, arr)

    def _truncate(self, seq_lens: Optional[List[int]]):
        """
        非对齐序列下按 seq_lens=[L_t, L_a, L_v] 截断 / pad.
        seq_lens 长度应为 3, 与 (text, audio, vision) 对应.
        """
        if not seq_lens or len(seq_lens) != 3:
            return
        L_t, L_a, L_v = seq_lens

        def _fit(x: np.ndarray, L: int) -> np.ndarray:
            # x: (N, cur_L, D)
            cur = x.shape[1]
            if cur == L:
                return x
            if cur > L:
                return x[:, :L]
            pad = np.zeros((x.shape[0], L - cur, x.shape[2]), dtype=x.dtype)
            return np.concatenate([x, pad], axis=1)

        if not self.use_bert:
            self.text = _fit(self.text, L_t)
        # text_bert 形如 (N, 3, L_t), 单独处理
        elif self.use_bert and self.text.ndim == 3:
            cur = self.text.shape[2]
            if cur > L_t:
                self.text = self.text[:, :, :L_t]
            elif cur < L_t:
                pad = np.zeros((self.text.shape[0], 3, L_t - cur), dtype=self.text.dtype)
                self.text = np.concatenate([self.text, pad], axis=2)

        self.audio = _fit(self.audio, L_a)
        self.vision = _fit(self.vision, L_v)

    def _normalize(self):
        """按特征维度做均值 / 方差归一化 (audio + vision)."""
        eps = 1e-6
        for attr in ("audio", "vision"):
            x = getattr(self, attr)
            mean = x.mean(axis=(0, 1), keepdims=True)
            std = x.std(axis=(0, 1), keepdims=True)
            setattr(self, attr, (x - mean) / (std + eps))

    # ---------- Dataset API ----------
    def __len__(self) -> int:
        return self.n_samples

    def _augment_feature(self, feat: np.ndarray) -> np.ndarray:
        """
        对单个样本的特征 (L, D) 做数据增强:
            1. 随机时间 mask: 连续 mask 一段时间步
            2. 随机特征 dropout: 某些特征维度置零
            3. 高斯噪声: 添加微小扰动
        """
        L, D = feat.shape
        p = self.augment_prob

        # 1) 时间 mask: 随机遮蔽连续 10-20% 的时间步
        if random.random() < p:
            mask_len = max(1, int(L * random.uniform(0.1, 0.2)))
            start = random.randint(0, max(0, L - mask_len))
            feat[start:start + mask_len, :] = 0.0

        # 2) 特征维度 dropout: 随机 10-20% 的维度置零
        if random.random() < p:
            n_drop = max(1, int(D * random.uniform(0.1, 0.2)))
            drop_dims = random.sample(range(D), n_drop)
            feat[:, drop_dims] = 0.0

        # 3) 高斯噪声
        if random.random() < p:
            noise_std = feat.std() * 0.05  # 5% 标准差的噪声
            noise = np.random.randn(*feat.shape).astype(feat.dtype) * noise_std
            feat = feat + noise

        return feat

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        audio_feat = self.audio[idx].copy()
        vision_feat = self.vision[idx].copy()

        # --- 数据增强 (仅训练集) ---
        if self.augment_audio:
            audio_feat = self._augment_feature(audio_feat)
        if self.augment_vision:
            vision_feat = self._augment_feature(vision_feat)

        sample = {
            "text": torch.from_numpy(self.text[idx]),
            "audio": torch.from_numpy(audio_feat),
            "vision": torch.from_numpy(vision_feat),
            "id": self.ids[idx],
            "index": torch.tensor(idx, dtype=torch.long),
        }
        # 序列长度 (非对齐时)
        if self.audio_lengths is not None:
            sample["audio_lengths"] = torch.tensor(self.audio_lengths[idx], dtype=torch.long)
        if self.vision_lengths is not None:
            sample["vision_lengths"] = torch.tensor(self.vision_lengths[idx], dtype=torch.long)

        # 多任务标签字典
        labels = {}
        for tag, arr in self.labels.items():
            if self.dataset_name in REGRESSION_DATASETS:
                labels[tag] = torch.tensor(arr[idx], dtype=torch.float32)
            else:
                labels[tag] = torch.tensor(arr[idx], dtype=torch.long)
        sample["labels"] = labels
        # 兼容字段: 主任务标签直接放外层
        sample["label"] = labels["M"]
        return sample


# =====================================================================
# 工厂函数
# =====================================================================

def _collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """通用 collate, 自动堆叠 tensor, 字符串 / 标量原样保留为 list."""
    out: Dict[str, Any] = {}
    keys = batch[0].keys()
    for k in keys:
        vals = [b[k] for b in batch]
        if k == "labels":
            sub_keys = vals[0].keys()
            out["labels"] = {sk: torch.stack([v[sk] for v in vals], dim=0) for sk in sub_keys}
        elif k == "id":
            out["ids"] = vals
        elif isinstance(vals[0], torch.Tensor):
            out[k] = torch.stack(vals, dim=0)
        else:
            out[k] = vals
    return out


def MMDataLoader(args: Any, num_workers: int = 0) -> Dict[str, DataLoader]:
    """
    构建 train / valid / test 三个 DataLoader.

    args 需提供:
        dataset_name:      str
        featurePath:       str   (.pkl 路径)
        batch_size:        int
        seq_lens:          [L_t, L_a, L_v]      (非对齐时)
        use_bert:          bool
        need_data_aligned: bool
        need_normalized:   bool
        feature_T/A/V:     str  外部特征覆盖 (可空)
    """
    datasets = {
        "train": MMDataset(args, mode="train"),
        "valid": MMDataset(args, mode="valid"),
        "test":  MMDataset(args, mode="test"),
    }

    batch_size = int(getattr(args, "batch_size", 32))
    pin_memory = bool(getattr(args, "pin_memory", True))

    loaders = {
        split: DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
            collate_fn=_collate_fn,
        )
        for split, ds in datasets.items()
    }
    return loaders