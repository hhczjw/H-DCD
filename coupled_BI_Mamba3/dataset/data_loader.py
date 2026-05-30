"""
标准多模态情感识别 DataLoader
=================================

支持数据集:
    - MOSI / MOSEI         (回归: regression_labels)
    - SIMS / SIMS2         (多任务: regression_labels + T/A/V 子任务标签)
    - IEMOCAP / MELD       (分类: classification_labels)

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
                "id": List[str]                          # 样本 id
            },
            "valid": {...}, "test": {...}
        }

返回 (DataLoader 单 batch dict):
    {
        "text":    Tensor (B, L_t, D_t)  或 (B, 3, L_t) 当 use_bert=True
        "audio":   Tensor (B, L_a, D_a)
        "vision":  Tensor (B, L_v, D_v)
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

        # ---- ★ CAGMamba 对齐: WAV 在线加载模式 ----
        self.use_wav_audio = bool(getattr(args, 'use_wav_audio', False))
        self.wav_dir = getattr(args, 'wav_dir', '')
        self._feature_extractor = None
        if self.use_wav_audio:
            import pandas as pd
            csv_path = getattr(args, 'audio_csv_path', '')
            if not csv_path or not os.path.isfile(csv_path):
                raise FileNotFoundError(f"WAV mode requires audio_csv_path: {csv_path}")
            df = pd.read_csv(csv_path)
            df = df[df['mode'] == mode]
            # 构建 sample ID → WAV 路径的映射
            self._wav_paths = {}
            for _, row in df.iterrows():
                vid, cid = str(row['video_id']), str(int(row['clip_id']))
                sid = f"{vid}_{cid}"  # CSV 格式
                wav_path = os.path.join(self.wav_dir, vid, f"{cid}.wav")
                if os.path.isfile(wav_path):
                    self._wav_paths[sid] = wav_path
            # Wav2Vec2FeatureExtractor (对齐 CAGMamba)
            from transformers import Wav2Vec2FeatureExtractor
            self._feature_extractor = Wav2Vec2FeatureExtractor(
                feature_size=1, sampling_rate=16000, padding_value=0.0,
                do_normalize=True, return_attention_mask=True,
            )
            logger.info(f"[{mode}] WAV mode enabled: {len(self._wav_paths)} paths")

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

        # ---- 4) sample id ----
        self.ids = list(split.get("id", [f"{mode}_{i}" for i in range(len(self.labels['M']))]))

        # ---- ★ Phase 5: 对话上下文 ----
        self.use_context = bool(getattr(self.args, 'use_context', False))

        # ---- ★ Phase 3: 动态替换特征 (外部 .pkl 预提取特征) ----
        # ★ 必须在 _init_context 之前运行, 确保 context 特征也用新维度
        def _load_external_feat(attr_name, feat_key):
            """从外部 .pkl 加载特征, 按样本 ID 对齐后替换对应属性."""
            path = getattr(self.args, attr_name, "")
            if not path or not os.path.isfile(path):
                return
            logger.info(f"[{mode}] Loading external {feat_key} features: {path}")
            with open(path, "rb") as f:
                ext_data = pickle.load(f)
            if mode not in ext_data or feat_key not in ext_data[mode]:
                return
            ext_features = np.asarray(ext_data[mode][feat_key], dtype=np.float32)

            # ★ 关键: 按 ID 对齐, 不按索引
            # ext_data 中的 id 格式为 "video_id_clip_id" (CSV 风格)
            # self.ids (来自 .pkl) 格式为 "video_id$_$clip_id"
            # 统一 ID 格式后按索引映射
            ext_ids = list(ext_data[mode].get(f"{feat_key}_ids", []))
            if not ext_ids:
                # 如果没有存 id, 尝试从原始 label.csv 重建
                logger.warning(
                    f"[{mode}] External {feat_key} .pkl 缺少 id 字段, "
                    f"无法对齐, 跳过替换!"
                )
                return

            # 统一 ID 分隔符: "$_$" → "_", "_" → "$_$" 都映射到统一格式
            self_ids = [str(i).replace('$_$', '_') for i in self.ids]
            ext_ids  = [str(i).replace('$_$', '_') for i in ext_ids]

            # 建立 ext ID → ext index 的映射
            ext_id_to_idx = {eid: i for i, eid in enumerate(ext_ids)}

            # 为每个 .pkl 样本找到对应的 ext 特征
            aligned = np.zeros_like(ext_features[:len(self_ids)])
            hit = 0
            for i, sid in enumerate(self_ids):
                if sid in ext_id_to_idx:
                    aligned[i] = ext_features[ext_id_to_idx[sid]]
                    hit += 1
            logger.info(
                f"[{mode}] External {feat_key}: aligned {hit}/{len(self_ids)} samples"
            )
            if hit == 0:
                logger.warning(
                    f"[{mode}] External {feat_key}: ZERO alignment! Order mismatch?"
                )
                return

            setattr(self, feat_key, aligned)
            dim_map = {"text": 0, "audio": 1, "vision": 2}
            if feat_key in dim_map:
                idx = dim_map[feat_key]
                if hasattr(self.args, 'feature_dims') and len(self.args.feature_dims) > idx:
                    self.args.feature_dims[idx] = int(aligned.shape[-1])
            logger.info(f"[{mode}] Replaced self.{feat_key} → shape {aligned.shape}")

        _load_external_feat("feature_T", "text")
        _load_external_feat("feature_A", "audio")
        _load_external_feat("feature_V", "vision")

        # ---- 5) 截断 + 归一化 ----
        if self.need_truncated:
            self._truncate(getattr(args, "seq_lens", None))
        if self.need_normalize:
            # ★ Phase 3: 外部预训练特征 (如 Data2Vec) 无需再归一化
            self._normalize(skip_feature_A=bool(getattr(args, 'feature_A', '')),
                           skip_feature_V=bool(getattr(args, 'feature_V', '')))

        # ---- ★ Phase 5: 对话上下文初始化 (必须在外部特征加载后, 确保用新维度) ----
        if self.use_context:
            self._init_context(split, mode)

        # ---- 6) 形状校验 ----
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
        统一为 (N, L, D). 如果检测到 (N, D, L) (即 D < L), 则转置.
        """
        if x.ndim != 3:
            return x
        if x.shape[1] < x.shape[2]:
            return np.transpose(x, (0, 2, 1))
        return x

    # ---- ★ Phase 5: 对话上下文初始化 ----
    def _init_context(self, split: dict, mode: str):
        """
        加载 context 特征. 
        优先从原始 pkl 读取预计算的 context, 否则使用 offset-based fallback
        (context = 前一条话语的特征, 首条用自身).
        """
        # 文本 context
        if 'context_text' in split:
            self.context_text = np.asarray(split['context_text'], dtype=np.float32)
        elif self.use_bert and 'text_bert' in split:
            self.context_text = np.roll(split['text_bert'], shift=1, axis=0)
            self.context_text[0] = split['text_bert'][0]
        elif 'text' in split:
            raw_text = np.asarray(split['text'], dtype=np.float32)
            raw_text = self._maybe_transpose(raw_text)
            self.context_text = np.roll(raw_text, shift=1, axis=0)
            self.context_text[0] = raw_text[0]
        else:
            self.context_text = np.roll(self.text, shift=1, axis=0)
            self.context_text[0] = self.text[0]

        # 音频 context
        if 'context_audio' in split:
            self.context_audio = np.asarray(split['context_audio'], dtype=np.float32)
        else:
            self.context_audio = np.roll(self.audio, shift=1, axis=0)
            self.context_audio[0] = self.audio[0]

        # 视频 context
        if 'context_vision' in split:
            self.context_vision = np.asarray(split['context_vision'], dtype=np.float32)
        else:
            self.context_vision = np.roll(self.vision, shift=1, axis=0)
            self.context_vision[0] = self.vision[0]

        logger.info(
            f"[{mode}] Context features loaded "
            f"(text={tuple(self.context_text.shape)}, "
            f"audio={tuple(self.context_audio.shape)}, "
            f"video={tuple(self.context_vision.shape)})"
        )

    def _truncate(self, seq_lens: Optional[List[int]]):
        """
        非对齐序列下按 seq_lens=[L_t, L_a, L_v] 截断 / pad.
        """
        if not seq_lens or len(seq_lens) != 3:
            return
        L_t, L_a, L_v = seq_lens

        def _fit(x: np.ndarray, L: int) -> np.ndarray:
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

    def _normalize(self, skip_feature_A: bool = False, skip_feature_V: bool = False):
        """按特征维度做均值 / 方差归一化.
        外部预训练特征 (Data2Vec 等) 可跳过归一化, 因预训练模型已有 normalize.
        """
        eps = 1e-6
        skip_map = {"audio": skip_feature_A, "vision": skip_feature_V}
        for attr in ("audio", "vision"):
            if skip_map.get(attr, False):
                continue
            x = getattr(self, attr)
            mean = x.mean(axis=(0, 1), keepdims=True)
            std = x.std(axis=(0, 1), keepdims=True)
            setattr(self, attr, (x - mean) / (std + eps))

    # ---------- Dataset API ----------
    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = {
            "text": torch.from_numpy(self.text[idx]),
            "id": self.ids[idx],
            "index": torch.tensor(idx, dtype=torch.long),
        }

        # ---- ★ CAGMamba 对齐: WAV 在线加载 ----
        if self.use_wav_audio and self._feature_extractor is not None:
            sid = str(self.ids[idx]).replace('$_$', '_')
            wav_path = self._wav_paths.get(sid, '')
            if wav_path:
                try:
                    import torchaudio
                    sound, sr = torchaudio.load(wav_path)
                    sound_mono = torch.mean(sound, dim=0, keepdim=False)
                except Exception:
                    import librosa
                    audio_np, _ = librosa.load(wav_path, sr=16000, mono=True)
                    sound_mono = torch.from_numpy(audio_np.astype(np.float32))
                feat = self._feature_extractor(
                    sound_mono, sampling_rate=16000, max_length=96000,
                    return_attention_mask=True, truncation=True, padding="max_length",
                )
                sample["audio"] = torch.tensor(np.array(feat['input_values']), dtype=torch.float32).squeeze()
            else:
                sample["audio"] = torch.zeros(96000, dtype=torch.float32)
        else:
            sample["audio"] = torch.from_numpy(self.audio[idx].copy())

        sample["vision"] = torch.from_numpy(self.vision[idx].copy())

        # ★ Phase 5: 上下文特征
        if self.use_context:
            sample["context_text"] = torch.from_numpy(
                self.context_text[idx].copy() if self.use_bert
                else np.asarray(self.context_text[idx])
            )
            sample["context_audio"] = torch.from_numpy(
                self.context_audio[idx].copy()
            )
            sample["context_video"] = torch.from_numpy(
                self.context_vision[idx].copy()
            )

        # 多任务标签字典
        labels = {}
        for tag, arr in self.labels.items():
            if self.dataset_name in REGRESSION_DATASETS:
                labels[tag] = torch.tensor(arr[idx], dtype=torch.float32)
            else:
                labels[tag] = torch.tensor(arr[idx], dtype=torch.long)
        sample["labels"] = labels
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