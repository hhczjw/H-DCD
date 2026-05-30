"""
Coupled-BI-Mamba3 models package
================================

主要模块:
    - CoupledMamba3Fork / CrossMamba3Cell: 基于 Mamba-3 的跨模态 Q/K/V 融合 (策略 A)
    - MSAClassifier: MOSI/MOSEI 回归头 / IEMOCAP/MELD 分类头
    - TextPretrainedEncoder: 通用文本编码器 (BERT/RoBERTa/DeBERTa)
    - AudioPretrainedEncoder: 预训练音频编码器 (Data2Vec/HuBERT/WavLM)
    - context_fusion: 对话上下文特征级拼接 (Phase 5)
"""

from .coupled_mamba3_fork import (
    CoupledMamba3Fork,
    CrossMamba3Cell,
    MAMBA3_AVAILABLE,
)
from .classifier import MSAClassifier
from .text_encoder import TextPretrainedEncoder
from .context_fusion import encode_with_context

__all__ = [
    "CoupledMamba3Fork",
    "CrossMamba3Cell",
    "MAMBA3_AVAILABLE",
    "MSAClassifier",
    "TextPretrainedEncoder",
    "encode_with_context",
]