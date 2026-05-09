"""
Coupled-BI-Mamba3 models package
================================

主要模块:
    - CoupledMamba3Fork / CrossMamba3Cell: 基于 Mamba-3 的跨模态 Q/K/V 融合 (策略 A)
    - MSAClassifier: MOSI/MOSEI 回归头 / IEMOCAP/MELD 分类头
"""

from .coupled_mamba3_fork import (
    CoupledMamba3Fork,
    CrossMamba3Cell,
    MAMBA3_AVAILABLE,
)
from .classifier import MSAClassifier

__all__ = [
    "CoupledMamba3Fork",
    "CrossMamba3Cell",
    "MAMBA3_AVAILABLE",
    "MSAClassifier",
]