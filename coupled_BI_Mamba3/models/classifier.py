"""
MSAClassifier: 多模态情感识别任务头 / 整体模型封装
==================================================

接收三模态融合输出 (audio / visual / lexical), 输出情感预测。
    - 回归任务 (MOSI / MOSEI):  num_classes=1, 输出情感分数
    - 分类任务 (IEMOCAP / MELD): num_classes=K, 输出类别 logits
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .coupled_mamba3_fork import CoupledMamba3Fork


class MSAClassifier(nn.Module):
    """
    端到端多模态情感识别模型:
        T / A / V 输入 --(投影)--> d_model --> CoupledMamba3Fork × L --> 池化 --> 分类头
    """

    def __init__(
        self,
        text_input_dim: int,
        audio_input_dim: int,
        video_input_dim: int,
        d_model: int = 128,
        num_layers: int = 2,
        num_classes: int = 1,
        task_type: str = "regression",      # "regression" | "classification"
        pool_type: str = "mean",            # "mean" | "last" | "cls"
        dropout: float = 0.1,
        # --- CoupledMamba3Fork 透传 ---
        d_state: int = 64,
        expand: int = 2,
        headdim: int = 32,
        ngroups: int = 1,
        rope_fraction: float = 0.5,
        is_mimo: bool = False,
        mimo_rank: int = 4,
        chunk_size: int = 64,
        is_outproj_norm: bool = False,
        device=None,
        dtype=None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.task_type = task_type
        self.pool_type = pool_type
        self.num_classes = num_classes

        # 1) 输入投影 (Linear, 也可换成 Conv1d)
        self.proj_text = nn.Linear(text_input_dim, d_model, **factory_kwargs)
        self.proj_audio = nn.Linear(audio_input_dim, d_model, **factory_kwargs)
        self.proj_video = nn.Linear(video_input_dim, d_model, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)

        # 2) 堆叠 L 层 CoupledMamba3Fork
        self.layers = nn.ModuleList([
            CoupledMamba3Fork(
                d_model=d_model, d_state=d_state, expand=expand, headdim=headdim,
                ngroups=ngroups, rope_fraction=rope_fraction,
                is_mimo=is_mimo, mimo_rank=mimo_rank,
                chunk_size=chunk_size, is_outproj_norm=is_outproj_norm,
                device=device, dtype=dtype,
            )
            for _ in range(num_layers)
        ])

        # 3) 融合 + 分类头 (拼接后线性)
        fused_dim = 3 * d_model
        self.fusion_norm = nn.LayerNorm(fused_dim)
        self.head = nn.Sequential(
            nn.Linear(fused_dim, d_model, **factory_kwargs),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes, **factory_kwargs),
        )

    def _pool(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, D) -> (B, D)"""
        if self.pool_type == "mean":
            return x.mean(dim=1)
        elif self.pool_type == "last":
            return x[:, -1]
        elif self.pool_type == "cls":
            return x[:, 0]
        else:
            raise ValueError(f"Unknown pool_type: {self.pool_type}")

    def forward(
        self,
        text: torch.Tensor,          # (B, L_t, Dt)
        audio: torch.Tensor,         # (B, L_a, Da)
        video: torch.Tensor,         # (B, L_v, Dv)
        cu_seqlens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # 注意: CoupledMamba3Fork 默认三个模态等长, 这里假设已对齐 (need_data_aligned=True)
        # 不对齐场景需上游裁切/pad 到相同 L
        xt = self.dropout(F.gelu(self.proj_text(text)))        # (B, L, D)
        xa = self.dropout(F.gelu(self.proj_audio(audio)))
        xv = self.dropout(F.gelu(self.proj_video(video)))

        # 对齐到相同 L (按最短截断, 简单策略; 工程中可改为 pad / interp)
        L = min(xt.size(1), xa.size(1), xv.size(1))
        xt, xa, xv = xt[:, :L], xa[:, :L], xv[:, :L]

        # 逐层跨模态融合: audio / visual / lexical
        out_a, out_v, out_l = xa, xv, xt
        for layer in self.layers:
            out_a, out_v, out_l = layer(out_a, out_v, out_l, cu_seqlens=cu_seqlens)

        # 池化 + 拼接
        pa = self._pool(out_a)
        pv = self._pool(out_v)
        pl = self._pool(out_l)
        fused = self.fusion_norm(torch.cat([pa, pv, pl], dim=-1))
        logits = self.head(fused)                              # (B, num_classes)
        return logits