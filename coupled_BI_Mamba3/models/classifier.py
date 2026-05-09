"""
MSAClassifier: 多模态情感识别任务头 / 整体模型封装
==================================================

接收三模态融合输出 (audio / visual / lexical), 输出情感预测。
    - 回归任务 (MOSI / MOSEI):  num_classes=1, 输出情感分数
    - 分类任务 (IEMOCAP / MELD): num_classes=K, 输出类别 logits

关键修复 (vs. 初版):
    1) 内嵌 BertTextEncoder, 直接吃 text_bert (B, 3, L) 三通道 input_ids/mask/segment;
       初版把 token_id 当 1 维信号, 文本语义完全丢失 (MOSI 飙到 MAE=1.34 的主因).
    2) 不在这里做 (B, L, D) 截断对齐; 由 forward 接收已对齐/已 pad 的等长张量.
       MOSI unaligned_50.pkl 中 text/audio/vision 都是 L=50, 本来就等长.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .coupled_mamba3_fork import CoupledMamba3Fork


# -------------------------------------------------------------
# BERT 文本编码器 (可选; 不装 transformers 时自动 fallback 为 nn.Embedding)
# -------------------------------------------------------------
class BertTextEncoder(nn.Module):
    """
    输入: text_bert (B, 3, L)  包含 input_ids / attention_mask / token_type_ids
    输出: (B, L, 768)
    """

    def __init__(self, pretrained: str = "bert-base-uncased", finetune: bool = True):
        super().__init__()
        try:
            from transformers import BertModel
            self.bert = BertModel.from_pretrained(pretrained)
            self.out_dim = self.bert.config.hidden_size
            self.use_hf = True
        except Exception as e:
            # fallback: 随机初始化 embedding (至少保留语义 token 区分度)
            print(f"[WARN] transformers/BertModel 不可用, 用 nn.Embedding fallback: {e}")
            self.bert = nn.Embedding(30522, 768, padding_idx=0)
            self.out_dim = 768
            self.use_hf = False
        if self.use_hf and not finetune:
            for p in self.bert.parameters():
                p.requires_grad = False

    def forward(self, text_bert: torch.Tensor) -> torch.Tensor:
        # text_bert: (B, 3, L) float, 需要转 long
        input_ids = text_bert[:, 0].long()
        if self.use_hf:
            attention_mask = text_bert[:, 1].long()
            token_type_ids = text_bert[:, 2].long()
            out = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            return out.last_hidden_state  # (B, L, 768)
        return self.bert(input_ids)  # (B, L, 768)


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
        use_bert: bool = True,
        bert_pretrained: str = "bert-base-uncased",
        bert_finetune: bool = True,
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
        self.use_bert = use_bert

        # 0) 文本编码器 (可选 BERT)
        if use_bert:
            self.text_encoder = BertTextEncoder(bert_pretrained, finetune=bert_finetune)
            text_feat_dim = self.text_encoder.out_dim   # 768
        else:
            self.text_encoder = None
            text_feat_dim = text_input_dim

        # 1) 输入投影 (Linear, 也可换成 Conv1d)
        self.proj_text = nn.Linear(text_feat_dim, d_model, **factory_kwargs)
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
        text: torch.Tensor,          # (B, 3, L_t) if use_bert else (B, L_t, Dt)
        audio: torch.Tensor,         # (B, L_a, Da)
        video: torch.Tensor,         # (B, L_v, Dv)
        cu_seqlens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # 0) 文本嵌入: BERT (B,3,L) -> (B, L, 768)
        if self.use_bert and self.text_encoder is not None:
            text = self.text_encoder(text)

        # 1) 投影到 d_model
        xt = self.dropout(F.gelu(self.proj_text(text)))        # (B, Lt, D)
        xa = self.dropout(F.gelu(self.proj_audio(audio)))      # (B, La, D)
        xv = self.dropout(F.gelu(self.proj_video(video)))      # (B, Lv, D)

        # 2) 对齐到相同 L
        # 原实现: 最短截断 -> 会把 audio/vision 的大量帧丢掉;
        # 这里改为: 以 text 长度为基准, 对 audio/vision 做自适应平均池化 (更平滑保留信息)
        Lt = xt.size(1)
        if xa.size(1) != Lt:
            xa = F.adaptive_avg_pool1d(xa.transpose(1, 2), Lt).transpose(1, 2)
        if xv.size(1) != Lt:
            xv = F.adaptive_avg_pool1d(xv.transpose(1, 2), Lt).transpose(1, 2)

        # 3) 逐层跨模态融合: audio / visual / lexical
        out_a, out_v, out_l = xa, xv, xt
        for layer in self.layers:
            out_a, out_v, out_l = layer(out_a, out_v, out_l, cu_seqlens=cu_seqlens)

        # 4) 池化 + 拼接
        pa = self._pool(out_a)
        pv = self._pool(out_v)
        pl = self._pool(out_l)
        fused = self.fusion_norm(torch.cat([pa, pv, pl], dim=-1))
        logits = self.head(fused)                              # (B, num_classes)
        return logits