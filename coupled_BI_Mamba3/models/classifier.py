"""
MSAClassifier: 多模态情感识别任务头 / 整体模型封装
==================================================

接收三模态融合输出 (audio / visual / lexical), 输出情感预测。
    - 回归任务 (MOSI / MOSEI):  num_classes=1, 输出情感分数
    - 分类任务 (IEMOCAP / MELD): num_classes=K, 输出类别 logits

模型流水线:
    T / A / V
      ↓ BERT(T) + 线性投影
      ↓ 序列对齐 (adaptive_avg_pool1d)
      ↓ [ISMEncoder × ism_depth]  ← 单模态序列建模 (GLCE + BSSM), 各模态独立
      ↓ [CoupledMamba3Fork × num_layers]  ← 跨模态双向状态空间融合
      ↓ 池化 → 拼接 → 分类头
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .coupled_mamba3_fork import CoupledMamba3Fork

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from layers.ism import ISMEncoder


# -------------------------------------------------------------
# BERT 文本编码器
# -------------------------------------------------------------
class BertTextEncoder(nn.Module):
    """
    输入: text_bert (B, 3, L)  包含 input_ids / attention_mask / token_type_ids
    输出: (B, L, 768)
    """

    def __init__(self, pretrained: str = "bert-base-uncased", finetune: bool = True,
                 strict: bool = True):
        super().__init__()
        try:
            from transformers import BertModel
            self.bert = BertModel.from_pretrained(pretrained)
            self.out_dim = self.bert.config.hidden_size
            self.use_hf = True
        except Exception as e:
            msg = (
                f"[BertTextEncoder] 加载 HuggingFace BertModel 失败: {e}\n"
                f"  常见原因:\n"
                f"  1) libstdc++ 版本太旧 -> export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH\n"
                f"  2) 未联网下载 pretrained\n"
                f"  3) transformers 未安装 -> pip install transformers"
            )
            if strict:
                raise RuntimeError(msg) from e
            print(f"[WARN] {msg}\n  => fallback 到 nn.Embedding (效果会差)")
            self.bert = nn.Embedding(30522, 768, padding_idx=0)
            self.out_dim = 768
            self.use_hf = False
        if self.use_hf and not finetune:
            for p in self.bert.parameters():
                p.requires_grad = False

    def forward(self, text_bert: torch.Tensor) -> torch.Tensor:
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


# -------------------------------------------------------------
# MSAClassifier
# -------------------------------------------------------------
class MSAClassifier(nn.Module):
    """
    端到端多模态情感识别模型:

        T / A / V
          → BERT(T) + 线性投影 → d_model
          → [ISMEncoder]  单模态序列建模 (GLCE + BSSM), 各模态独立权重
          → [CoupledMamba3Fork × num_layers]  跨模态双向 SSM 融合
          → 池化 → 拼接 → 分类头

    新增参数:
        ism_depth  (int): ISMEncoder 堆叠层数, 0 = 关闭 ISM (等价于原始模型)
        ism_seq_len (int): ISM 期望的序列长度, 应与对齐后的 Lt 一致
        ism_d_state (int): ISM 内部 Mamba SSM 的状态维度
    """

    def __init__(
        self,
        text_input_dim: int,
        audio_input_dim: int,
        video_input_dim: int,
        d_model: int = 128,
        num_layers: int = 2,
        num_classes: int = 1,
        task_type: str = "regression",
        pool_type: str = "mean",
        dropout: float = 0.1,
        use_bert: bool = True,
        bert_pretrained: str = "bert-base-uncased",
        bert_finetune: bool = True,
        # --- ISM 参数 ---
        ism_depth: int = 1,          # 0 = 不加 ISM
        ism_seq_len: int = 50,       # 对齐后序列长度
        ism_d_state: int = 16,       # ISM 内 Mamba SSM 状态维
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
        self.ism_depth = ism_depth

        # 0) 文本编码器 (可选 BERT)
        if use_bert:
            self.text_encoder = BertTextEncoder(bert_pretrained, finetune=bert_finetune)
            text_feat_dim = self.text_encoder.out_dim   # 768
        else:
            self.text_encoder = None
            text_feat_dim = text_input_dim

        # 1) 输入投影
        self.proj_text  = nn.Linear(text_feat_dim, d_model, **factory_kwargs)
        self.proj_audio = nn.Linear(audio_input_dim, d_model, **factory_kwargs)
        self.proj_video = nn.Linear(video_input_dim, d_model, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)

        # 2) ISM — 单模态序列建模, 放在跨模态融合之前
        #    三个模态各自独立一套 ISMEncoder (参数不共享)
        if ism_depth > 0:
            self.ism_text  = ISMEncoder(
                d_model=d_model, seq_len=ism_seq_len, depth=ism_depth,
                d_state=ism_d_state, d_conv=4, expand=2, dropout=dropout,
            )
            self.ism_audio = ISMEncoder(
                d_model=d_model, seq_len=ism_seq_len, depth=ism_depth,
                d_state=ism_d_state, d_conv=4, expand=2, dropout=dropout,
            )
            self.ism_video = ISMEncoder(
                d_model=d_model, seq_len=ism_seq_len, depth=ism_depth,
                d_state=ism_d_state, d_conv=4, expand=2, dropout=dropout,
            )
        else:
            self.ism_text = self.ism_audio = self.ism_video = None

        # 3) 跨模态融合: 堆叠 num_layers 层 CoupledMamba3Fork
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

        # 4) 分类头
        fused_dim = 3 * d_model
        self.fusion_norm = nn.LayerNorm(fused_dim)
        self.head = nn.Sequential(
            nn.Linear(fused_dim, d_model, **factory_kwargs),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes, **factory_kwargs),
        )

    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    def forward(
        self,
        text: torch.Tensor,          # (B, 3, L_t) if use_bert else (B, L_t, Dt)
        audio: torch.Tensor,         # (B, L_a, Da)
        video: torch.Tensor,         # (B, L_v, Dv)
        cu_seqlens: Optional[torch.Tensor] = None,
    ):
        # ── 0) 文本嵌入 ──────────────────────────────────────────────
        if self.use_bert and self.text_encoder is not None:
            text = self.text_encoder(text)          # (B, L, 768)

        # ── 1) 投影到 d_model ────────────────────────────────────────
        xt = self.dropout(F.gelu(self.proj_text(text)))     # (B, Lt, D)
        xa = self.dropout(F.gelu(self.proj_audio(audio)))   # (B, La, D)
        xv = self.dropout(F.gelu(self.proj_video(video)))   # (B, Lv, D)

        # ── 2) 序列对齐 (以 text 长度为基准) ────────────────────────
        Lt = xt.size(1)
        if xa.size(1) != Lt:
            xa = F.adaptive_avg_pool1d(xa.transpose(1, 2), Lt).transpose(1, 2)
        if xv.size(1) != Lt:
            xv = F.adaptive_avg_pool1d(xv.transpose(1, 2), Lt).transpose(1, 2)

        # ── 3) ISM — 单模态序列建模 (跨模态融合之前) ────────────────
        if self.ism_depth > 0:
            xt = self.ism_text(xt)    # (B, Lt, D)
            xa = self.ism_audio(xa)   # (B, Lt, D)
            xv = self.ism_video(xv)   # (B, Lt, D)

        # ── 4) CoupledMamba3Fork — 跨模态融合 ───────────────────────
        out_a, out_v, out_l = xa, xv, xt
        for layer in self.layers:
            out_a, out_v, out_l = layer(out_a, out_v, out_l, cu_seqlens=cu_seqlens)

        # ── 5) 池化 + 拼接 + 分类头 ─────────────────────────────────
        pa = self._pool(out_a)
        pv = self._pool(out_v)
        pl = self._pool(out_l)
        fused  = self.fusion_norm(torch.cat([pa, pv, pl], dim=-1))
        logits = self.head(fused)     # (B, num_classes)
        return logits