"""
MSAClassifier: 多模态情感识别任务头 / 整体模型封装
==================================================

接收三模态融合输出 (audio / visual / lexical), 输出情感预测。
    - 回归任务 (MOSI / MOSEI):  num_classes=1, 输出情感分数
    - 分类任务 (IEMOCAP / MELD): num_classes=K, 输出类别 logits

模型流水线:
    T / A / V
      ↓ BERT(T) + 2层MLP投影
      ↓ 序列对齐 (adaptive_avg_pool1d)
      ↓ [ISMEncoder × ism_depth]  ← 单模态序列建模 (GLCE + BSSM), 各模态独立
      ↓ [CoupledMamba3Fork × num_layers]  ← 跨模态双向状态空间融合
      ↓ Attention 池化 → 拼接 → 分类头

改进:
    - 投影层: 单线性层 → 2层 MLP + LayerNorm (减少信息损失)
    - 池化: mean pooling → Attention Pooling (学习序列重要性加权)
    - 新增 get_modal_embeddings() 方法 (供 InfoNCE 对比损失使用)
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .coupled_mamba3_fork import CoupledMamba3Fork

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from layers.ism import ISMEncoder


# -------------------------------------------------------------
# 2层 MLP 投影
# -------------------------------------------------------------
class MLPProjection(nn.Module):
    """2层 MLP + LayerNorm 投影, 减少从高维到低维的信息损失"""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        mid_dim = (in_dim + out_dim) // 2
        self.net = nn.Sequential(
            nn.Linear(in_dim, mid_dim),
            nn.LayerNorm(mid_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mid_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# -------------------------------------------------------------
# Attention Pooling
# -------------------------------------------------------------
class AttentionPooling(nn.Module):
    """学习序列维度的注意力权重进行加权池化"""

    def __init__(self, d_model: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, D) -> (B, D)"""
        scores = self.attn(x)                      # (B, L, 1)
        weights = torch.softmax(scores, dim=1)     # (B, L, 1)
        return (x * weights).sum(dim=1)            # (B, D)


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
    端到端多模态情感识别模型 (改进版):

        T / A / V
          → BERT(T) + 2层MLP投影 → d_model
          → [ISMEncoder]  单模态序列建模 (GLCE + BSSM), 各模态独立权重
          → [CoupledMamba3Fork × num_layers]  跨模态双向 SSM 融合
          → Attention 池化 → 拼接 → 分类头
    """

    def __init__(
        self,
        text_input_dim: int,
        audio_input_dim: int,
        video_input_dim: int,
        d_model: int = 256,
        num_layers: int = 3,
        num_classes: int = 1,
        task_type: str = "regression",
        pool_type: str = "attention",
        dropout: float = 0.15,
        use_bert: bool = True,
        bert_pretrained: str = "bert-base-uncased",
        bert_finetune: bool = True,
        # --- ISM 参数 ---
        ism_depth: int = 2,
        ism_seq_len: int = 50,
        ism_d_state: int = 32,
        ism_mixer_type: str = "bimamba",     # "bimamba" (Mamba-2) | "bimamba3" (Mamba-3)
        ism_bimamba3_headdim: int = 64,
        ism_bimamba3_ngroups: int = 1,
        ism_bimamba3_rope_fraction: float = 0.5,
        ism_bimamba3_chunk_size: int = 64,
        ism_bimamba3_is_mimo: bool = False,
        ism_bimamba3_mimo_rank: int = 4,
        ism_bimamba3_is_outproj_norm: bool = False,
        ism_bimamba3_fusion: str = "add_divide2",
        ism_bimamba3_share_mimo: bool = True,
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
        # --- 辅助分类头 (回归任务专用, 用于直接优化 Acc7) ---
        aux_num_classes: int = 0,   # 0 = 不启用; MOSI 推荐 7
        # --- 模态级 sub_loss (回归任务专用, 对齐 MSAmba 的 sub_fc_T/A/V) ---
        # 0.0 = 关闭; 推荐 0.3~0.5; 仅当 ism_depth>0 时生效
        sub_loss_lambda: float = 0.0,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.aux_num_classes = int(aux_num_classes)
        self.sub_loss_lambda = float(sub_loss_lambda)
        factory_kwargs = {"device": device, "dtype": dtype}
        self.task_type = task_type
        self.pool_type = pool_type
        self.num_classes = num_classes
        self.use_bert = use_bert
        self.ism_depth = ism_depth
        self.d_model = d_model

        # 0) 文本编码器 (可选 BERT)
        if use_bert:
            self.text_encoder = BertTextEncoder(bert_pretrained, finetune=bert_finetune)
            text_feat_dim = self.text_encoder.out_dim   # 768
        else:
            self.text_encoder = None
            text_feat_dim = text_input_dim

        # 1) 输入投影 — 改为 2层 MLP + LayerNorm
        self.proj_text  = MLPProjection(text_feat_dim, d_model, dropout=dropout)
        self.proj_audio = MLPProjection(audio_input_dim, d_model, dropout=dropout)
        self.proj_video = MLPProjection(video_input_dim, d_model, dropout=dropout)

        # 2) ISM — 单模态序列建模
        if ism_depth > 0:
            ism_kwargs = dict(
                d_model=d_model, seq_len=ism_seq_len, depth=ism_depth,
                d_state=ism_d_state, d_conv=4, expand=2, dropout=dropout,
                mixer_type=ism_mixer_type,
                bimamba3_headdim=ism_bimamba3_headdim,
                bimamba3_ngroups=ism_bimamba3_ngroups,
                bimamba3_rope_fraction=ism_bimamba3_rope_fraction,
                bimamba3_chunk_size=ism_bimamba3_chunk_size,
                bimamba3_is_mimo=ism_bimamba3_is_mimo,
                bimamba3_mimo_rank=ism_bimamba3_mimo_rank,
                bimamba3_is_outproj_norm=ism_bimamba3_is_outproj_norm,
                bimamba3_fusion=ism_bimamba3_fusion,
                bimamba3_share_mimo=ism_bimamba3_share_mimo,
            )
            self.ism_text  = ISMEncoder(**ism_kwargs)
            self.ism_audio = ISMEncoder(**ism_kwargs)
            self.ism_video = ISMEncoder(**ism_kwargs)
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

        # 4) 池化
        if pool_type == "attention":
            self.pool_text  = AttentionPooling(d_model)
            self.pool_audio = AttentionPooling(d_model)
            self.pool_video = AttentionPooling(d_model)
        else:
            self.pool_text = self.pool_audio = self.pool_video = None

        # 5) 分类头
        fused_dim = 3 * d_model
        self.fusion_norm = nn.LayerNorm(fused_dim)
        self.head = nn.Sequential(
            nn.Linear(fused_dim, d_model, **factory_kwargs),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes, **factory_kwargs),
        )

        # 5.b) 辅助分类头 (仅回归任务启用): 用于直接优化 Acc7
        # 共享 backbone, 单独的轻量 head 输出 7 类 logits
        if self.aux_num_classes > 0 and task_type == "regression":
            self.aux_head = nn.Sequential(
                nn.Linear(fused_dim, d_model, **factory_kwargs),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, self.aux_num_classes, **factory_kwargs),
            )
        else:
            self.aux_head = None

        # 5.c) 模态级 sub_loss head (对齐 MSAmba sub_fc_T/A/V)
        # 各模态 ISM 输出的 CLS token (D 维) → 1 维回归
        # 仅当 sub_loss_lambda > 0 且 ism_depth > 0 且 task_type=regression 时启用
        self.use_sub_loss = (
            self.sub_loss_lambda > 0.0
            and ism_depth > 0
            and task_type == "regression"
        )
        if self.use_sub_loss:
            self.sub_fc_T = nn.Linear(d_model, 1, **factory_kwargs)
            self.sub_fc_A = nn.Linear(d_model, 1, **factory_kwargs)
            self.sub_fc_V = nn.Linear(d_model, 1, **factory_kwargs)
        else:
            self.sub_fc_T = self.sub_fc_A = self.sub_fc_V = None

    # ------------------------------------------------------------------
    def _pool(self, x: torch.Tensor, pool_module=None) -> torch.Tensor:
        """x: (B, L, D) -> (B, D)"""
        if self.pool_type == "attention" and pool_module is not None:
            return pool_module(x)
        elif self.pool_type == "mean":
            return x.mean(dim=1)
        elif self.pool_type == "last":
            return x[:, -1]
        elif self.pool_type == "cls":
            return x[:, 0]
        else:
            return x.mean(dim=1)

    # ------------------------------------------------------------------
    def _encode(self, text, audio, video, cu_seqlens=None, return_ism_cls: bool = False):
        """编码到融合后的三模态序列表征 (共享流水线)

        Args:
            return_ism_cls: True 时同时返回 ISM 阶段的 cls_T/A/V (sub_loss 用)
        """
        # 0) 文本嵌入
        if self.use_bert and self.text_encoder is not None:
            text = self.text_encoder(text)

        # 1) 投影到 d_model (2层 MLP)
        xt = self.proj_text(text)
        xa = self.proj_audio(audio)
        xv = self.proj_video(video)

        # 2) 序列对齐
        Lt = xt.size(1)
        if xa.size(1) != Lt:
            xa = F.adaptive_avg_pool1d(xa.transpose(1, 2), Lt).transpose(1, 2)
        if xv.size(1) != Lt:
            xv = F.adaptive_avg_pool1d(xv.transpose(1, 2), Lt).transpose(1, 2)

        # 3) ISM
        ism_cls_t = ism_cls_a = ism_cls_v = None
        if self.ism_depth > 0:
            if return_ism_cls:
                xt, ism_cls_t = self.ism_text(xt, return_cls=True)
                xa, ism_cls_a = self.ism_audio(xa, return_cls=True)
                xv, ism_cls_v = self.ism_video(xv, return_cls=True)
            else:
                xt = self.ism_text(xt)
                xa = self.ism_audio(xa)
                xv = self.ism_video(xv)

        # 4) CoupledMamba3Fork
        out_a, out_v, out_l = xa, xv, xt
        for layer in self.layers:
            out_a, out_v, out_l = layer(out_a, out_v, out_l, cu_seqlens=cu_seqlens)

        if return_ism_cls:
            return out_l, out_a, out_v, ism_cls_t, ism_cls_a, ism_cls_v
        return out_l, out_a, out_v

    # ------------------------------------------------------------------
    def forward(
        self,
        text: torch.Tensor,
        audio: torch.Tensor,
        video: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
    ):
        if self.use_sub_loss:
            out_l, out_a, out_v, c_t, c_a, c_v = self._encode(
                text, audio, video, cu_seqlens, return_ism_cls=True
            )
        else:
            out_l, out_a, out_v = self._encode(text, audio, video, cu_seqlens)
            c_t = c_a = c_v = None

        # 池化 + 拼接 + 分类头
        pl = self._pool(out_l, self.pool_text if self.pool_type == "attention" else None)
        pa = self._pool(out_a, self.pool_audio if self.pool_type == "attention" else None)
        pv = self._pool(out_v, self.pool_video if self.pool_type == "attention" else None)
        fused  = self.fusion_norm(torch.cat([pl, pa, pv], dim=-1))
        logits = self.head(fused)

        # 当未启用任何辅助 head 时, 保持向后兼容: 直接返回 Tensor
        has_aux = self.aux_head is not None
        if not has_aux and not self.use_sub_loss:
            return logits

        out: dict = {"logits": logits}
        if has_aux:
            out["aux_logits"] = self.aux_head(fused)
        if self.use_sub_loss:
            out["sub_T"] = self.sub_fc_T(c_t)   # (B, 1)
            out["sub_A"] = self.sub_fc_A(c_a)
            out["sub_V"] = self.sub_fc_V(c_v)
        return out

    # ------------------------------------------------------------------
    def get_modal_embeddings(
        self,
        text: torch.Tensor,
        audio: torch.Tensor,
        video: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        返回三模态的池化表征 (B, D), 供 InfoNCE 对比损失使用.
        """
        out_l, out_a, out_v = self._encode(text, audio, video)
        pl = self._pool(out_l, self.pool_text if self.pool_type == "attention" else None)
        pa = self._pool(out_a, self.pool_audio if self.pool_type == "attention" else None)
        pv = self._pool(out_v, self.pool_video if self.pool_type == "attention" else None)
        return pl, pa, pv