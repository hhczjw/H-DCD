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
      ↓ Mean 池化 → 拼接 → 分类头
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .coupled_mamba3_fork import CoupledMamba3Fork
from .pairwise_cross_mamba3 import PairwiseCrossMamba3Fork
from .audio_encoder import AudioPretrainedEncoder      # Phase 3: 预训练音频编码器
from .text_encoder import TextPretrainedEncoder        # Phase 2: 通用文本编码器
from .context_fusion import encode_with_context         # Phase 5: 对话上下文融合

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from layers.ism import ISMEncoder


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
          → Mean 池化 → 拼接 → 分类头
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
        ism_depth: int = 3,
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
        # --- 跨模态 V_self 注入 (0=关闭即旧行为) ---
        v_self_ratio: float = 0.0,
        # --- 多任务开关 (SIMS/SIMS2 用于自动启用 sub_fc_T/A/V) ---
        multi_task: bool = False,
        # --- ★ Phase 3: 预训练音频编码器 ---
        use_pretrained_audio: bool = False,
        audio_pretrained: str = "facebook/data2vec-audio-base-960h",
        audio_finetune: bool = True,
        # --- ★ 跳过音频 ISM (Data2Vec 已编码时序) ---
        skip_audio_ism: bool = False,
        use_pairwise_mamba: bool = False,
        # ★ Phase 19: BSSM 门控 (CAGMamba 对齐)
        use_bssm_gate: bool = False,
        bssm_gate_expand: int = 2,
        # ★ Phase 20: GCMN 三流门控融合
        use_gcmn_gate: bool = False,
        # ★ Phase 17: 双向上下文 (CAGMamba 对齐)
        bidirectional: bool = False,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.multi_task = bool(multi_task)
        self.use_pretrained_audio = use_pretrained_audio
        self.skip_audio_ism = skip_audio_ism
        factory_kwargs = {"device": device, "dtype": dtype}
        self.task_type = task_type
        self.pool_type = pool_type
        self.num_classes = num_classes
        self.use_bert = use_bert
        self.ism_depth = ism_depth
        self.d_model = d_model
        self.bidirectional = bidirectional  # ★ Phase 17

        # 0) 文本编码器 (可选 BERT)
        if use_bert:
            self.text_encoder = TextPretrainedEncoder(
            pretrained=bert_pretrained,     # ★ 默认 "roberta-base"
            finetune=bert_finetune,
            )
            text_feat_dim = self.text_encoder.out_dim  # ★ 自动获取 (768 或 1024)
        else:
            self.text_encoder = None
            text_feat_dim = text_input_dim

        # ★ 0.b) 音频编码器 (Phase 3: Data2Vec 预训练)
        if use_pretrained_audio:
            self.audio_encoder = AudioPretrainedEncoder(
                pretrained=audio_pretrained,
                finetune=audio_finetune,
                target_frames=ism_seq_len,
            )
            audio_feat_dim = self.audio_encoder.out_dim  # 768 or 1024
        else:
            self.audio_encoder = None
            audio_feat_dim = audio_input_dim

        # 1) 输入投影 — 单线性层
        self.proj_text  = nn.Linear(text_feat_dim, d_model, **factory_kwargs)
        self.proj_audio = nn.Linear(audio_feat_dim, d_model, **factory_kwargs)
        self.proj_video = nn.Linear(video_input_dim, d_model, **factory_kwargs)

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
                use_bssm_gate=use_bssm_gate,
                bssm_gate_expand=bssm_gate_expand,
            )
            self.ism_text  = ISMEncoder(**ism_kwargs)
            self.ism_audio = ISMEncoder(**ism_kwargs)
            self.ism_video = ISMEncoder(**ism_kwargs)
        else:
            self.ism_text = self.ism_audio = self.ism_video = None

        # 3) 跨模态融合: 堆叠 num_layers 层 CoupledMamba3Fork 或 PairwiseCrossMamba3Fork
        mamba_class = PairwiseCrossMamba3Fork if use_pairwise_mamba else CoupledMamba3Fork
        self.layers = nn.ModuleList([
            mamba_class(
                d_model=d_model, d_state=d_state, expand=expand, headdim=headdim,
                ngroups=ngroups, rope_fraction=rope_fraction,
                is_mimo=is_mimo, mimo_rank=mimo_rank,
                chunk_size=chunk_size, is_outproj_norm=is_outproj_norm,
                v_self_ratio=float(v_self_ratio),
                use_gcmn_gate=use_gcmn_gate,
                device=device, dtype=dtype,
            )
            for _ in range(num_layers)
        ])

        # 4) 分类头
        # ★ Phase 17: 双向上下文 → 6*d_model; 否则 3*d_model
        fused_dim = 6 * d_model if self.bidirectional else 3 * d_model
        self.fusion_norm = nn.LayerNorm(fused_dim)
        self.head = nn.Sequential(
            nn.Linear(fused_dim, d_model, **factory_kwargs),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes, **factory_kwargs),
        )

        # 4.b) 模态级 sub_loss head (SIMS多任务用)
        # 仅当 multi_task=True 且 ism_depth > 0 且 task_type=regression 时启用
        self.use_sub_loss = (
            self.multi_task
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
    def _pool(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """x: (B, L, D) -> (B, D)"""
        if self.pool_type == "mean":
            if mask is not None:
                m = mask.float().unsqueeze(-1)
                denom = m.sum(dim=1).clamp(min=1.0)
                return (x * m).sum(dim=1) / denom
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
        ism_full_frame = getattr(self, '_ism_full_frame', False)
        # 0) 文本嵌入
        if self.use_bert and self.text_encoder is not None:
            text, _ = self.text_encoder(text)

        # 1) 投影到 d_model
        xt = self.proj_text(text)

        # ★ Phase 3: 音频编码分支 — 检测原始波形 vs 预提取特征
        if audio.dim() == 2 and audio.size(-1) > 500:
            # 原始波形 (B, T_wave), T_wave≈96000 → Data2Vec 在线编码
            if self.use_pretrained_audio and self.audio_encoder is not None:
                audio_hidden, _ = self.audio_encoder(audio)
                xa = self.proj_audio(audio_hidden)
            else:
                # 无编码器时取均值再投影 (退化路径)
                xa = self.proj_audio(audio.mean(dim=-1, keepdim=True).expand(-1, self.ism_seq_len))
        else:
            xa = self.proj_audio(audio)

        xv = self.proj_video(video)

        # ★ 方案 B: ISM 全帧处理 — 跳过对齐, 让 ISM 处理原始帧长
        if not ism_full_frame:
            # 2) 序列对齐 (旧行为: ISM 前池化到文本长度)
            Lt = xt.size(1)
            if xa.size(1) != Lt:
                xa = F.adaptive_avg_pool1d(xa.transpose(1, 2), Lt).transpose(1, 2)
            if xv.size(1) != Lt:
                xv = F.adaptive_avg_pool1d(xv.transpose(1, 2), Lt).transpose(1, 2)

        # 3) ISM (各模态独立)
        ism_cls_t = ism_cls_a = ism_cls_v = None
        if self.ism_depth > 0:
            if return_ism_cls:
                xt, ism_cls_t = self.ism_text(xt, return_cls=True)
                xv, ism_cls_v = self.ism_video(xv, return_cls=True)
                if self.skip_audio_ism:
                    ism_cls_a = xa.mean(dim=1)
                else:
                    xa, ism_cls_a = self.ism_audio(xa, return_cls=True)
            else:
                xt = self.ism_text(xt)
                xv = self.ism_video(xv)
                if not self.skip_audio_ism:
                    xa = self.ism_audio(xa)

        # ★ 方案 B: ISM 全帧后桥接池化 → CrossMamba 统一长度
        if ism_full_frame:
            Lt = xt.size(1)
            if xa.size(1) != Lt:
                xa = F.adaptive_avg_pool1d(xa.transpose(1, 2), Lt).transpose(1, 2)
            if xv.size(1) != Lt:
                xv = F.adaptive_avg_pool1d(xv.transpose(1, 2), Lt).transpose(1, 2)

        # 4) CoupledMamba3Fork (跨模态融合)
        out_a, out_v, out_l = xa, xv, xt
        for layer in self.layers:
            out_a, out_v, out_l = layer(out_a, out_v, out_l, cu_seqlens=cu_seqlens)

        if return_ism_cls:
            return out_l, out_a, out_v, ism_cls_t, ism_cls_a, ism_cls_v
        return out_l, out_a, out_v

    # ------------------------------------------------------------------
    # Phase 5+17: 对话上下文编码 (委托 context_fusion.encode_with_context)
    # ------------------------------------------------------------------
    def _encode_with_context(self, *args, **kwargs):
        """委托 context_fusion.encode_with_context, 传入 self 作为 model."""
        return encode_with_context(self, *args, **kwargs)

    # ------------------------------------------------------------------
    def forward(
        self,
        text: torch.Tensor,
        audio: torch.Tensor,
        video: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
        audio_lengths: Optional[torch.Tensor] = None,
        vision_lengths: Optional[torch.Tensor] = None,
        # ★ Phase 5: 上下文输入
        context_text: Optional[torch.Tensor] = None,
        context_audio: Optional[torch.Tensor] = None,
        context_video: Optional[torch.Tensor] = None,
    ):
        has_context = all(x is not None for x in [
            context_text, context_audio, context_video
        ])

        # ★ 方案 B: ISM 全帧标志 (被 _encode 通过 getattr 读取)
        if not hasattr(self, '_ism_full_frame'):
            self._ism_full_frame = False

        if has_context:
            # ★ 上下文路径 — Phase 17: 双向增强
            rev_l = rev_a = rev_v = None
            if self.use_sub_loss:
                if self.bidirectional:
                    out_l, out_a, out_v, rev_l, rev_a, rev_v, c_t, c_a, c_v = \
                        self._encode_with_context(
                            text, audio, video,
                            context_text, context_audio, context_video,
                            cu_seqlens, return_ism_cls=True,
                            bidirectional=True,
                        )
                else:
                    out_l, out_a, out_v, c_t, c_a, c_v = self._encode_with_context(
                        text, audio, video,
                        context_text, context_audio, context_video,
                        cu_seqlens, return_ism_cls=True,
                        bidirectional=False,
                    )
            else:
                if self.bidirectional:
                    out_l, out_a, out_v, rev_l, rev_a, rev_v = \
                        self._encode_with_context(
                            text, audio, video,
                            context_text, context_audio, context_video,
                            cu_seqlens, bidirectional=True,
                        )
                else:
                    out_l, out_a, out_v = self._encode_with_context(
                        text, audio, video,
                        context_text, context_audio, context_video,
                        cu_seqlens,
                        bidirectional=False,
                    )
                c_t = c_a = c_v = None
        else:
            # 回退: 原始单话语路径
            if self.use_sub_loss:
                out_l, out_a, out_v, c_t, c_a, c_v = self._encode(
                    text, audio, video, cu_seqlens, return_ism_cls=True,
                )
            else:
                out_l, out_a, out_v = self._encode(
                    text, audio, video, cu_seqlens,
                )
                c_t = c_a = c_v = None

        # 池化 + 拼接 + 分类头
        if has_context:
            # ★ _encode_with_context 已经做了池化, 直接使用
            pl, pa, pv = out_l, out_a, out_v
            if self.bidirectional and rev_l is not None:
                rl, ra, rv = rev_l, rev_a, rev_v
                fused = self.fusion_norm(torch.cat([pl, pa, pv, rl, ra, rv], dim=-1))
            else:
                fused = self.fusion_norm(torch.cat([pl, pa, pv], dim=-1))
        else:
            pl = self._pool(out_l)
            pa = self._pool(out_a)
            pv = self._pool(out_v)
            fused  = self.fusion_norm(torch.cat([pl, pa, pv], dim=-1))
        logits = self.head(fused)

        # 当未启用 sub_loss 时, 直接返回 Tensor
        if not self.use_sub_loss:
            return logits

        out: dict = {"logits": logits}
        if self.use_sub_loss:
            out["sub_T"] = self.sub_fc_T(c_t)
            out["sub_A"] = self.sub_fc_A(c_a)
            out["sub_V"] = self.sub_fc_V(c_v)
        return out

    # ------------------------------------------------------------------
    def get_modal_embeddings(
        self,
        text: torch.Tensor,
        audio: torch.Tensor,
        video: torch.Tensor,
        audio_lengths: Optional[torch.Tensor] = None,
        vision_lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        返回三模态的池化表征 (B, D), 供 InfoNCE 对比损失使用.
        """
        out_l, out_a, out_v = self._encode(text, audio, video)
        pl = self._pool(out_l)
        pa = self._pool(out_a)
        pv = self._pool(out_v)
        return pl, pa, pv