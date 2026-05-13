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
    """学习序列维度的注意力权重进行加权池化

    支持可选 padding mask: pad 位置不参与 softmax 竞争 (问题 ② 修复).
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1, bias=False),
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x:    (B, L, D)
            mask: (B, L) bool/byte, True/1 = 有效, False/0 = pad. None 则全有效.
        Returns:
            (B, D)
        """
        scores = self.attn(x)                          # (B, L, 1)
        if mask is not None:
            # 将 pad 位置的 score 置为 -inf, softmax 后变为 0
            neg_inf = torch.finfo(scores.dtype).min
            scores = scores.masked_fill(~mask.unsqueeze(-1).bool(), neg_inf)
            # 防御: 整行全 pad 时 softmax 会 NaN, 退化为均匀分布
            all_pad = (~mask.bool()).all(dim=1, keepdim=True)              # (B, 1)
            if all_pad.any():
                scores = torch.where(
                    all_pad.unsqueeze(-1).expand_as(scores),
                    torch.zeros_like(scores),
                    scores,
                )
        weights = torch.softmax(scores, dim=1)         # (B, L, 1)
        return (x * weights).sum(dim=1)                # (B, D)


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

    def forward(self, text_bert: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            hidden:         (B, L, 768)
            attention_mask: (B, L) long, 1=有效 0=pad
        """
        input_ids = text_bert[:, 0].long()
        if self.use_hf:
            attention_mask = text_bert[:, 1].long()
            token_type_ids = text_bert[:, 2].long()
            out = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            return out.last_hidden_state, attention_mask  # (B, L, 768), (B, L)
        # 回退路径: 直接由 input_ids != 0 推 mask
        attention_mask = (input_ids != 0).long()
        return self.bert(input_ids), attention_mask


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
        # --- 跨模态 V_self 注入 (问题 ③ 修复, 0=关闭即旧行为) ---
        v_self_ratio: float = 0.0,
        # --- 多任务开关 (SIMS/SIMS2 用于自动启用 sub_fc_T/A/V) ---
        multi_task: bool = False,
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
        self.multi_task = bool(multi_task)
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
                v_self_ratio=float(v_self_ratio),
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
        # 核心修改：双通道解耦 (Dual-Stream Disentanglement)。 
        # fused_dim 是 TAV 跨模态融合的特征集，太偏向 Text。我们在此基础上额外拼接原生的 Audio 和 Vision 特征，
        # 让 aux_head 直接感知无文本污染的情感强度。
        if self.aux_num_classes > 0 and task_type == "regression":
            # 引入额外的模态投影，专门供 aux_head 分辨类别
            self.aux_a_proj = nn.Linear(d_model, d_model // 2, **factory_kwargs)
            self.aux_v_proj = nn.Linear(d_model, d_model // 2, **factory_kwargs)
            # aux_head 的输入维度为: 融合层(fused_dim) + 原始音频(d_model//2) + 原始视频(d_model//2)
            aux_in_dim = fused_dim + (d_model // 2) * 2
            self.aux_head = nn.Sequential(
                nn.Linear(aux_in_dim, d_model, **factory_kwargs),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, self.aux_num_classes, **factory_kwargs),
            )
        else:
            self.aux_head = None

        # 5.c) 模态级 sub_loss head (对齐 MSAmba sub_fc_T/A/V)
        # 各模态 ISM 输出的 CLS token (D 维) → 1 维回归
        # 仅当 sub_loss_lambda > 0 / multi_task=True 且 ism_depth > 0 且 task_type=regression 时启用
        self.use_sub_loss = (
            (self.sub_loss_lambda > 0.0 or self.multi_task)
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
    @staticmethod
    def _build_mask_from_lengths(lengths: torch.Tensor, L: int) -> torch.Tensor:
        """根据真实长度构造 padding mask.

        Args:
            lengths: (B,) long, 真实长度 (来自 audio_lengths/vision_lengths,
                     可能 > L, 内部会 clamp 到 L)
            L:       目标序列长度 (一般为投影/对齐后的统一长度)
        Returns:
            mask: (B, L) bool, True=有效, False=pad
        """
        L_eff = torch.clamp(lengths, max=L)
        idx = torch.arange(L, device=lengths.device).unsqueeze(0)   # (1, L)
        return idx < L_eff.unsqueeze(1)                              # (B, L)

    def _pool(self, x: torch.Tensor, pool_module=None,
              mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """x: (B, L, D) -> (B, D)
        mask: (B, L) bool, True=有效, False=pad. None 则不 mask.
        """
        if self.pool_type == "attention" and pool_module is not None:
            return pool_module(x, mask=mask)
        elif self.pool_type == "mean":
            if mask is not None:
                m = mask.float().unsqueeze(-1)                       # (B, L, 1)
                denom = m.sum(dim=1).clamp(min=1.0)                  # (B, 1)
                return (x * m).sum(dim=1) / denom
            return x.mean(dim=1)
        elif self.pool_type == "last":
            if mask is not None:
                valid = mask.bool()
                lengths = valid.sum(dim=1).clamp(min=1)
                idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, x.size(-1))
                pooled = x.gather(dim=1, index=idx).squeeze(1)
                if (~valid).all(dim=1).any():
                    pooled = torch.where((~valid).all(dim=1, keepdim=True), torch.zeros_like(pooled), pooled)
                return pooled
            return x[:, -1]
        elif self.pool_type == "cls":
            if mask is not None:
                valid = mask.bool()
                first_idx = valid.float().argmax(dim=1)
                idx = first_idx.view(-1, 1, 1).expand(-1, 1, x.size(-1))
                pooled = x.gather(dim=1, index=idx).squeeze(1)
                if (~valid).all(dim=1).any():
                    pooled = torch.where((~valid).all(dim=1, keepdim=True), torch.zeros_like(pooled), pooled)
                return pooled
            return x[:, 0]
        else:
            return x.mean(dim=1)

    # ------------------------------------------------------------------
    @staticmethod
    def _apply_mask(x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        """zero-out pad 位置 (P0 修复, 防止 LayerNorm/GELU 把 0 变成非 0 后污染 ISM/Fork).
        x:    (B, L, D)
        mask: (B, L) bool, True=有效. None 时不做处理.
        """
        if mask is None:
            return x
        return x * mask.unsqueeze(-1).to(x.dtype)

    def _encode(self, text, audio, video, cu_seqlens=None, return_ism_cls: bool = False,
                audio_lengths: Optional[torch.Tensor] = None,
                vision_lengths: Optional[torch.Tensor] = None):
        """编码到融合后的三模态序列表征 (共享流水线)

        端到端 padding 感知 (P0 完整修复):
            1) 在 MLP 投影后立刻 zero-out pad (防止 bias/GELU 把 0 变成非 0)
            2) ISM 接收 mask, 内部每个 Block 前后都 zero-out
            3) Fork 之前再 zero-out 一次, Fork 输出后再 zero-out 一次
            4) Pool 阶段也用 mask
        审计点: ISM/Fork 的内部状态扫描在数学上仍会被 pad 步影响,
            但通过多处 zero-out 已把 hidden 的污染降到一阶以下.

        Args:
            return_ism_cls: True 时同时返回 ISM 阶段的 cls_T/A/V (sub_loss 用)
            audio_lengths/vision_lengths: 原始长度 (基于 truncate 后维度);
                若提供, 则在全链路用于构造 padding mask.
        Returns (基础): (out_l, out_a, out_v, mask_t, mask_a, mask_v)
                        若 return_ism_cls=True 再附 (cls_t, cls_a, cls_v)
            mask_*: (B, L) bool 或 None
        """
        # 0) 文本嵌入
        text_mask = None  # (B, L_t) BERT 输出 token 级 mask
        if self.use_bert and self.text_encoder is not None:
            text, text_mask = self.text_encoder(text)

        # 1) 投影到 d_model (2层 MLP)
        xt = self.proj_text(text)
        xa = self.proj_audio(audio)
        xv = self.proj_video(video)

        # 2) 序列对齐
        # 真实数据流 (MOSI unaligned_50 + _truncate([50,50,50])):
        #   dataset/_truncate 已将 audio/vision 的长度维度硬截/补到 Lt=50,
        #   所以模型收到的 xa/xv 本来就是 (B, 50, D), L_a_in == L_v_in == Lt,
        #   下面的 pool 分支通常不进入 (仅在某些数据集 L_in != Lt 时作为兜底).
        Lt = xt.size(1)
        L_a_in = xa.size(1)
        L_v_in = xv.size(1)
        if L_a_in != Lt:
            xa = F.adaptive_avg_pool1d(xa.transpose(1, 2), Lt).transpose(1, 2)
        if L_v_in != Lt:
            xv = F.adaptive_avg_pool1d(xv.transpose(1, 2), Lt).transpose(1, 2)

        # 2.b) 构造三模态 padding mask (B, Lt)
        # text:        BERT attention_mask
        # audio/vision: 若发生过 pool (L_in != Lt), 用比例缩放避免边界误差 (P2 修复);
        #               否则直接用 _build_mask_from_lengths (主路径).
        mask_t = text_mask.bool() if text_mask is not None else None
        mask_a = self._make_av_mask(audio_lengths, L_a_in, Lt) if audio_lengths is not None else None
        mask_v = self._make_av_mask(vision_lengths, L_v_in, Lt) if vision_lengths is not None else None

        # 2.c) 投影后立刻 zero-out pad (P0 关键)
        # MLP 末尾的 LayerNorm + GELU + bias 会把 0 输入变成非 0, 必须强制清零,
        # 否则 ISM/Fork 会把"虚假的 pad 表征"卷入有效区状态.
        xt = self._apply_mask(xt, mask_t)
        xa = self._apply_mask(xa, mask_a)
        xv = self._apply_mask(xv, mask_v)

        # 3) ISM (各模态独立, 内部每层前后都 zero-out)
        ism_cls_t = ism_cls_a = ism_cls_v = None
        if self.ism_depth > 0:
            if return_ism_cls:
                xt, ism_cls_t = self.ism_text(xt,  mask=mask_t, return_cls=True)
                xa, ism_cls_a = self.ism_audio(xa, mask=mask_a, return_cls=True)
                xv, ism_cls_v = self.ism_video(xv, mask=mask_v, return_cls=True)
            else:
                xt = self.ism_text(xt,  mask=mask_t)
                xa = self.ism_audio(xa, mask=mask_a)
                xv = self.ism_video(xv, mask=mask_v)
            # ISM 输出再 zero-out 一次 (双保险, 即使 ISM 内部漏 mask 也不会传到 Fork)
            xt = self._apply_mask(xt, mask_t)
            xa = self._apply_mask(xa, mask_a)
            xv = self._apply_mask(xv, mask_v)

        # 4) CoupledMamba3Fork (跨模态融合, 输入前 + 每层后 zero-out)
        out_a, out_v, out_l = xa, xv, xt
        for layer in self.layers:
            out_a, out_v, out_l = layer(out_a, out_v, out_l, cu_seqlens=cu_seqlens)
            # 每层后 zero-out 三模态, 防止 pad 经过 SSM 状态扫描积累
            out_a = self._apply_mask(out_a, mask_a)
            out_v = self._apply_mask(out_v, mask_v)
            out_l = self._apply_mask(out_l, mask_t)

        if return_ism_cls:
            return out_l, out_a, out_v, mask_t, mask_a, mask_v, ism_cls_t, ism_cls_a, ism_cls_v
        return out_l, out_a, out_v, mask_t, mask_a, mask_v

    # ------------------------------------------------------------------
    @staticmethod
    def _make_av_mask(lengths: torch.Tensor, L_in: int, L_out: int) -> torch.Tensor:
        """构造 audio/vision 在对齐后长度上的 mask (B, L_out).

        分两种情况 (P2 修复跨数据集边界):
          - L_in == L_out (主路径, MOSI 当前情况):
              直接 clamp(lengths, max=L_out) 构造 hard mask, 与 pad 区严格对应
          - L_in != L_out (兜底, 走过 adaptive_avg_pool1d 的数据集):
              按比例缩放 effective_len' = ceil(min(lengths, L_in) * L_out / L_in),
              避免硬阈值在 pool 后的边界误差
        """
        if L_in == L_out:
            return MSAClassifier._build_mask_from_lengths(lengths, L_out)
        # 比例缩放路径
        eff = torch.clamp(lengths.long(), min=0, max=L_in)
        scaled = torch.ceil(eff.float() * L_out / max(L_in, 1)).long()
        scaled = torch.clamp(scaled, min=0, max=L_out)
        idx = torch.arange(L_out, device=lengths.device).unsqueeze(0)
        return idx < scaled.unsqueeze(1)

    # ------------------------------------------------------------------
    def forward(
        self,
        text: torch.Tensor,
        audio: torch.Tensor,
        video: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
        audio_lengths: Optional[torch.Tensor] = None,
        vision_lengths: Optional[torch.Tensor] = None,
    ):
        need_cls = self.use_sub_loss or (self.aux_head is not None)
        if need_cls:
            (out_l, out_a, out_v,
             mask_t, mask_a, mask_v,
             c_t, c_a, c_v) = self._encode(
                text, audio, video, cu_seqlens, return_ism_cls=True,
                audio_lengths=audio_lengths, vision_lengths=vision_lengths,
            )
        else:
            out_l, out_a, out_v, mask_t, mask_a, mask_v = self._encode(
                text, audio, video, cu_seqlens,
                audio_lengths=audio_lengths, vision_lengths=vision_lengths,
            )
            c_t = c_a = c_v = None

        # 池化 + 拼接 + 分类头
        pl = self._pool(out_l, self.pool_text  if self.pool_type == "attention" else None, mask=mask_t)
        pa = self._pool(out_a, self.pool_audio if self.pool_type == "attention" else None, mask=mask_a)
        pv = self._pool(out_v, self.pool_video if self.pool_type == "attention" else None, mask=mask_v)
        fused  = self.fusion_norm(torch.cat([pl, pa, pv], dim=-1))
        logits = self.head(fused)

        # 当未启用任何辅助 head 时, 保持向后兼容: 直接返回 Tensor
        has_aux = self.aux_head is not None
        if not has_aux and not self.use_sub_loss:
            return logits

        out: dict = {"logits": logits}
        if has_aux:
            # 双通解耦：将独立提纯的 Audio 和 Video 加入 aux_head，绕开文本污染
            pure_a = self.aux_a_proj(c_a) # c_a 是 (B, D_model), Mamba 之前的纯音视特征
            pure_v = self.aux_v_proj(c_v) 
            aux_feat = torch.cat([fused, pure_a, pure_v], dim=-1)
            out["aux_logits"] = self.aux_head(aux_feat)
            
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
        audio_lengths: Optional[torch.Tensor] = None,
        vision_lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        返回三模态的池化表征 (B, D), 供 InfoNCE 对比损失使用.
        """
        out_l, out_a, out_v, mask_t, mask_a, mask_v = self._encode(
            text, audio, video,
            audio_lengths=audio_lengths, vision_lengths=vision_lengths,
        )
        pl = self._pool(out_l, self.pool_text  if self.pool_type == "attention" else None, mask=mask_t)
        pa = self._pool(out_a, self.pool_audio if self.pool_type == "attention" else None, mask=mask_a)
        pv = self._pool(out_v, self.pool_video if self.pool_type == "attention" else None, mask=mask_v)
        return pl, pa, pv