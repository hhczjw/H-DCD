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

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from layers.ism import ISMEncoder


# ============================================================
# 改造: 通用文本预训练编码器 (支持 BERT/RoBERTa/DeBERTa 等)
# 替换原有的 BertTextEncoder 类
# ============================================================

class TextPretrainedEncoder(nn.Module):
    """
    通用文本预训练编码器, 支持 BERT / RoBERTa / DeBERTa 等 HuggingFace 模型.
    
    与原有 BertTextEncoder 的差异:
    1. 支持 RoBERTa (无 token_type_ids, 使用 <s>/</s> 分隔符)
    2. 支持 DeBERTa (相对位置编码)
    3. 自动检测模型类型, 适配不同的输入格式
    4. ★ CAGMamba 对齐: 当输入是 BERT token ID 但模型非 BERT 时, 自动解码→重编码
       (decode BERT ids → raw text → re-encode with current tokenizer)
    """

    # 预定义模型配置映射
    SUPPORTED_MODELS = {
        "bert-base-uncased":    {"dim": 768, "type": "bert"},
        "bert-large-uncased":   {"dim": 1024, "type": "bert"},
        "roberta-base":         {"dim": 768, "type": "roberta"},
        "roberta-large":        {"dim": 1024, "type": "roberta"},
        "microsoft/deberta-base": {"dim": 768, "type": "deberta"},
    }

    def __init__(
        self,
        pretrained: str = "roberta-base",      # ★ 默认改为 RoBERTa-base
        finetune: bool = True,
        strict: bool = True,
    ):
        super().__init__()
        self.pretrained_name = pretrained
        self.use_hf = False
        
        # 自动检测模型类型
        self.model_type = "bert"  # fallback
        for key, info in self.SUPPORTED_MODELS.items():
            if key in pretrained.lower():
                self.model_type = info["type"]
                self.out_dim = info["dim"]
                break
        
        # ★ 安全默认值: 防止 SUPPORTED_MODELS 未匹配时 out_dim 未定义
        if not hasattr(self, 'out_dim'):
            self.out_dim = 768
        
        try:
            from transformers import AutoModel, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(pretrained)
            self.transformer = AutoModel.from_pretrained(pretrained)
            self.out_dim = self.transformer.config.hidden_size
            self.use_hf = True
        except Exception as e:
            if strict:
                raise RuntimeError(f"加载 {pretrained} 失败: {e}")
            print(f"[WARN] 回退到 Embedding: {e}")
            self.transformer = nn.Embedding(50265, self.out_dim, padding_idx=1)
            self.tokenizer = None

        # ★ CAGMamba 对齐: 当模型非 BERT 时, 预加载 BERT tokenizer 用于解码旧的 .pkl 数据
        self._bert_tokenizer = None
        if self.model_type != "bert" and self.use_hf:
            try:
                from transformers import BertTokenizer
                self._bert_tokenizer = BertTokenizer.from_pretrained(
                    "bert-base-uncased"
                )
            except Exception:
                # 静默失败: 回退到旧行为 (依赖上游传入正确 token ID)
                pass

        if self.use_hf and not finetune:
            for p in self.transformer.parameters():
                p.requires_grad = False

    def forward(
        self, text_input: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            text_input: 
                - 离线三通道模式 (兼容旧 BERT .pkl): (B, 3, L)
                  channels -> [input_ids, attention_mask, token_type_ids]
                  当模型非 BERT 时, 自动用 _bert_tokenizer 解码→重编码
                - 在线模式: (B, L) 仅 input_ids
        Returns:
            hidden:         (B, L, out_dim)
            attention_mask: (B, L) long
        """
        # ==== 检测输入格式 ====
        if text_input.dim() == 3 and text_input.size(1) == 3:
            # ─── 旧 BERT 三通道格式: [ids, mask, segment] ───
            input_ids = text_input[:, 0].long()
            attention_mask = text_input[:, 1].long()
            old_max_len = text_input.size(2)

            # ★ CAGMamba 对齐: BERT token ID → 模型 token ID 的自动转换
            if self._bert_tokenizer is not None and self.model_type != "bert":
                # 1) 用 BERT tokenizer 解码回原始文本
                raw_texts = self._bert_tokenizer.batch_decode(
                    input_ids, skip_special_tokens=True
                )
                # 2) 用当前模型 tokenizer 重新编码 (对齐 max_length)
                encoded = self.tokenizer(
                    raw_texts,
                    padding="max_length",
                    truncation=True,
                    max_length=old_max_len,
                    return_tensors="pt",
                )
                input_ids = encoded["input_ids"].to(text_input.device)
                attention_mask = encoded["attention_mask"].to(text_input.device)

        elif text_input.dim() == 2:
            # ─── 纯 token ID 模式 ───
            input_ids = text_input.long()
            attention_mask = (
                input_ids != self.tokenizer.pad_token_id
            ).long() if self.tokenizer else (input_ids != 0).long()
        else:
            input_ids = text_input.squeeze(1).long()
            attention_mask = (input_ids != 0).long()

        if not self.use_hf:
            return self.transformer(input_ids), attention_mask

        # 根据模型类型适配参数
        kw = {"input_ids": input_ids, "attention_mask": attention_mask}
        # RoBERTa / DeBERTa 不需要 token_type_ids
        # BERT 的 token_type_ids 已在重编码时被正确生成 (或在上游忽略)
        
        out = self.transformer(**kw)
        return out.last_hidden_state, attention_mask


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
        device=None,
        dtype=None,
    ):
        super().__init__()
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
            self.text_encoder = TextPretrainedEncoder(
            pretrained=bert_pretrained,     # ★ 默认 "roberta-base"
            finetune=bert_finetune,
            )
            text_feat_dim = self.text_encoder.out_dim  # ★ 自动获取 (768 或 1024)
        else:
            self.text_encoder = None
            text_feat_dim = text_input_dim

        # 1) 输入投影 — 单线性层
        self.proj_text  = nn.Linear(text_feat_dim, d_model, **factory_kwargs)
        self.proj_audio = nn.Linear(audio_input_dim, d_model, **factory_kwargs)
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

        # 4) 分类头
        fused_dim = 3 * d_model
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
        Returns (基础): (out_l, out_a, out_v)
                        若 return_ism_cls=True 再附 (cls_t, cls_a, cls_v)
        """
        # 0) 文本嵌入
        if self.use_bert and self.text_encoder is not None:
            text, _ = self.text_encoder(text)

        # 1) 投影到 d_model
        xt = self.proj_text(text)
        xa = self.proj_audio(audio)
        xv = self.proj_video(video)

        # 2) 序列对齐
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
                xa, ism_cls_a = self.ism_audio(xa, return_cls=True)
                xv, ism_cls_v = self.ism_video(xv, return_cls=True)
            else:
                xt = self.ism_text(xt)
                xa = self.ism_audio(xa)
                xv = self.ism_video(xv)

        # 4) CoupledMamba3Fork (跨模态融合)
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
        audio_lengths: Optional[torch.Tensor] = None,
        vision_lengths: Optional[torch.Tensor] = None,
    ):
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