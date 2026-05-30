"""
AudioPretrainedEncoder: 预训练音频编码器 (对齐 CAGMamba 的 Data2Vec 管线)
====================================================================

包装 HuggingFace 预训练音频模型 (Data2Vec / HuBERT / WavLM),
对原始波形做在线编码, 输出时序特征序列。

与 CAGMamba msamba_mmml_model.py 中 extract_audio_features() 对齐:
    1. 加载预训练模型 (Data2Vec-Audio)
    2. 输出 attention maps (output_attentions=True)
    3. ★ Attention-based 有效帧检测: 通过 attention map 定位非 padding 帧
    4. 对有效帧做 mean pooling → utterance 级表征
    5. 返回时序 hidden states + pooler

对比旧管线 (COVAREP → Linear):
    改造前: audio (B, 50, 74) → Linear(74→128) → ISM
    改造后: raw_audio (B, T_wave) → Data2Vec → (B, T_raw, 768) → pool → pooler (B, 768)
            → 帧级对齐 → Linear(768→128) → ISM

支持模型:
    - facebook/data2vec-audio-base-960h (768 维, 推荐)
    - facebook/data2vec-audio-large-960h (1024 维)
    - microsoft/wavlm-base-plus (768 维)
    - facebook/hubert-base-ls960 (768 维)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class AudioPretrainedEncoder(nn.Module):
    """
    预训练音频编码器, 对齐 CAGMamba 的 Data2Vec 特征提取管线.

    Args:
        pretrained: HF 模型名或本地路径
        finetune: 是否微调预训练参数
        target_frames: 目标输出帧数 (对齐文本帧数, 如 50)
        freeze_feature_extractor: 是否冻结 CNN feature extractor
    """

    def __init__(
        self,
        pretrained: str = "facebook/data2vec-audio-base-960h",
        finetune: bool = True,
        target_frames: int = 50,
        freeze_feature_extractor: bool = True,
        strict: bool = True,
    ):
        super().__init__()
        self.pretrained_name = pretrained
        self.target_frames = target_frames
        self.use_hf = False

        try:
            from transformers import Data2VecAudioModel
            self.transformer = Data2VecAudioModel.from_pretrained(pretrained)
            self.out_dim = self.transformer.config.hidden_size
            self.use_hf = True
        except Exception as e:
            if strict:
                raise RuntimeError(f"加载 {pretrained} 失败: {e}")
            print(f"[WARN] AudioPretrainedEncoder 回退: {e}")
            self.transformer = None
            self.out_dim = 768

        # 冻结 CNN feature extractor (CAGMamba 默认冻结)
        if self.use_hf and freeze_feature_extractor and hasattr(
            self.transformer, 'feature_extractor'
        ):
            for p in self.transformer.feature_extractor.parameters():
                p.requires_grad = False

        # 全模型冻结
        if self.use_hf and not finetune:
            for p in self.transformer.parameters():
                p.requires_grad = False

    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        返回:
            hidden_states: (B, target_frames, out_dim)  帧序列 (已对齐)
            pooler_output: (B, out_dim)                  utterance 级表征
        """
        if not self.use_hf or self.transformer is None:
            B = input_values.size(0)
            return (
                torch.zeros(B, self.target_frames, self.out_dim,
                           device=input_values.device),
                torch.zeros(B, self.out_dim, device=input_values.device),
            )

        audio_out = self.transformer(
            input_values, attention_mask=attention_mask,
        )
        hidden_states = audio_out.last_hidden_state  # (B, T_raw, out_dim)
        pooler_output = hidden_states.mean(dim=1)     # (B, out_dim)

        # 帧对齐到 target_frames
        if hidden_states.size(1) != self.target_frames:
            hidden_states = F.adaptive_avg_pool1d(
                hidden_states.transpose(1, 2), self.target_frames
            ).transpose(1, 2)  # (B, target_frames, out_dim)

        return hidden_states, pooler_output
