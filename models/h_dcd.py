"""
H-DCD - Hierarchical Decoupled Contrastive Distillation
分层解耦对比蒸馏模型（主模型）

整合所有子模块：
1. Feature Projection (TextProjection + AudioVideoProjection)
2. Decouple Encoder (模态解耦)
3. HMNF (异构多模态融合流 - CoupledHMNF)
4. HMPN (同构感知网络流 - HMPN)
5. Hierarchical Classifiers (分层分类器)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import sys
import os

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(__file__))

# 导入子模块
from feature_projection import TextProjection, AudioVideoProjection
from decouple_encoder import DecoupleEncoder
from models.hmnf import CoupledHMNF
from models.hmpn import HMPN


class H_DCD(nn.Module):
    """
    H-DCD 主模型
    Hierarchical Decoupled Contrastive Distillation
    
    完整架构整合：
    - 特征投影层
    - 解耦编码器
    - 双流融合网络 (HMNF + HMPN)
    - 分层分类器
    """
    
    def __init__(
        self,
        # 输入维度
        text_input_dim: int = 768,      # BERT
        audio_input_dim: int = 74,      # ComParE
        video_input_dim: int = 35,      # DenseFace
        
        # 统一维度
        d_model: int = 128,
        
        # 序列长度（用于BiGRU等）
        text_hidden_dim: int = 256,
        text_num_layers: int = 2,
        
        # Decouple Encoder 参数
        decouple_disc_hidden: int = 256,
        decouple_lambda_grl: float = 1.0,
        
        # HMNF 参数 (CoupledHMNF)
        hmnf_d_state: int = 64,
        hmnf_d_conv: int = 4,
        hmnf_expand: int = 2,
        hmnf_num_layers: int = 1,
        
        # HMPN 参数
        hmpn_d_state: int = 64,
        hmpn_d_conv: int = 4,
        hmpn_expand: int = 2,
        hmpn_num_heads: int = 4,
        
        # 分类
        num_classes: int = 4,           # 情感类别数
        dropout: float = 0.1,
    ):
        super(H_DCD, self).__init__()
        
        self.d_model = d_model
        self.num_classes = num_classes
        
        # ====================================================================
        # 1. 特征投影层 (Feature Projection)
        # ====================================================================
        self.text_projection = TextProjection(
            input_dim=text_input_dim,
            hidden_dim=text_hidden_dim,
            output_dim=d_model,
            num_layers=text_num_layers,
            dropout=dropout
        )
        
        self.audio_projection = AudioVideoProjection(
            input_dim=audio_input_dim,
            output_dim=d_model,
            dropout=dropout
        )
        
        self.video_projection = AudioVideoProjection(
            input_dim=video_input_dim,
            output_dim=d_model,
            dropout=dropout
        )
        
        # ====================================================================
        # 2. 解耦编码器 (Decouple Encoder)
        # ====================================================================
        self.decouple_encoder = DecoupleEncoder(
            d_model=d_model,
            num_modalities=3,
            disc_hidden_dim=decouple_disc_hidden,
            dropout=dropout,
            lambda_grl=decouple_lambda_grl
        )
        
        # ====================================================================
        # 3. 双流融合网络
        # ====================================================================
        # 流A: HMNF (异构多模态融合 - CoupledHMNF)
        # 使用 CoupledHMNF 替代旧的 HMNF
        self.hmnf = CoupledHMNF(
            d_model=d_model,
            num_layers=hmnf_num_layers,
            d_state=hmnf_d_state,
            d_conv=hmnf_d_conv,
            expand=hmnf_expand,
            dropout=dropout
        )
        
        # HMNF 融合层 (将 CoupledHMNF 的三个输出融合为一个向量)
        self.hmnf_fusion = nn.Sequential(
            nn.Linear(3 * d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model)
        )
        
        # 流B: HMPN (同构感知网络)
        self.hmpn = HMPN(
            d_model=d_model,
            d_state=hmpn_d_state,
            d_conv=hmpn_d_conv,
            expand=hmpn_expand,
            headdim=32,
            ngroups=1,
            num_heads=hmpn_num_heads
        )
        
        # ====================================================================
        # 4. 分层分类器 (Hierarchical Classifiers)
        # ====================================================================
        # 池化层（用于序列特征 → 向量）
        self.pooling = nn.AdaptiveAvgPool1d(1)
        
        # 4.1 单模态分类头 (使用私有特征或共享特征)
        self.head_uni_text = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
        self.head_uni_audio = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
        self.head_uni_video = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
        # 4.2 双模态分类头 (使用 HMNF 中间态)
        self.head_bi_ta = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
        self.head_bi_tv = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
        self.head_bi_av = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
        # 4.3 全模态分类头 (融合 HMNF 和 HMPN)
        # 方式1: 拼接
        self.fusion_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Tanh()
        )
        
        self.head_multi = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
    def forward(
        self,
        x_text: torch.Tensor,    # [B, L_t, text_input_dim]
        x_audio: torch.Tensor,   # [B, L_a, audio_input_dim]
        x_video: torch.Tensor,   # [B, L_v, video_input_dim]
        return_all: bool = True  # 是否返回所有中间结果（训练时需要）
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            x_text: 文本特征 [B, L_t, 768]
            x_audio: 音频特征 [B, L_a, 74]
            x_video: 视频特征 [B, L_v, 35]
            return_all: 是否返回所有中间结果（用于计算多个Loss）
            
        Returns:
            字典包含所有输出
        """
        batch_size = x_text.size(0)
        
        # ====================================================================
        # 步骤1: 特征投影 (Feature Projection)
        # ====================================================================
        X_t = self.text_projection(x_text)    # [B, L_t, d_model]
        X_a = self.audio_projection(x_audio)  # [B, L_a, d_model]
        X_v = self.video_projection(x_video)  # [B, L_v, d_model]
        
        # ====================================================================
        # 步骤2: 解耦编码 (Decouple Encoder)
        # ====================================================================
        decouple_outputs = self.decouple_encoder(
            X_t, X_a, X_v, 
            return_disc=self.training  # 训练时返回鉴别器输出
        )
        
        # 提取私有特征和共享特征
        s_text = decouple_outputs['s_text']    # [B, L_t, d_model]
        s_audio = decouple_outputs['s_audio']  # [B, L_a, d_model]
        s_video = decouple_outputs['s_video']  # [B, L_v, d_model]
        
        c_text = decouple_outputs['c_text']    # [B, L_t, d_model]
        c_audio = decouple_outputs['c_audio']  # [B, L_a, d_model]
        c_video = decouple_outputs['c_video']  # [B, L_v, d_model]
        
        # ====================================================================
        # 步骤3: 双流并行处理
        # ====================================================================
        
        # --- 序列对齐 (Sequence Alignment) ---
        # HMNF 和 HMPN 可能需要对齐的序列长度，或者内部处理
        # 这里我们假设输入长度可能不同，先进行简单的对齐（取最小长度）
        # 注意：实际应用中可能需要更复杂的对齐策略
        min_len = min(c_text.size(1), c_audio.size(1), c_video.size(1))
        
        def align_seq(x, length):
            if x.size(1) == length:
                return x
            return F.adaptive_avg_pool1d(x.transpose(1, 2), length).transpose(1, 2)
            
        c_text_aligned = align_seq(c_text, min_len)
        c_audio_aligned = align_seq(c_audio, min_len)
        c_video_aligned = align_seq(c_video, min_len)
        
        # --- 流A: HMNF (异构融合流) ---
        # CoupledHMNF 输入: (x_a, x_v, x_l)
        hmnf_out_a, hmnf_out_v, hmnf_out_l = self.hmnf(
            c_audio_aligned, c_video_aligned, c_text_aligned
        )
        
        # 池化
        hmnf_pool_a = torch.mean(hmnf_out_a, dim=1)
        hmnf_pool_v = torch.mean(hmnf_out_v, dim=1)
        hmnf_pool_l = torch.mean(hmnf_out_l, dim=1)
        
        # 融合生成 h_mnf
        hmnf_fused_feat = self.hmnf_fusion(
            torch.cat([hmnf_pool_a, hmnf_pool_v, hmnf_pool_l], dim=-1)
        )
        
        # --- 流B: HMPN (同构感知流) ---
        # HMPN 输入: (h_t, h_a, h_v)
        hmpn_final = self.hmpn(
            c_text_aligned, c_audio_aligned, c_video_aligned
        )
        
        # ====================================================================
        # 步骤4: 分层分类
        # ====================================================================
        # 4.1 单模态分类 (使用共享特征 c_x)
        c_text_pooled = self.pooling(c_text.transpose(1, 2)).squeeze(-1)
        c_audio_pooled = self.pooling(c_audio.transpose(1, 2)).squeeze(-1)
        c_video_pooled = self.pooling(c_video.transpose(1, 2)).squeeze(-1)
        
        pred_uni_text = self.head_uni_text(c_text_pooled)
        pred_uni_audio = self.head_uni_audio(c_audio_pooled)
        pred_uni_video = self.head_uni_video(c_video_pooled)
        
        # 4.2 双模态分类 (使用 HMNF 输出特征)
        # 使用 HMNF 处理后的特征进行双模态分类
        pred_bi_ta = self.head_bi_ta(hmnf_pool_l + hmnf_pool_a)
        pred_bi_tv = self.head_bi_tv(hmnf_pool_l + hmnf_pool_v)
        pred_bi_av = self.head_bi_av(hmnf_pool_a + hmnf_pool_v)
        
        # 4.3 全模态分类 (融合 HMNF 和 HMPN)
        concat_feat = torch.cat([hmnf_fused_feat, hmpn_final], dim=-1)  # [B, 2*d_model]
        fused_final = self.fusion_gate(concat_feat)  # [B, d_model]
        pred_multi = self.head_multi(fused_final)    # [B, num_classes]
        
        # ====================================================================
        # 步骤5: 组织输出
        # ====================================================================
        outputs = {
            # 分类输出
            'logits_uni': [pred_uni_text, pred_uni_audio, pred_uni_video],
            'logits_bi': [pred_bi_ta, pred_bi_tv, pred_bi_av],
            'logits_multi': pred_multi,
            
            # 对比蒸馏特征 (HMNF vs HMPN)
            'features_contrast': {
                'hmnf': hmnf_fused_feat,  # [B, d_model]
                'hmpn': hmpn_final         # [B, d_model]
            },
        }
        
        # 训练时返回更多信息用于计算Loss
        if return_all:
            outputs.update({
                # 解耦相关项 (用于重构Loss和对抗Loss)
                # Flatten structure for compatibility with losses.py
                'decouple_items': {
                    's_text': s_text,
                    's_audio': s_audio,
                    's_video': s_video,
                    'c_text': c_text,
                    'c_audio': c_audio,
                    'c_video': c_video,
                    'recon_text': decouple_outputs['recon_text'],
                    'recon_audio': decouple_outputs['recon_audio'],
                    'recon_video': decouple_outputs['recon_video'],
                    'original_text': X_t,
                    'original_audio': X_a,
                    'original_video': X_v
                },
                
                # 对抗鉴别器输出
                'adv_logits': decouple_outputs.get('disc_logits', None),
            })
        
        return outputs


def test_h_dcd():
    """测试 H-DCD 主模型"""
    print("=" * 80)
    print("H-DCD 主模型测试")
    print("=" * 80)
    
    # 模拟输入
    batch_size = 8
    L_t, L_a, L_v = 50, 100, 75
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n设备: {device}")
    
    # 创建模型
    model = H_DCD(
        text_input_dim=768,
        audio_input_dim=74,
        video_input_dim=35,
        d_model=128,
        num_classes=4,
    ).to(device)
    
    # 参数统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n模型参数:")
    print(f"  总参数: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"  可训练参数: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
    
    # 创建测试输入
    x_text = torch.randn(batch_size, L_t, 768, device=device)
    x_audio = torch.randn(batch_size, L_a, 74, device=device)
    x_video = torch.randn(batch_size, L_v, 35, device=device)
    
    print(f"\n输入形状:")
    print(f"  Text: {x_text.shape}")
    print(f"  Audio: {x_audio.shape}")
    print(f"  Video: {x_video.shape}")
    
    # 前向传播
    model.eval()
    with torch.no_grad():
        outputs = model(x_text, x_audio, x_video, return_all=True)
    
    print(f"\n输出:")
    print(f"  单模态分类 (logits_uni):")
    for i, name in enumerate(['Text', 'Audio', 'Video']):
        print(f"    {name}: {outputs['logits_uni'][i].shape}")
    
    print(f"  双模态分类 (logits_bi):")
    for i, name in enumerate(['TA', 'TV', 'AV']):
        print(f"    {name}: {outputs['logits_bi'][i].shape}")
    
    print(f"  全模态分类 (logits_multi): {outputs['logits_multi'].shape}")
    
    print(f"\n对比蒸馏特征:")
    print(f"  HMNF: {outputs['features_contrast']['hmnf'].shape}")
    print(f"  HMPN: {outputs['features_contrast']['hmpn'].shape}")
    
    print("\n✓ H-DCD 测试成功！")
    print("=" * 80)
    
    return model


if __name__ == "__main__":
    test_h_dcd()
