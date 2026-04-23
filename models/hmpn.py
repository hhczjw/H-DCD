"""
HMPN: Hierarchical Mamba Perception Network
分层 Mamba 感知网络

参考架构: MHFE (Multimodal Hierarchical Fusion with Enhanced attention)
核心思想:
    1. 使用 Mamba2 进行单模态精炼
    2. 跨模态交互增强（类似 MHFE 的 cross-modal attention）
    3. 超模态融合（hyper-fusion）
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

# ============================================================================
# 导入 Mamba2
# ============================================================================
try:
    from mamba_ssm import Mamba2
    MAMBA2_AVAILABLE = True
except ImportError:
    MAMBA2_AVAILABLE = False


# ============================================================================
# 使用公共 RMSNorm
# ============================================================================
from common import RMSNorm


# ============================================================================
# MambaBlock: 单模态精炼
# ============================================================================
class MambaBlock(nn.Module):
    """
    单个模态的 Mamba 处理块
    输入: [B, L, d_model]
    输出: [B, L, d_model]
    """
    def __init__(
        self,
        d_model: int = 128,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        headdim: int = 32,
        ngroups: int = 1,
    ):
        super().__init__()
        self.d_model = d_model
        
        # 强制要求 Mamba2
        if not MAMBA2_AVAILABLE:
            raise ImportError("HMPN requires mamba_ssm.Mamba2. Please install: pip install mamba-ssm")
        
        # RMSNorm
        self.norm = RMSNorm(d_model)
        
        # Mamba2
        self.mamba = Mamba2(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            headdim=headdim,
            ngroups=ngroups,
        )
        
    def forward(self, x):
        """
        x: [B, L, d_model]
        return: [B, L, d_model]
        """
        # 确保 tensor 满足 Mamba2 的 stride 要求
        # Mamba2 要求 channel-last 格式时，stride(0) 和 stride(2) 必须是 8 的倍数
        x = x.contiguous()
        x_normed = self.norm(x)
        
        # 如果 stride 不满足要求，重新创建连续的 tensor
        if x_normed.stride(0) % 8 != 0 or x_normed.stride(2) % 8 != 0:
            x_normed = x_normed.clone()
        
        out = self.mamba(x_normed)
        return x + out  # residual


# ============================================================================
# CrossModalReinforcement: 跨模态增强
# ============================================================================
class CrossModalReinforcement(nn.Module):
    """
    跨模态增强模块（类似 MHFE 的 cross-modal attention）
    
    用法:
        h_at = CrossModalReinforcement(d_model)(h_a, h_t)
        h_vt = CrossModalReinforcement(d_model)(h_v, h_t)
    
    流程:
        1. Q = W_q @ x_source
        2. K = W_k @ x_target
        3. V = W_v @ x_target
        4. Attention(Q, K, V) → attn_out
        5. Mamba2(attn_out) → mamba_out
        6. output = x_source + mamba_out (residual)
    """
    def __init__(
        self,
        d_model: int = 128,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        headdim: int = 32,
        num_heads: int = 4,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        # 强制要求 Mamba2
        if not MAMBA2_AVAILABLE:
            raise ImportError("HMPN requires mamba_ssm.Mamba2. Please install: pip install mamba-ssm")
        
        # 1. 多头注意力投影
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
        # 2. Mamba2
        self.norm_mamba = RMSNorm(d_model)
        self.mamba = Mamba2(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            headdim=headdim,
        )
    
    def forward(self, x_source, x_target):
        """
        x_source: [B, L_source, d_model]  例如 h_a
        x_target: [B, L_target, d_model]  例如 h_t
        return:   [B, L_source, d_model]
        """
        B = x_source.size(0)
        
        # ====================================================================
        # 步骤1: 多头注意力
        # ====================================================================
        Q = self.q_proj(x_source)  # [B, L_source, d_model]
        K = self.k_proj(x_target)  # [B, L_target, d_model]
        V = self.v_proj(x_target)  # [B, L_target, d_model]
        
        # reshape for multi-head
        Q = Q.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)  # [B, num_heads, L_source, head_dim]
        K = K.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)  # [B, num_heads, L_target, head_dim]
        V = V.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)  # [B, num_heads, L_target, head_dim]
        
        # scaled dot-product attention
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [B, num_heads, L_source, L_target]
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)  # [B, num_heads, L_source, head_dim]
        
        # concat heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, -1, self.d_model)  # [B, L_source, d_model]
        attn_out = self.out_proj(attn_output)
        
        # ====================================================================
        # 步骤2: Mamba2 序列建模
        # ====================================================================
        mamba_in = self.norm_mamba(attn_out)
        
        # 确保 stride 满足 Mamba2 要求
        if mamba_in.stride(0) % 8 != 0 or mamba_in.stride(2) % 8 != 0:
            mamba_in = mamba_in.clone()
        
        mamba_out = self.mamba(mamba_in)
        
        # ====================================================================
        # 步骤3: Residual 连接
        # ====================================================================
        output = x_source + mamba_out
        return output


# ============================================================================
# HMPN: 主模块
# ============================================================================
class HMPN(nn.Module):
    """
    Hierarchical Mamba Perception Network
    分层 Mamba 感知网络
    
    基于 MHFE 架构，使用 Mamba2 实现：
    1. 单模态精炼 (Mamba Block)
    2. 跨模态增强 (Cross-modal Reinforcement)
    3. 超模态融合 (FC)
    
    处理流程：
    Input: h_t^s, h_a^s, h_v^s
      ↓
    Mamba Block → h'_t, h'_a, h'_v
      ↓
    CrossModal Reinforcement:
      - h_at = M(h'_a, h'_t) + residual
      - h_vt = M(h'_v, h'_t) + residual
      ↓
    FC → η (hyper)
    """
    
    def __init__(
        self,
        d_model: int = 128,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        headdim: int = 32,
        ngroups: int = 1,
        num_heads: int = 4,
    ):
        super().__init__()
        
        self.d_model = d_model
        
        # 强制要求 Mamba2
        if not MAMBA2_AVAILABLE:
            raise ImportError("HMPN requires mamba_ssm.Mamba2. Please install: pip install mamba-ssm")
        
        # ====================================================================
        # 1. 单模态 Mamba Block（三个模态各一个）
        # ====================================================================
        self.mamba_text = MambaBlock(d_model, d_state, d_conv, expand, headdim, ngroups)
        self.mamba_audio = MambaBlock(d_model, d_state, d_conv, expand, headdim, ngroups)
        self.mamba_video = MambaBlock(d_model, d_state, d_conv, expand, headdim, ngroups)
        
        # ====================================================================
        # 2. 跨模态增强（两个：audio→text, video→text）
        # ====================================================================
        self.cross_at = CrossModalReinforcement(d_model, d_state, d_conv, expand, headdim, num_heads)
        self.cross_vt = CrossModalReinforcement(d_model, d_state, d_conv, expand, headdim, num_heads)
        
        # ====================================================================
        # 3. 超模态融合 (Hyper Fusion)
        # ====================================================================
        # 输入: [h'_t, h_at, h_vt] → concat → 3*d_model → FC → d_model
        self.hyper_fc = nn.Sequential(
            nn.Linear(3 * d_model, d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, d_model)
        )
    
    def forward(self, h_t, h_a, h_v):
        """
        输入:
            h_t: [B, L_t, d_model]  text specific
            h_a: [B, L_a, d_model]  audio specific
            h_v: [B, L_v, d_model]  video specific
        
        输出:
            η: [B, d_model]  超模态特征
        """
        B = h_t.size(0)
        L_t = h_t.size(1)
        L_a = h_a.size(1)
        L_v = h_v.size(1)
        
        # ====================================================================
        # 步骤1: 单模态精炼
        # ====================================================================
        h_t_prime = self.mamba_text(h_t)    # [B, L_t, d_model]
        h_a_prime = self.mamba_audio(h_a)   # [B, L_a, d_model]
        h_v_prime = self.mamba_video(h_v)   # [B, L_v, d_model]
        
        # ====================================================================
        # 步骤2: 序列长度对齐（如果不同）
        # ====================================================================
        # 取最小长度，使用 adaptive_avg_pool1d 对齐
        min_len = min(L_t, L_a, L_v)
        
        if L_t != min_len:
            h_t_prime = F.adaptive_avg_pool1d(h_t_prime.transpose(1, 2), min_len).transpose(1, 2)
        if L_a != min_len:
            h_a_prime = F.adaptive_avg_pool1d(h_a_prime.transpose(1, 2), min_len).transpose(1, 2)
        if L_v != min_len:
            h_v_prime = F.adaptive_avg_pool1d(h_v_prime.transpose(1, 2), min_len).transpose(1, 2)
        
        # ====================================================================
        # 步骤3: 跨模态增强
        # ====================================================================
        h_at = self.cross_at(h_a_prime, h_t_prime)  # [B, min_len, d_model]
        h_vt = self.cross_vt(h_v_prime, h_t_prime)  # [B, min_len, d_model]
        
        # ====================================================================
        # 步骤4: 池化为固定长度
        # ====================================================================
        # mean pooling over sequence dimension
        h_t_pooled = torch.mean(h_t_prime, dim=1)  # [B, d_model]
        h_at_pooled = torch.mean(h_at, dim=1)      # [B, d_model]
        h_vt_pooled = torch.mean(h_vt, dim=1)      # [B, d_model]
        
        # ====================================================================
        # 步骤5: 超模态融合
        # ====================================================================
        hyper_input = torch.cat([h_t_pooled, h_at_pooled, h_vt_pooled], dim=-1)  # [B, 3*d_model]
        eta = self.hyper_fc(hyper_input)  # [B, d_model]
        
        return eta


# ============================================================================
# 测试代码
# ============================================================================
if __name__ == "__main__":
    print("=" * 80)
    print("HMPN (Hierarchical Mamba Perception Network) 测试")
    print("=" * 80)
    
    if not MAMBA2_AVAILABLE:
        print("❌ Mamba2 不可用，无法测试")
        sys.exit(1)
    
    # 参数
    batch_size = 8
    d_model = 128
    L_t, L_a, L_v = 50, 100, 75  # 不同的序列长度
    
    # 创建模型
    model = HMPN(
        d_model=d_model,
        d_state=64,
        d_conv=4,
        expand=2,
        headdim=32,
        ngroups=1,
        num_heads=4,
    )
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n总参数量: {total_params:,} ({total_params/1e6:.2f}M)")
    
    # 创建测试数据
    h_t = torch.randn(batch_size, L_t, d_model)
    h_a = torch.randn(batch_size, L_a, d_model)
    h_v = torch.randn(batch_size, L_v, d_model)
    
    print(f"\n输入形状:")
    print(f"  h_t (text):  {list(h_t.shape)}")
    print(f"  h_a (audio): {list(h_a.shape)}")
    print(f"  h_v (video): {list(h_v.shape)}")
    
    # 前向传播
    model.eval()
    with torch.no_grad():
        eta = model(h_t, h_a, h_v)
    
    print(f"\n输出形状:")
    print(f"  η (hyper): {list(eta.shape)}")
    
    print("\n✓ HMPN 测试通过!")
    print("=" * 80)
