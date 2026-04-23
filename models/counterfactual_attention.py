"""
跨模态反事实注意力模块 (Cross-Modal Counterfactual Attention)

从 AtCAF 项目移植到 H-DCD 架构的核心创新模块之二。

原理 (反事实推断 / Counterfactual Inference):
    在跨模态注意力的 softmax 之后，对 attention weights 进行反事实干预，
    生成"如果注意力分布不是真实的，预测结果会怎样"的反事实预测。
    
    真实效果 = 真实融合特征 - 反事实融合特征
    即: fusion = factual_fusion - counterfactual_fusion
    
    四种反事实策略 (在 softmax 后操作 attention weights):
    1. random:   用随机值替换非零权重，L1归一化
    2. shuffle:  在batch维度打乱注意力权重（破坏样本-注意力对应关系）
    3. reversed: 取注意力权重的倒数并归一化（关注原本不重要的位置）
    4. uniform:  用均匀分布替换非零权重（消除选择性注意力）

适配说明:
    - 使用标准 PyTorch nn.MultiheadAttention 的基础上自定义 forward
    - 反事实干预仅在训练时启用
    - 输入输出均为 batch-first 格式 [B, L, D]
    - 适配H-DCD的CrossModalReinforcement位置，嵌入HMPN的跨模态交互流程
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CounterfactualMultiheadAttention(nn.Module):
    """
    支持反事实干预的多头注意力
    
    在标准多头注意力的基础上，于softmax之后对attention weights执行反事实策略。
    
    Args:
        d_model (int): 模型维度
        num_heads (int): 注意力头数
        dropout (float): 注意力dropout
    """
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        assert self.head_dim * num_heads == d_model, "d_model必须能被num_heads整除"
        
        self.scaling = self.head_dim ** -0.5
        
        # QKV投影
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.attn_dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, 
                key_padding_mask=None,
                counterfactual_type=None):
        """
        Args:
            query: [B, L_q, d_model]
            key:   [B, L_k, d_model]
            value: [B, L_v, d_model] (通常L_k == L_v)
            key_padding_mask: [B, L_k], True表示填充位置
            counterfactual_type: 反事实策略 ('random'/'shuffle'/'reversed'/'uniform'/None)
        
        Returns:
            output: [B, L_q, d_model]
            attn_weights: [B, num_heads, L_q, L_k] (用于可视化)
        """
        B, L_q, _ = query.shape
        L_k = key.shape[1]
        
        # 线性投影
        Q = self.q_proj(query) * self.scaling  # [B, L_q, d_model]
        K = self.k_proj(key)                    # [B, L_k, d_model]
        V = self.v_proj(value)                  # [B, L_v, d_model]
        
        # reshape为多头: [B, num_heads, L, head_dim]
        Q = Q.view(B, L_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, L_k, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, L_k, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力分数: [B, num_heads, L_q, L_k]
        attn_scores = torch.matmul(Q, K.transpose(-2, -1))
        
        # 应用padding mask
        if key_padding_mask is not None:
            # key_padding_mask: [B, L_k] → [B, 1, 1, L_k]
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(mask, float('-inf'))
        
        # Softmax
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # ====================================================================
        # 反事实干预: 在softmax之后对attention weights进行操作
        # 仅在训练时且指定了counterfactual_type时启用
        # ====================================================================
        if counterfactual_type is not None:
            attn_weights = self._apply_counterfactual(
                attn_weights, counterfactual_type, B, L_q, L_k
            )
        
        # Dropout
        attn_weights_dropped = self.attn_dropout(attn_weights)
        
        # 加权求和: [B, num_heads, L_q, head_dim]
        attn_output = torch.matmul(attn_weights_dropped, V)
        
        # reshape回: [B, L_q, d_model]
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L_q, self.d_model)
        
        # 输出投影
        output = self.out_proj(attn_output)
        
        return output, attn_weights
    
    def _apply_counterfactual(self, attn_weights, cf_type, B, L_q, L_k):
        """
        对attention weights应用反事实策略
        
        Args:
            attn_weights: [B, num_heads, L_q, L_k]
            cf_type: 反事实策略类型
            B, L_q, L_k: 维度信息
        
        Returns:
            modified attn_weights: [B, num_heads, L_q, L_k]
        """
        if cf_type == 'random':
            # ============================================================
            # Random策略: 用随机值替换非零权重，L1归一化
            # 模拟"随机关注"的反事实场景
            # ============================================================
            non_zero_mask = attn_weights != 0
            random_values = torch.rand_like(attn_weights)
            new_weights = torch.where(non_zero_mask, random_values, attn_weights)
            # L1归一化 (在最后一个维度上)
            l1_norm = new_weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            attn_weights = new_weights / l1_norm
            
        elif cf_type == 'shuffle':
            # ============================================================
            # Shuffle策略: 在batch维度打乱注意力权重
            # 破坏样本与注意力模式的对应关系
            # ============================================================
            perm_indices = torch.randperm(B, device=attn_weights.device)
            attn_weights = attn_weights[perm_indices]
            
        elif cf_type == 'reversed':
            # ============================================================
            # Reversed策略: 取倒数后归一化
            # 关注原本不重要的位置（反转注意力偏好）
            # ============================================================
            non_zero_mask = attn_weights != 0
            reciprocal = torch.zeros_like(attn_weights)
            reciprocal[non_zero_mask] = 1.0 / attn_weights[non_zero_mask]
            # 归一化
            sum_vals = torch.where(
                non_zero_mask, reciprocal, torch.zeros_like(attn_weights)
            ).sum(dim=-1, keepdim=True).clamp(min=1e-8)
            attn_weights = torch.where(
                non_zero_mask, reciprocal / sum_vals, attn_weights
            )
            
        elif cf_type == 'uniform':
            # ============================================================
            # Uniform策略: 非零位置用均匀值替换
            # 消除选择性注意力，模拟"平等关注所有位置"
            # ============================================================
            non_zero_mask = attn_weights != 0
            non_zero_count = non_zero_mask.sum(dim=-1, keepdim=True).clamp(min=1)
            non_zero_sum = attn_weights.sum(dim=-1, keepdim=True)
            uniform_val = non_zero_sum / non_zero_count
            attn_weights = torch.where(non_zero_mask, uniform_val, attn_weights)
        
        return attn_weights


class CounterfactualCrossAttention(nn.Module):
    """
    跨模态反事实注意力模块
    
    完整的跨模态注意力块，包含:
    1. CounterfactualMultiheadAttention (支持反事实干预)
    2. LayerNorm + Residual
    3. FFN
    
    用于H-DCD中替代/增补CrossModalReinforcement，
    生成反事实融合特征用于因果效应估计。
    
    Args:
        d_model (int): 模型维度
        num_heads (int): 注意力头数
        num_layers (int): 注意力层数
        dim_feedforward (int): FFN中间维度
        dropout (float): Dropout概率
    """
    
    def __init__(
        self,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.num_layers = num_layers
        
        # 输入投影（如果源和目标模态维度不同时使用）
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = CounterfactualCrossAttentionLayer(
                d_model=d_model,
                num_heads=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
            )
            self.layers.append(layer)
        
        self.final_norm = nn.LayerNorm(d_model)
    
    def forward(self, x_query, x_kv, 
                query_padding_mask=None, 
                kv_padding_mask=None,
                counterfactual_type=None):
        """
        Args:
            x_query: Query模态特征 [B, L_q, d_model]
            x_kv:    Key/Value模态特征 [B, L_kv, d_model]
            query_padding_mask: [B, L_q] (True=填充)
            kv_padding_mask:    [B, L_kv] (True=填充)
            counterfactual_type: 反事实策略 (仅训练时使用)
        
        Returns:
            output: [B, L_q, d_model]
        """
        output = x_query
        for layer in self.layers:
            output = layer(
                output, x_kv,
                kv_padding_mask=kv_padding_mask,
                counterfactual_type=counterfactual_type,
            )
        
        output = self.final_norm(output)
        return output


class CounterfactualCrossAttentionLayer(nn.Module):
    """
    单层反事实跨注意力
    
    结构: CrossAttn → Add&Norm → FFN → Add&Norm
    """
    
    def __init__(self, d_model, num_heads, dim_feedforward, dropout):
        super().__init__()
        
        self.cross_attn = CounterfactualMultiheadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.norm1 = nn.LayerNorm(d_model)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(self, x_query, x_kv, 
                kv_padding_mask=None, 
                counterfactual_type=None):
        """
        Args:
            x_query: [B, L_q, d_model]
            x_kv:    [B, L_kv, d_model]
            kv_padding_mask: [B, L_kv]
            counterfactual_type: 反事实策略
        
        Returns:
            output: [B, L_q, d_model]
        """
        # 跨注意力 + 残差 + LayerNorm
        residual = x_query
        attn_out, _ = self.cross_attn(
            query=x_query,
            key=x_kv,
            value=x_kv,
            key_padding_mask=kv_padding_mask,
            counterfactual_type=counterfactual_type,
        )
        x = self.norm1(residual + attn_out)
        
        # FFN + 残差 + LayerNorm
        residual = x
        ffn_out = self.ffn(x)
        output = self.norm2(residual + ffn_out)
        
        return output