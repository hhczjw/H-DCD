"""
单模态因果去偏模块 (Unimodal Causal Debiasing Module)

从 AtCAF 项目移植到 H-DCD 架构的核心创新模块之一。

原理 (前门调整 / Front-Door Adjustment):
    对于每个模态，通过双路径机制消除混杂因子的影响：
    - IS路径 (Individual Self-attention): 局部自注意力，捕获模态内部的局部依赖
    - CS路径 (Confounder-aware Cross-attention): 与可学习的混杂因子字典进行全局跨注意力，
      通过边缘化混杂因子来消除虚假关联
    最终将两条路径的输出拼接，再通过MLP映射回原始维度。

适配说明:
    - AtCAF使用 seq_len-first 格式 (seq_len, batch, dim)，H-DCD使用 batch-first 格式 (batch, seq_len, dim)
    - 本模块统一使用 batch-first 格式，内部转换后调用注意力层
    - 混杂因子字典支持KMeans初始化（需提供.npy文件）或随机初始化
    - 自注意力和跨注意力使用标准PyTorch nn.MultiheadAttention，无需AtCAF的自定义实现
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class UnimodalDebiasModule(nn.Module):
    """
    单模态因果去偏模块
    
    通过前门调整方法，利用双路径（局部自注意力 + 全局混杂因子跨注意力）
    去除单模态特征中的虚假关联/混杂偏差。
    
    Args:
        d_model (int): 输入特征维度（H-DCD统一维度，如128）
        num_heads (int): 注意力头数
        num_layers (int): 注意力层数
        confounder_size (int): 混杂因子字典大小（KMeans聚类数）
        dropout (float): Dropout概率
        confounder_npy_path (str, optional): KMeans初始化的.npy文件路径
    """
    
    def __init__(
        self,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        confounder_size: int = 50,
        dropout: float = 0.1,
        confounder_npy_path: str = None,
    ):
        super(UnimodalDebiasModule, self).__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.confounder_size = confounder_size
        
        # ====================================================================
        # IS路径: 局部自注意力编码器 (Individual Self-attention)
        # 捕获模态内部的局部时序依赖，不受混杂因子影响
        # ====================================================================
        is_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,  # H-DCD使用batch-first格式
            activation='relu',
        )
        self.is_encoder = nn.TransformerEncoder(
            encoder_layer=is_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )
        
        # ====================================================================
        # CS路径: 混杂因子跨注意力编码器 (Confounder-aware Cross-attention)
        # Query=模态特征, Key/Value=混杂因子字典
        # 通过与字典的注意力交互来边缘化混杂因子
        # ====================================================================
        # 使用nn.MultiheadAttention实现跨注意力（Query来自输入，K/V来自字典）
        self.cs_cross_attn_layers = nn.ModuleList()
        self.cs_norms = nn.ModuleList()
        self.cs_ffn_layers = nn.ModuleList()
        self.cs_ffn_norms = nn.ModuleList()
        
        for _ in range(num_layers):
            # 跨注意力层
            self.cs_cross_attn_layers.append(
                nn.MultiheadAttention(
                    embed_dim=d_model,
                    num_heads=num_heads,
                    dropout=dropout,
                    batch_first=True,
                )
            )
            self.cs_norms.append(nn.LayerNorm(d_model))
            
            # FFN层
            self.cs_ffn_layers.append(nn.Sequential(
                nn.Linear(d_model, 4 * d_model),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(4 * d_model, d_model),
                nn.Dropout(dropout),
            ))
            self.cs_ffn_norms.append(nn.LayerNorm(d_model))
        
        # ====================================================================
        # 混杂因子字典 (Confounder Dictionary)
        # 可学习参数，形状 [confounder_size, d_model]
        # 支持KMeans初始化或随机初始化
        # ====================================================================
        self._init_confounder_dictionary(confounder_npy_path)
        
        # ====================================================================
        # 融合MLP: 将IS和CS路径的输出拼接后映射回d_model
        # IS输出: [B, L, d_model], CS输出: [B, L, d_model]
        # 拼接后: [B, L, 2*d_model] → MLP → [B, L, d_model]
        # ====================================================================
        self.fusion_mlp = nn.Sequential(
            nn.Linear(2 * d_model, 4 * d_model),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, 2 * d_model),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model),
        )
    
    def _init_confounder_dictionary(self, npy_path):
        """
        初始化混杂因子字典
        
        如果提供了npy_path，则使用KMeans聚类中心初始化（需要预先运行KMeans）；
        否则使用随机初始化（值除以100以保持较小的初始值，与AtCAF保持一致）。
        
        Args:
            npy_path: .npy文件路径，包含 [confounder_size, d_model] 的聚类中心
        """
        if npy_path is not None and os.path.exists(npy_path):
            # KMeans初始化
            center_data = np.load(npy_path)  # [confounder_size, d_model]
            if center_data.ndim == 1:
                center_data = center_data.reshape(self.confounder_size, -1)
            # 确保维度匹配
            assert center_data.shape == (self.confounder_size, self.d_model), \
                f"混杂因子字典维度不匹配: 期望({self.confounder_size}, {self.d_model}), " \
                f"实际{center_data.shape}"
            self.confounder_dict = nn.Parameter(
                torch.from_numpy(center_data).float()
            )
            print(f"[CausalDebias] 使用KMeans初始化混杂因子字典: {npy_path}")
        else:
            # 随机初始化（除以100保持小值，与AtCAF一致）
            self.confounder_dict = nn.Parameter(
                torch.rand(self.confounder_size, self.d_model) / 100.0
            )
            print(f"[CausalDebias] 随机初始化混杂因子字典: "
                  f"size={self.confounder_size}, dim={self.d_model}")
    
    def forward(self, x, src_key_padding_mask=None):
        """
        前向传播
        
        Args:
            x: 输入特征 [B, L, d_model] (batch-first格式)
            src_key_padding_mask: 填充掩码 [B, L], True表示填充位置
        
        Returns:
            output: 去偏后的特征 [B, L, d_model]
        """
        B, L, D = x.shape
        
        # ====================================================================
        # IS路径: 局部自注意力
        # 直接对输入做自注意力编码，捕获模态内部依赖
        # ====================================================================
        is_output = self.is_encoder(
            x, 
            src_key_padding_mask=src_key_padding_mask
        )  # [B, L, d_model]
        
        # ====================================================================
        # CS路径: 与混杂因子字典的跨注意力
        # Query来自输入特征, Key/Value来自混杂因子字典
        # 混杂因子字典不需要位置编码（全局统计量，与AtCAF设计一致）
        # ====================================================================
        # 扩展字典到batch维度: [confounder_size, d_model] → [B, confounder_size, d_model]
        dict_expanded = self.confounder_dict.unsqueeze(0).expand(B, -1, -1)
        
        cs_output = x  # 初始化为输入（用于残差）
        for i in range(self.num_layers):
            # 跨注意力: Query=模态特征, Key/Value=混杂因子字典
            residual = cs_output
            attn_out, _ = self.cs_cross_attn_layers[i](
                query=cs_output,        # [B, L, d_model]
                key=dict_expanded,       # [B, confounder_size, d_model]
                value=dict_expanded,     # [B, confounder_size, d_model]
                key_padding_mask=None,   # 字典无填充
            )
            cs_output = self.cs_norms[i](residual + attn_out)
            
            # FFN
            residual = cs_output
            ffn_out = self.cs_ffn_layers[i](cs_output)
            cs_output = self.cs_ffn_norms[i](residual + ffn_out)
        
        # ====================================================================
        # 融合: 拼接IS和CS路径的输出，通过MLP映射回d_model
        # ====================================================================
        # 拼接: [B, L, 2*d_model]
        combined = torch.cat([is_output, cs_output], dim=-1)
        
        # MLP映射: [B, L, 2*d_model] → [B, L, d_model]
        output = self.fusion_mlp(combined)
        
        return output


class MultiModalDebiasWrapper(nn.Module):
    """
    多模态去偏包装器
    
    为H-DCD的三个模态（text, audio, video）分别创建UnimodalDebiasModule，
    统一管理和调用。
    
    Args:
        d_model (int): 统一特征维度
        num_heads (int): 注意力头数
        num_layers (int): 每个去偏模块的注意力层数
        confounder_size (int): 混杂因子字典大小
        dropout (float): Dropout概率
        debias_text (bool): 是否对文本去偏
        debias_audio (bool): 是否对音频去偏
        debias_video (bool): 是否对视频去偏
        confounder_npy_dir (str, optional): 包含KMeans .npy文件的目录路径
        dataset_name (str): 数据集名称（用于定位.npy文件）
    """
    
    def __init__(
        self,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        confounder_size: int = 50,
        dropout: float = 0.1,
        debias_text: bool = True,
        debias_audio: bool = True,
        debias_video: bool = True,
        confounder_npy_dir: str = None,
        dataset_name: str = 'mosi',
    ):
        super(MultiModalDebiasWrapper, self).__init__()
        
        self.debias_text = debias_text
        self.debias_audio = debias_audio
        self.debias_video = debias_video
        
        # 尝试为每个模态构建npy路径
        def _get_npy_path(modal_name):
            if confounder_npy_dir is None:
                return None
            path = os.path.join(
                confounder_npy_dir,
                f"kmeans_{dataset_name}-{confounder_size}_{modal_name}.npy"
            )
            return path if os.path.exists(path) else None
        
        # 为启用去偏的模态创建去偏模块
        if debias_text:
            self.text_debias = UnimodalDebiasModule(
                d_model=d_model,
                num_heads=num_heads,
                num_layers=num_layers,
                confounder_size=confounder_size,
                dropout=dropout,
                confounder_npy_path=_get_npy_path('text'),
            )
        
        if debias_audio:
            self.audio_debias = UnimodalDebiasModule(
                d_model=d_model,
                num_heads=num_heads,
                num_layers=num_layers,
                confounder_size=confounder_size,
                dropout=dropout,
                confounder_npy_path=_get_npy_path('audio'),
            )
        
        if debias_video:
            self.video_debias = UnimodalDebiasModule(
                d_model=d_model,
                num_heads=num_heads,
                num_layers=num_layers,
                confounder_size=confounder_size,
                dropout=dropout,
                confounder_npy_path=_get_npy_path('visual'),
            )
    
    def forward(self, x_text, x_audio, x_video,
                text_mask=None, audio_mask=None, video_mask=None):
        """
        对三个模态分别执行因果去偏
        
        Args:
            x_text:  [B, L_t, d_model]
            x_audio: [B, L_a, d_model]
            x_video: [B, L_v, d_model]
            text_mask:  [B, L_t] padding mask (True=填充位置)
            audio_mask: [B, L_a] padding mask
            video_mask: [B, L_v] padding mask
        
        Returns:
            debiased_text:  [B, L_t, d_model]
            debiased_audio: [B, L_a, d_model]
            debiased_video: [B, L_v, d_model]
        """
        # 文本去偏
        if self.debias_text:
            debiased_text = self.text_debias(x_text, src_key_padding_mask=text_mask)
        else:
            debiased_text = x_text
        
        # 音频去偏
        if self.debias_audio:
            debiased_audio = self.audio_debias(x_audio, src_key_padding_mask=audio_mask)
        else:
            debiased_audio = x_audio
        
        # 视频去偏
        if self.debias_video:
            debiased_video = self.video_debias(x_video, src_key_padding_mask=video_mask)
        else:
            debiased_video = x_video
        
        return debiased_text, debiased_audio, debiased_video