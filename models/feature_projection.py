"""
特征投影模块 (Feature Projection Module)
用于将不同模态的特征统一映射到相同的维度空间

输入:
- Text: (Batch, L_t, D_t) - 文本序列特征
- Audio: (Batch, L_a, D_a) - 音频序列特征  
- Video: (Batch, L_v, D_v) - 视频序列特征

输出:
- text_proj: (Batch, L_t, d_model)
- audio_proj: (Batch, L_a, d_model)
- video_proj: (Batch, L_v, d_model)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextProjection(nn.Module):
    """
    文本特征投影模块
    使用BiGRU进行序列特征提取，然后映射到目标维度
    """
    def __init__(self, input_dim, hidden_dim=256, output_dim=128, 
                 num_layers=2, dropout=0.1, bidirectional=True):
        """
        Args:
            input_dim: 输入文本特征维度 D_t
            hidden_dim: GRU隐藏层维度
            output_dim: 输出特征维度 d_model
            num_layers: GRU层数
            dropout: Dropout比率
            bidirectional: 是否使用双向GRU
        """
        super(TextProjection, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # BiGRU层
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # 计算GRU输出维度（双向时需要×2）
        gru_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        # 投影层：将GRU输出映射到d_model
        self.projection = nn.Linear(gru_output_dim, output_dim)
        
        # Layer Normalization
        self.layer_norm = nn.LayerNorm(output_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Args:
            x: (Batch, L_t, D_t) 文本特征
        Returns:
            (Batch, L_t, d_model) 投影后的文本特征
        """
        # BiGRU特征提取
        # gru_out: (Batch, L_t, hidden_dim*2)
        gru_out, _ = self.gru(x)
        
        # 线性投影
        # proj_out: (Batch, L_t, d_model)
        proj_out = self.projection(gru_out)
        
        # LayerNorm + Dropout
        proj_out = self.layer_norm(proj_out)
        proj_out = self.dropout(proj_out)
        
        return proj_out


class AudioVideoProjection(nn.Module):
    """
    音频/视频特征投影模块
    使用多层DNN进行特征映射
    """
    def __init__(self, input_dim, hidden_dim=256, output_dim=128, dropout=0.1):
        """
        Args:
            input_dim: 输入特征维度 D_a 或 D_v
            hidden_dim: 隐藏层维度
            output_dim: 输出特征维度 d_model
            dropout: Dropout比率
        """
        super(AudioVideoProjection, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # DNN投影网络
        self.dnn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU()
        )
        
        # Layer Normalization
        self.layer_norm = nn.LayerNorm(output_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Args:
            x: (Batch, L, D) 音频或视频特征
        Returns:
            (Batch, L, d_model) 投影后的特征
        """
        # DNN投影
        # proj_out: (Batch, L, d_model)
        proj_out = self.dnn(x)
        
        # LayerNorm + Dropout
        proj_out = self.layer_norm(proj_out)
        proj_out = self.dropout(proj_out)
        
        return proj_out


class FeatureProjection(nn.Module):
    """
    完整的特征投影模块
    统一处理文本、音频、视频三种模态的特征投影
    """
    def __init__(self, 
                 text_dim,
                 audio_dim,
                 video_dim,
                 d_model=128,
                 text_hidden_dim=256,
                 av_hidden_dim=256,
                 text_num_layers=2,
                 dropout=0.1):
        """
        Args:
            text_dim: 文本输入维度 D_t
            audio_dim: 音频输入维度 D_a
            video_dim: 视频输入维度 D_v
            d_model: 统一的输出维度
            text_hidden_dim: 文本BiGRU隐藏层维度
            av_hidden_dim: 音频/视频DNN隐藏层维度
            text_num_layers: BiGRU层数
            dropout: Dropout比率
        """
        super(FeatureProjection, self).__init__()
        
        self.text_dim = text_dim
        self.audio_dim = audio_dim
        self.video_dim = video_dim
        self.d_model = d_model
        
        # 文本投影模块（BiGRU）
        self.text_projection = TextProjection(
            input_dim=text_dim,
            hidden_dim=text_hidden_dim,
            output_dim=d_model,
            num_layers=text_num_layers,
            dropout=dropout,
            bidirectional=True
        )
        
        # 音频投影模块（DNN）
        self.audio_projection = AudioVideoProjection(
            input_dim=audio_dim,
            hidden_dim=av_hidden_dim,
            output_dim=d_model,
            dropout=dropout
        )
        
        # 视频投影模块（DNN）
        self.video_projection = AudioVideoProjection(
            input_dim=video_dim,
            hidden_dim=av_hidden_dim,
            output_dim=d_model,
            dropout=dropout
        )
    
    def forward(self, text, audio, video):
        """
        Args:
            text: (Batch, L_t, D_t) 文本特征
            audio: (Batch, L_a, D_a) 音频特征
            video: (Batch, L_v, D_v) 视频特征
        
        Returns:
            text_proj: (Batch, L_t, d_model) 投影后的文本特征
            audio_proj: (Batch, L_a, d_model) 投影后的音频特征
            video_proj: (Batch, L_v, d_model) 投影后的视频特征
        """
        # 分别投影三种模态
        text_proj = self.text_projection(text)
        audio_proj = self.audio_projection(audio)
        video_proj = self.video_projection(video)
        
        return text_proj, audio_proj, video_proj
    
    def get_output_dim(self):
        """返回输出维度"""
        return self.d_model


# 测试代码
if __name__ == "__main__":
    # 设置随机种子
    torch.manual_seed(42)
    
    # 模拟输入数据
    batch_size = 4
    L_t, L_a, L_v = 50, 100, 75  # 序列长度
    D_t, D_a, D_v = 768, 74, 35  # 特征维度（BERT, ComParE, DenseFace）
    
    text = torch.randn(batch_size, L_t, D_t)
    audio = torch.randn(batch_size, L_a, D_a)
    video = torch.randn(batch_size, L_v, D_v)
    
    print("输入形状:")
    print(f"Text: {text.shape}")
    print(f"Audio: {audio.shape}")
    print(f"Video: {video.shape}")
    print()
    
    # 创建特征投影模块
    feature_proj = FeatureProjection(
        text_dim=D_t,
        audio_dim=D_a,
        video_dim=D_v,
        d_model=128,
        text_hidden_dim=256,
        av_hidden_dim=256,
        text_num_layers=2,
        dropout=0.1
    )
    
    # 前向传播
    text_proj, audio_proj, video_proj = feature_proj(text, audio, video)
    
    print("输出形状:")
    print(f"Text projection: {text_proj.shape}")
    print(f"Audio projection: {audio_proj.shape}")
    print(f"Video projection: {video_proj.shape}")
    print()
    
    # 验证输出维度
    assert text_proj.shape == (batch_size, L_t, 128)
    assert audio_proj.shape == (batch_size, L_a, 128)
    assert video_proj.shape == (batch_size, L_v, 128)
    
    print("✓ 所有维度检查通过！")
    print(f"✓ 统一输出维度: d_model = {feature_proj.get_output_dim()}")
    
    # 计算参数量
    total_params = sum(p.numel() for p in feature_proj.parameters())
    trainable_params = sum(p.numel() for p in feature_proj.parameters() if p.requires_grad)
    print(f"\n模型参数:")
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
