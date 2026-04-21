"""
解耦编码器模块 (Decouple Encoder)
借鉴DMD的实现方式，使用Conv1d进行特征解耦
用于将多模态特征解耦为：
1. 模态特有特征 (Modality-Specific): s_x
2. 模态通用特征 (Modality-Invariant/Common): c_x
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradientReversalLayer(torch.autograd.Function):
    """
    梯度反转层 (Gradient Reversal Layer - GRL)
    前向传播时保持输入不变，反向传播时反转梯度
    用于对抗训练
    """
    @staticmethod
    def forward(ctx, x, lambda_grl):
        ctx.lambda_grl = lambda_grl
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_grl, None


class GRL(nn.Module):
    """梯度反转层包装器"""
    def __init__(self, lambda_grl=1.0):
        super(GRL, self).__init__()
        self.lambda_grl = lambda_grl
    
    def forward(self, x):
        return GradientReversalLayer.apply(x, self.lambda_grl)


class ModalityDiscriminator(nn.Module):
    """
    模态鉴别器 (Modality Discriminator)
    借鉴DMD实现，用于区分不同模态的共同特征
    目标：让共同特征尽可能模态不变 (modality-invariant)
    """
    def __init__(self, input_dim, hidden_dim=256, num_modalities=3):
        super(ModalityDiscriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, num_modalities)
        )
    
    def forward(self, x):
        """
        Args:
            x: [batch_size, input_dim] 或 [3*batch_size, input_dim]
        Returns:
            [batch_size, num_modalities] 模态分类logits
        """
        return self.net(x)


class DecoupleEncoder(nn.Module):
    """
    解耦编码器 (Decouple Encoder)
    借鉴DMD的设计，使用Conv1d进行特征解耦
    
    架构：
    1. 模态特定编码器 (encoder_s_x): 提取模态独有特征
    2. 模态通用编码器 (encoder_c): 提取跨模态通用特征
    3. 解码器 (decoder_x): 从s_x和c_x重构原始特征
    4. 对抗鉴别器: 确保c_x是模态不变的
    
    参考: DMD (Deep Multimodal Decoupling)
    """
    def __init__(self,
                 d_model=128,
                 num_modalities=3,
                 disc_hidden_dim=256,
                 dropout=0.1,
                 lambda_grl=1.0):
        """
        Args:
            d_model: 统一的特征维度
            num_modalities: 模态数量（文本、音频、视频）
            disc_hidden_dim: 鉴别器隐藏层维度
            dropout: Dropout比率
            lambda_grl: 梯度反转层强度
        """
        super(DecoupleEncoder, self).__init__()
        
        self.d_model = d_model
        self.num_modalities = num_modalities
        self.lambda_grl = lambda_grl
        
        # 1. 模态特定编码器 (Modality-Specific Encoders)
        # 使用Conv1d，kernel_size=1相当于逐时间步的线性变换
        self.encoder_s_text = nn.Conv1d(d_model, d_model, kernel_size=1, bias=False)
        self.encoder_s_audio = nn.Conv1d(d_model, d_model, kernel_size=1, bias=False)
        self.encoder_s_video = nn.Conv1d(d_model, d_model, kernel_size=1, bias=False)
        
        # 2. 模态通用编码器 (Modality-Invariant/Common Encoder)
        # 所有模态共享同一个编码器
        self.encoder_c = nn.Conv1d(d_model, d_model, kernel_size=1, bias=False)
        
        # 3. 解码器 (Decoders for Reconstruction)
        # 输入是s_x和c_x的拼接，输出重构的特征
        self.decoder_text = nn.Conv1d(d_model * 2, d_model, kernel_size=1, bias=False)
        self.decoder_audio = nn.Conv1d(d_model * 2, d_model, kernel_size=1, bias=False)
        self.decoder_video = nn.Conv1d(d_model * 2, d_model, kernel_size=1, bias=False)
        
        # 4. 对齐层 (Alignment Layers)
        # 用于将序列特征聚合成固定维度，便于计算相似度和对抗损失
        # 注意：这些层的输入维度需要根据实际序列长度动态计算
        # 这里我们使用自适应池化代替
        self.align_pool = nn.AdaptiveAvgPool1d(1)
        
        # 5. 梯度反转层
        self.grl = GRL(lambda_grl)
        
        # 6. 对抗鉴别器 (Adversarial Discriminator)
        self.discriminator = ModalityDiscriminator(
            input_dim=d_model,
            hidden_dim=disc_hidden_dim,
            num_modalities=num_modalities
        )
        
        # LayerNorm for stability
        self.norm_s_text = nn.LayerNorm(d_model)
        self.norm_s_audio = nn.LayerNorm(d_model)
        self.norm_s_video = nn.LayerNorm(d_model)
        self.norm_c = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, text_proj, audio_proj, video_proj, return_disc=True):
        """
        Args:
            text_proj: [Batch, L_t, d_model] 投影后的文本特征
            audio_proj: [Batch, L_a, d_model] 投影后的音频特征
            video_proj: [Batch, L_v, d_model] 投影后的视频特征
            return_disc: 是否返回鉴别器输出（训练时需要）
        
        Returns:
            字典包含解耦后的特征和鉴别器输出
        """
        batch_size = text_proj.size(0)
        
        # 转换为Conv1d格式: [Batch, Dim, Seq]
        x_text = text_proj.transpose(1, 2)  # [B, d_model, L_t]
        x_audio = audio_proj.transpose(1, 2)  # [B, d_model, L_a]
        x_video = video_proj.transpose(1, 2)  # [B, d_model, L_v]
        
        # ========== 1. 特征解耦 ==========
        # 模态特定特征 (Specific features)
        s_text = self.encoder_s_text(x_text)  # [B, d_model, L_t]
        s_audio = self.encoder_s_audio(x_audio)  # [B, d_model, L_a]
        s_video = self.encoder_s_video(x_video)  # [B, d_model, L_v]
        
        # 模态通用特征 (Common features)
        c_text = self.encoder_c(x_text)  # [B, d_model, L_t]
        c_audio = self.encoder_c(x_audio)  # [B, d_model, L_a]
        c_video = self.encoder_c(x_video)  # [B, d_model, L_v]
        
        # ========== 2. 特征重构 ==========
        # 拼接s_x和c_x进行重构
        recon_text = self.decoder_text(torch.cat([s_text, c_text], dim=1))  # [B, d_model, L_t]
        recon_audio = self.decoder_audio(torch.cat([s_audio, c_audio], dim=1))  # [B, d_model, L_a]
        recon_video = self.decoder_video(torch.cat([s_video, c_video], dim=1))  # [B, d_model, L_v]
        
        # ========== 3. 对抗鉴别器 ==========
        disc_logits = None
        if return_disc:
            # 对共同特征进行时间维度池化: [B, d_model, L] -> [B, d_model]
            c_text_pooled = self.align_pool(c_text).squeeze(-1)  # [B, d_model]
            c_audio_pooled = self.align_pool(c_audio).squeeze(-1)  # [B, d_model]
            c_video_pooled = self.align_pool(c_video).squeeze(-1)  # [B, d_model]
            
            # 拼接所有模态的共同特征: [3*B, d_model]
            common_features = torch.cat([c_text_pooled, c_audio_pooled, c_video_pooled], dim=0)
            
            # 应用梯度反转层（训练时）
            if self.training:
                common_features = self.grl(common_features)
            
            # 鉴别器预测模态类别
            disc_logits = self.discriminator(common_features)  # [3*B, num_modalities]
        
        # ========== 4. 格式转换回 [Batch, Seq, Dim] ==========
        # 转回序列格式，并应用LayerNorm
        s_text = self.norm_s_text(s_text.transpose(1, 2))  # [B, L_t, d_model]
        s_audio = self.norm_s_audio(s_audio.transpose(1, 2))  # [B, L_a, d_model]
        s_video = self.norm_s_video(s_video.transpose(1, 2))  # [B, L_v, d_model]
        
        c_text = self.norm_c(c_text.transpose(1, 2))  # [B, L_t, d_model]
        c_audio = self.norm_c(c_audio.transpose(1, 2))  # [B, L_a, d_model]
        c_video = self.norm_c(c_video.transpose(1, 2))  # [B, L_v, d_model]
        
        recon_text = recon_text.transpose(1, 2)  # [B, L_t, d_model]
        recon_audio = recon_audio.transpose(1, 2)  # [B, L_a, d_model]
        recon_video = recon_video.transpose(1, 2)  # [B, L_v, d_model]
        
        return {
            # 模态特定特征 (Specific)
            's_text': s_text,
            's_audio': s_audio,
            's_video': s_video,
            # 模态通用特征 (Common)
            'c_text': c_text,
            'c_audio': c_audio,
            'c_video': c_video,
            # 重构特征
            'recon_text': recon_text,
            'recon_audio': recon_audio,
            'recon_video': recon_video,
            # 鉴别器输出
            'disc_logits': disc_logits
        }


# 测试代码
if __name__ == "__main__":
    torch.manual_seed(42)
    
    # 模拟输入数据（已经过Feature Projection，维度统一为d_model=128）
    batch_size = 4
    L_t, L_a, L_v = 50, 100, 75
    d_model = 128
    
    text_proj = torch.randn(batch_size, L_t, d_model)
    audio_proj = torch.randn(batch_size, L_a, d_model)
    video_proj = torch.randn(batch_size, L_v, d_model)
    
    print("输入形状（已投影）:")
    print(f"Text: {text_proj.shape}")
    print(f"Audio: {audio_proj.shape}")
    print(f"Video: {video_proj.shape}")
    print()
    
    # 创建解耦编码器
    decouple_encoder = DecoupleEncoder(
        d_model=128,
        num_modalities=3,
        disc_hidden_dim=256,
        dropout=0.1,
        lambda_grl=1.0
    )
    
    # 前向传播
    decouple_encoder.train()  # 训练模式
    outputs = decouple_encoder(text_proj, audio_proj, video_proj, return_disc=True)
    
    print("解耦后的特征形状:")
    print("\n模态特定特征 (Specific):")
    print(f"  s_text:  {outputs['s_text'].shape}")
    print(f"  s_audio: {outputs['s_audio'].shape}")
    print(f"  s_video: {outputs['s_video'].shape}")
    
    print("\n模态通用特征 (Common):")
    print(f"  c_text:  {outputs['c_text'].shape}")
    print(f"  c_audio: {outputs['c_audio'].shape}")
    print(f"  c_video: {outputs['c_video'].shape}")
    
    print("\n重构特征:")
    print(f"  recon_text:  {outputs['recon_text'].shape}")
    print(f"  recon_audio: {outputs['recon_audio'].shape}")
    print(f"  recon_video: {outputs['recon_video'].shape}")
    
    print("\n鉴别器输出:")
    print(f"  disc_logits: {outputs['disc_logits'].shape}")
    print(f"  (应该是 [3*{batch_size}, 3] = [{3*batch_size}, 3])")
    
    # 验证输出维度
    assert outputs['s_text'].shape == (batch_size, L_t, d_model)
    assert outputs['c_text'].shape == (batch_size, L_t, d_model)
    assert outputs['recon_text'].shape == (batch_size, L_t, d_model)
    assert outputs['disc_logits'].shape == (3 * batch_size, 3)
    
    print("\n✓ 所有维度检查通过！")
    
    # 计算参数量
    total_params = sum(p.numel() for p in decouple_encoder.parameters())
    trainable_params = sum(p.numel() for p in decouple_encoder.parameters() if p.requires_grad)
    print(f"\n模型参数:")
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
