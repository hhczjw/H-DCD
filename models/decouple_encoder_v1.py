"""
解耦编码器 - 原始版本 (Decouple Encoder V1)
基于线性层的实现方式

架构设计：
1. 共享编码器 (Shared Encoder) - 将每个模态映射到共享空间
2. 通用-特有分解器 (Common-Private Decomposer) - 分解为通用特征和特有特征
3. 对抗鉴别器 (Adversarial Discriminator) - 确保通用特征是模态不变的
4. 模态鉴别器 (Modality Discriminator) - 确保特有特征真的特有于某个模态

对比DMD版本：
- DMD版本：使用Conv1d + 重构损失
- 原始版本：使用Linear + 双层分解（共享→通用/特有）
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


class SharedEncoder(nn.Module):
    """
    共享编码器 E_shared
    将每个模态的特征映射到共享的潜在空间
    """
    def __init__(self, input_dim, shared_dim, dropout=0.1):
        super(SharedEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, shared_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(shared_dim * 2, shared_dim),
            nn.ReLU(),
            nn.LayerNorm(shared_dim)
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, input_dim] 或 [batch_size, input_dim]
        Returns:
            [batch_size, seq_len, shared_dim] 或 [batch_size, shared_dim]
        """
        return self.dropout(self.encoder(x))


class CommonPrivateDecomposer(nn.Module):
    """
    通用-特有分解器
    将共享特征进一步分解为：
    1. 通用特征 (Common): 跨模态共享的信息
    2. 特有特征 (Private): 模态特定的信息
    """
    def __init__(self, shared_dim, common_dim, private_dim, dropout=0.1):
        super(CommonPrivateDecomposer, self).__init__()
        
        # 通用特征分支
        self.common_encoder = nn.Sequential(
            nn.Linear(shared_dim, common_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(common_dim * 2, common_dim),
            nn.LayerNorm(common_dim)
        )
        
        # 特有特征分支
        self.private_encoder = nn.Sequential(
            nn.Linear(shared_dim, private_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(private_dim * 2, private_dim),
            nn.LayerNorm(private_dim)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, shared_dim] 或 [batch_size, shared_dim]
        Returns:
            common: [batch_size, seq_len, common_dim]
            private: [batch_size, seq_len, private_dim]
        """
        common = self.dropout(self.common_encoder(x))
        private = self.dropout(self.private_encoder(x))
        return common, private


class AdversarialDiscriminator(nn.Module):
    """
    对抗鉴别器 D_adv
    用于区分不同模态的通用特征
    目标：让通用特征尽可能模态不变
    """
    def __init__(self, input_dim, num_modalities=3, hidden_dim=256, dropout=0.1):
        super(AdversarialDiscriminator, self).__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_modalities)
        )
    
    def forward(self, x):
        """
        Args:
            x: [batch_size, input_dim] 通用特征
        Returns:
            [batch_size, num_modalities] 模态分类logits
        """
        return self.discriminator(x)


class ModalityDiscriminator(nn.Module):
    """
    模态鉴别器 D_modality
    确保特有特征真的特有于某个模态
    输出该特征属于对应模态的概率
    """
    def __init__(self, input_dim, hidden_dim=128, dropout=0.1):
        super(ModalityDiscriminator, self).__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        Args:
            x: [batch_size, input_dim] 特有特征
        Returns:
            [batch_size, 1] 属于对应模态的概率
        """
        return self.discriminator(x)


class SharedDecoder(nn.Module):
    """
    共享特征解码器
    将 Common + Private 特征重构回 Shared 特征
    """
    def __init__(self, common_dim, private_dim, shared_dim, dropout=0.1):
        super(SharedDecoder, self).__init__()
        input_dim = common_dim + private_dim
        self.decoder = nn.Sequential(
            nn.Linear(input_dim, shared_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(shared_dim * 2, shared_dim),
            nn.LayerNorm(shared_dim)
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, common, private):
        """
        Args:
            common: [batch_size, seq_len, common_dim] 通用特征
            private: [batch_size, seq_len, private_dim] 特有特征
        Returns:
            [batch_size, seq_len, shared_dim] 重构的共享特征
        """
        # 拼接 common 和 private
        combined = torch.cat([common, private], dim=-1)
        # 解码回 shared 空间
        recon_shared = self.dropout(self.decoder(combined))
        return recon_shared


class FeatureDecoder(nn.Module):
    """
    特征解码器
    将 Shared 特征重构回原始输入特征空间
    """
    def __init__(self, shared_dim, d_model, dropout=0.1):
        super(FeatureDecoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(shared_dim, shared_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(shared_dim, d_model),
            nn.LayerNorm(d_model)
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, shared):
        """
        Args:
            shared: [batch_size, seq_len, shared_dim] 共享特征
        Returns:
            [batch_size, seq_len, d_model] 重构的原始特征
        """
        recon = self.dropout(self.decoder(shared))
        return recon


class DecoupleEncoderV1(nn.Module):
    """
    解耦编码器 - 原始版本
    
    架构流程：
    Input (x_text, x_audio, x_video) 
      ↓ 
    [Shared Encoder] → 共享特征空间
      ↓
    [Common-Private Decomposer] → 通用特征 + 特有特征
      ↓
    [Discriminators] → 对抗训练确保特征解耦
    
    输出：
    - common features: 跨模态共享的语义信息
    - private features: 模态特定的独有信息
    """
    def __init__(self,
                 d_model=128,
                 shared_dim=256,
                 common_dim=128,
                 private_dim=128,
                 num_modalities=3,
                 disc_hidden_dim=256,
                 dropout=0.1,
                 lambda_grl=1.0):
        """
        Args:
            d_model: 输入特征维度（来自Feature Projection）
            shared_dim: 共享编码空间维度
            common_dim: 通用特征维度
            private_dim: 特有特征维度
            num_modalities: 模态数量
            disc_hidden_dim: 鉴别器隐藏层维度
            dropout: Dropout比率
            lambda_grl: 梯度反转强度
        """
        super(DecoupleEncoderV1, self).__init__()
        
        self.d_model = d_model
        self.shared_dim = shared_dim
        self.common_dim = common_dim
        self.private_dim = private_dim
        self.num_modalities = num_modalities
        self.lambda_grl = lambda_grl
        
        # 1. 共享编码器（每个模态一个）
        self.text_shared_encoder = SharedEncoder(d_model, shared_dim, dropout)
        self.audio_shared_encoder = SharedEncoder(d_model, shared_dim, dropout)
        self.video_shared_encoder = SharedEncoder(d_model, shared_dim, dropout)
        
        # 2. 通用-特有分解器（每个模态一个）
        self.text_decomposer = CommonPrivateDecomposer(shared_dim, common_dim, private_dim, dropout)
        self.audio_decomposer = CommonPrivateDecomposer(shared_dim, common_dim, private_dim, dropout)
        self.video_decomposer = CommonPrivateDecomposer(shared_dim, common_dim, private_dim, dropout)
        
        # 3. 梯度反转层
        self.grl = GRL(lambda_grl)
        
        # 4. 对抗鉴别器（用于通用特征）
        self.adv_discriminator = AdversarialDiscriminator(
            input_dim=common_dim,
            num_modalities=num_modalities,
            hidden_dim=disc_hidden_dim,
            dropout=dropout
        )
        
        # 5. 模态鉴别器（用于特有特征，每个模态一个）
        self.text_modality_disc = ModalityDiscriminator(private_dim, hidden_dim=128, dropout=dropout)
        self.audio_modality_disc = ModalityDiscriminator(private_dim, hidden_dim=128, dropout=dropout)
        self.video_modality_disc = ModalityDiscriminator(private_dim, hidden_dim=128, dropout=dropout)
        
        # 6. 重构解码器
        # 6.1 共享特征解码器（从 Common+Private 重构 Shared）
        self.text_shared_decoder = SharedDecoder(common_dim, private_dim, shared_dim, dropout)
        self.audio_shared_decoder = SharedDecoder(common_dim, private_dim, shared_dim, dropout)
        self.video_shared_decoder = SharedDecoder(common_dim, private_dim, shared_dim, dropout)
        
        # 6.2 特征解码器（从 Shared 重构原始输入）
        self.text_feature_decoder = FeatureDecoder(shared_dim, d_model, dropout)
        self.audio_feature_decoder = FeatureDecoder(shared_dim, d_model, dropout)
        self.video_feature_decoder = FeatureDecoder(shared_dim, d_model, dropout)
        
        # 池化层（用于将序列特征聚合成固定维度）
        self.pool = nn.AdaptiveAvgPool1d(1)
    
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
        
        # ========== 1. 共享编码 ==========
        text_shared = self.text_shared_encoder(text_proj)    # [B, L_t, shared_dim]
        audio_shared = self.audio_shared_encoder(audio_proj)  # [B, L_a, shared_dim]
        video_shared = self.video_shared_encoder(video_proj)  # [B, L_v, shared_dim]
        
        # ========== 2. 通用-特有分解 ==========
        text_common, text_private = self.text_decomposer(text_shared)
        audio_common, audio_private = self.audio_decomposer(audio_shared)
        video_common, video_private = self.video_decomposer(video_shared)
        
        # ========== 3. 重构机制 ==========
        # 3.1 从 Common+Private 重构 Shared
        text_shared_recon = self.text_shared_decoder(text_common, text_private)
        audio_shared_recon = self.audio_shared_decoder(audio_common, audio_private)
        video_shared_recon = self.video_shared_decoder(video_common, video_private)
        
        # 3.2 从 Shared 重构原始输入
        text_recon = self.text_feature_decoder(text_shared_recon)
        audio_recon = self.audio_feature_decoder(audio_shared_recon)
        video_recon = self.video_feature_decoder(video_shared_recon)
        
        # ========== 4. 对抗鉴别（训练时） ==========
        disc_logits = None
        text_mod_prob = None
        audio_mod_prob = None
        video_mod_prob = None
        
        if return_disc:
            # 对通用特征进行池化：[B, L, dim] -> [B, dim]
            text_common_pooled = self.pool(text_common.transpose(1, 2)).squeeze(-1)
            audio_common_pooled = self.pool(audio_common.transpose(1, 2)).squeeze(-1)
            video_common_pooled = self.pool(video_common.transpose(1, 2)).squeeze(-1)
            
            # 拼接所有模态的通用特征：[3*B, common_dim]
            all_common = torch.cat([text_common_pooled, audio_common_pooled, video_common_pooled], dim=0)
            
            # 应用梯度反转层（训练时）
            if self.training:
                all_common = self.grl(all_common)
            
            # 对抗鉴别器预测模态
            disc_logits = self.adv_discriminator(all_common)  # [3*B, num_modalities]
            
            # 模态鉴别器：确保特有特征真的特有
            text_private_pooled = self.pool(text_private.transpose(1, 2)).squeeze(-1)
            audio_private_pooled = self.pool(audio_private.transpose(1, 2)).squeeze(-1)
            video_private_pooled = self.pool(video_private.transpose(1, 2)).squeeze(-1)
            
            text_mod_prob = self.text_modality_disc(text_private_pooled)      # [B, 1]
            audio_mod_prob = self.audio_modality_disc(audio_private_pooled)   # [B, 1]
            video_mod_prob = self.video_modality_disc(video_private_pooled)   # [B, 1]
        
        return {
            # 共享特征
            'text_shared': text_shared,
            'audio_shared': audio_shared,
            'video_shared': video_shared,
            
            # 通用特征 (跨模态共享)
            'text_common': text_common,
            'audio_common': audio_common,
            'video_common': video_common,
            
            # 特有特征 (模态独有)
            'text_private': text_private,
            'audio_private': audio_private,
            'video_private': video_private,
            
            # 重构特征
            'text_shared_recon': text_shared_recon,     # [B, L_t, shared_dim]
            'audio_shared_recon': audio_shared_recon,   # [B, L_a, shared_dim]
            'video_shared_recon': video_shared_recon,   # [B, L_v, shared_dim]
            'text_recon': text_recon,                   # [B, L_t, d_model]
            'audio_recon': audio_recon,                 # [B, L_a, d_model]
            'video_recon': video_recon,                 # [B, L_v, d_model]
            
            # 鉴别器输出
            'disc_logits': disc_logits,           # [3*B, num_modalities]
            'text_mod_prob': text_mod_prob,       # [B, 1]
            'audio_mod_prob': audio_mod_prob,     # [B, 1]
            'video_mod_prob': video_mod_prob      # [B, 1]
        }


# 测试代码
if __name__ == "__main__":
    torch.manual_seed(42)
    
    # 模拟输入数据（已经过Feature Projection）
    batch_size = 4
    L_t, L_a, L_v = 50, 100, 75
    d_model = 128
    
    text_proj = torch.randn(batch_size, L_t, d_model)
    audio_proj = torch.randn(batch_size, L_a, d_model)
    video_proj = torch.randn(batch_size, L_v, d_model)
    
    print("=" * 80)
    print("解耦编码器 V1 测试 (原始版本 - 基于Linear)")
    print("=" * 80)
    print("\n输入形状（已投影）:")
    print(f"Text: {text_proj.shape}")
    print(f"Audio: {audio_proj.shape}")
    print(f"Video: {video_proj.shape}")
    print()
    
    # 创建解耦编码器 V1
    decouple_v1 = DecoupleEncoderV1(
        d_model=128,
        shared_dim=256,
        common_dim=128,
        private_dim=128,
        num_modalities=3,
        disc_hidden_dim=256,
        dropout=0.1,
        lambda_grl=1.0
    )
    
    # 前向传播
    decouple_v1.train()
    outputs = decouple_v1(text_proj, audio_proj, video_proj, return_disc=True)
    
    print("解耦后的特征形状:")
    print("\n1. 共享特征 (Shared):")
    print(f"  text_shared:  {outputs['text_shared'].shape}")
    print(f"  audio_shared: {outputs['audio_shared'].shape}")
    print(f"  video_shared: {outputs['video_shared'].shape}")
    
    print("\n2. 通用特征 (Common - 跨模态共享):")
    print(f"  text_common:  {outputs['text_common'].shape}")
    print(f"  audio_common: {outputs['audio_common'].shape}")
    print(f"  video_common: {outputs['video_common'].shape}")
    
    print("\n3. 特有特征 (Private - 模态独有):")
    print(f"  text_private:  {outputs['text_private'].shape}")
    print(f"  audio_private: {outputs['audio_private'].shape}")
    print(f"  video_private: {outputs['video_private'].shape}")
    
    print("\n4. 重构特征:")
    print(f"  重构Shared特征:")
    print(f"    text_shared_recon:  {outputs['text_shared_recon'].shape}")
    print(f"    audio_shared_recon: {outputs['audio_shared_recon'].shape}")
    print(f"    video_shared_recon: {outputs['video_shared_recon'].shape}")
    print(f"  重构原始特征:")
    print(f"    text_recon:  {outputs['text_recon'].shape}")
    print(f"    audio_recon: {outputs['audio_recon'].shape}")
    print(f"    video_recon: {outputs['video_recon'].shape}")
    
    print("\n5. 鉴别器输出:")
    print(f"  disc_logits (对抗): {outputs['disc_logits'].shape}")
    print(f"  text_mod_prob:      {outputs['text_mod_prob'].shape}")
    print(f"  audio_mod_prob:     {outputs['audio_mod_prob'].shape}")
    print(f"  video_mod_prob:     {outputs['video_mod_prob'].shape}")
    
    # 验证维度
    assert outputs['text_shared'].shape == (batch_size, L_t, 256)
    assert outputs['text_common'].shape == (batch_size, L_t, 128)
    assert outputs['text_private'].shape == (batch_size, L_t, 128)
    assert outputs['text_shared_recon'].shape == (batch_size, L_t, 256)
    assert outputs['text_recon'].shape == (batch_size, L_t, 128)
    assert outputs['disc_logits'].shape == (3 * batch_size, 3)
    assert outputs['text_mod_prob'].shape == (batch_size, 1)
    
    # 计算重构误差
    recon_error_text = F.mse_loss(outputs['text_recon'], text_proj)
    recon_error_audio = F.mse_loss(outputs['audio_recon'], audio_proj)
    recon_error_video = F.mse_loss(outputs['video_recon'], video_proj)
    
    print("\n✓ 所有维度检查通过！")
    print(f"\n重构误差 (MSE):")
    print(f"  Text:  {recon_error_text.item():.6f}")
    print(f"  Audio: {recon_error_audio.item():.6f}")
    print(f"  Video: {recon_error_video.item():.6f}")
    
    # 参数统计
    total_params = sum(p.numel() for p in decouple_v1.parameters())
    trainable_params = sum(p.numel() for p in decouple_v1.parameters() if p.requires_grad)
    print(f"\n模型参数:")
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    
    print("\n" + "=" * 80)
    print("V1特点：双层分解架构 (Input → Shared → Common/Private → Recon)")
    print("重构路径：Common+Private → Shared_recon → Input_recon")
    print("对比DMD版本：Conv1d + 单层重构")
    print("=" * 80)
