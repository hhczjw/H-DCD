"""
H-DCD — Hierarchical Decoupled Contrastive Distillation
分层解耦对比蒸馏模型（主模型）

============================================================================
整体架构说明
============================================================================
H-DCD 是一个完全基于 Mamba2 的因果感知多模态情感分析框架。
核心设计哲学: "选择性即因果性" (Selectivity as Causality)

整合所有子模块:
    1. Feature Projection (TextProjection + AudioVideoProjection)
       功能: 将不同维度的原始特征统一映射到 d_model 维空间
    2. [创新1] SS-CD (State-Space Causal Debiasing)
       功能: 通过双向 Mamba2 + 条件 Mamba2 实现前门调整因果去偏
    3. Decouple Encoder (GRL 对抗解耦)
       功能: 将模态特征解耦为 Specific(模态特有) 和 Common(共享) 部分
    4. HMNF (CoupledHMNF — 异构多模态融合流)
       功能: 使用耦合双向 Mamba2 实现三模态深度交互融合
    5. HMPN (同构感知网络流)
       功能: 单模态 Mamba2 精炼 + 跨模态 MHA→Mamba2 增强 + 超融合
    6. [创新2] SCI (Selective Counterfactual Inference)
       功能: 双通道 Mamba2 实现反事实推断，量化跨模态因果效应
    7. [创新3] MutualInfoConstraint (MMILB + CPC)
       功能: 互信息约束框架 (纯 MLP, 架构无关)
    8. Hierarchical Classifiers (分层分类器)
       功能: 单模态/双模态/全模态分层分类

数据流:
    x_text/x_audio/x_video [B, L, D_raw]
      → 特征投影 → [B, L, d_model]
      → [创新1] SS-CD 因果去偏 → [B, L, d_model]
      → 解耦编码 → Specific + Common
      → Common → 双流融合 (HMNF 并行 HMPN)
      → [创新2] SCI 反事实推断 (训练时)
      → 分层分类 → logits
      → [创新3] 互信息约束损失 (训练时)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import sys
import os

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(__file__))

# 导入原有子模块
from feature_projection import TextProjection, AudioVideoProjection
from decouple_encoder import DecoupleEncoder
from hmnf import CoupledHMNF
from hmpn import HMPN

# === 导入 Mamba2 化的创新模块 ===
from causal_debias import MultiModalDebiasWrapper
from counterfactual_attention import CounterfactualCrossAttention
from mutual_info import MutualInfoConstraint


class H_DCD(nn.Module):
    """
    H-DCD 主模型 (完全基于 Mamba2 的因果感知多模态融合)

    完整架构:
        特征投影 → [创新1] SS-CD → 解耦编码 → 双流融合 (HMNF + HMPN)
        → [创新2] SCI → 分层分类 → [创新3] 互信息约束

    Args:
        text_input_dim (int): 文本输入维度 (如 768 for BERT)
        audio_input_dim (int): 音频输入维度 (如 74)
        video_input_dim (int): 视频输入维度 (如 35)
        d_model (int): 统一特征维度
        text_hidden_dim (int): 文本 BiGRU 隐层维度
        text_num_layers (int): 文本 BiGRU 层数
        decouple_disc_hidden (int): 解耦判别器隐层维度
        decouple_lambda_grl (float): GRL 梯度反转系数
        hmnf_d_state/d_conv/expand/num_layers: HMNF Mamba2 参数
        hmpn_d_state/d_conv/expand/num_heads: HMPN Mamba2 参数
        num_classes (int): 分类类别数
        dropout (float): Dropout 概率
        --- 创新点参数 ---
        use_causal_debias (bool): 是否启用 SS-CD 因果去偏
        debias_num_layers (int): SS-CD 双路径的 Mamba2 层数
        debias_confounder_size (int): 混杂因子字典大小
        debias_d_state (int): SS-CD Mamba2 状态维度
        debias_headdim (int): SS-CD Mamba2 头维度
        debias_text/audio/video (bool): 是否对各模态去偏
        confounder_npy_dir (str): KMeans 字典目录
        dataset_name (str): 数据集名称
        use_counterfactual (bool): 是否启用 SCI 反事实推断
        counterfactual_type (str): 反事实策略
        counterfactual_num_layers (int): SCI Mamba2 层数
        counterfactual_d_state (int): SCI Mamba2 状态维度
        counterfactual_headdim (int): SCI Mamba2 头维度
        use_mutual_info (bool): 是否启用互信息约束
        add_va_mi (bool): 是否添加 visual-audio MMILB
        cpc_layers (int): CPC 预测网络层数
    """

    def __init__(
        self,
        # === 输入维度 ===
        text_input_dim: int = 768,
        audio_input_dim: int = 74,
        video_input_dim: int = 35,
        # === 统一维度 ===
        d_model: int = 128,
        # === 文本投影参数 ===
        text_hidden_dim: int = 256,
        text_num_layers: int = 2,
        # === Decouple Encoder 参数 ===
        decouple_disc_hidden: int = 256,
        decouple_lambda_grl: float = 1.0,
        # === HMNF 参数 (CoupledHMNF) ===
        hmnf_d_state: int = 64,
        hmnf_d_conv: int = 4,
        hmnf_expand: int = 2,
        hmnf_num_layers: int = 1,
        # === HMPN 参数 ===
        hmpn_d_state: int = 64,
        hmpn_d_conv: int = 4,
        hmpn_expand: int = 2,
        hmpn_num_heads: int = 4,
        # === 分类参数 ===
        num_classes: int = 4,
        dropout: float = 0.1,
        # === [创新1] SS-CD 因果去偏参数 ===
        use_causal_debias: bool = True,
        debias_num_layers: int = 2,
        debias_confounder_size: int = 50,
        debias_d_state: int = 64,
        debias_headdim: int = 32,
        debias_text: bool = True,
        debias_audio: bool = True,
        debias_video: bool = True,
        confounder_npy_dir: str = None,
        dataset_name: str = 'mosi',
        # === [创新2] SCI 反事实推断参数 ===
        use_counterfactual: bool = True,
        counterfactual_type: str = 'shuffle',
        counterfactual_num_layers: int = 2,
        counterfactual_d_state: int = 64,
        counterfactual_headdim: int = 32,
        # === [创新3] 互信息约束参数 ===
        use_mutual_info: bool = True,
        add_va_mi: bool = True,
        cpc_layers: int = 1,
    ):
        super(H_DCD, self).__init__()

        self.d_model = d_model
        self.num_classes = num_classes

        # === 创新点开关 ===
        self.use_causal_debias = use_causal_debias
        self.use_counterfactual = use_counterfactual
        self.use_mutual_info = use_mutual_info
        self.counterfactual_type = counterfactual_type

        # ====================================================================
        # 1. 特征投影层 (Feature Projection)
        # ====================================================================
        self.text_projection = TextProjection(
            input_dim=text_input_dim, hidden_dim=text_hidden_dim,
            output_dim=d_model, num_layers=text_num_layers, dropout=dropout,
        )
        self.audio_projection = AudioVideoProjection(
            input_dim=audio_input_dim, output_dim=d_model, dropout=dropout,
        )
        self.video_projection = AudioVideoProjection(
            input_dim=video_input_dim, output_dim=d_model, dropout=dropout,
        )

        # ====================================================================
        # [创新1] SS-CD 因果去偏 (State-Space Causal Debiasing)
        # 位置: 特征投影之后、解耦编码之前
        # 功能: 通过 BiMamba2 + ConditionalMamba2 双路径前门调整去偏
        # ====================================================================
        if use_causal_debias:
            self.causal_debias = MultiModalDebiasWrapper(
                d_model=d_model,
                num_layers=debias_num_layers,
                confounder_size=debias_confounder_size,
                d_state=debias_d_state,
                d_conv=4,
                expand=2,
                dropout=dropout,
                headdim=debias_headdim,
                debias_text=debias_text,
                debias_audio=debias_audio,
                debias_video=debias_video,
                confounder_npy_dir=confounder_npy_dir,
                dataset_name=dataset_name,
            )

        # ====================================================================
        # 2. 解耦编码器 (Decouple Encoder)
        # ====================================================================
        self.decouple_encoder = DecoupleEncoder(
            d_model=d_model, num_modalities=3,
            disc_hidden_dim=decouple_disc_hidden, dropout=dropout,
            lambda_grl=decouple_lambda_grl,
        )

        # ====================================================================
        # 3. 双流融合网络
        # ====================================================================
        # 流A: HMNF (异构多模态融合)
        self.hmnf = CoupledHMNF(
            d_model=d_model, num_layers=hmnf_num_layers,
            d_state=hmnf_d_state, d_conv=hmnf_d_conv,
            expand=hmnf_expand, dropout=dropout,
        )
        self.hmnf_fusion = nn.Sequential(
            nn.Linear(3 * d_model, d_model), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(d_model, d_model),
        )

        # 流B: HMPN (同构感知网络)
        self.hmpn = HMPN(
            d_model=d_model, d_state=hmpn_d_state,
            d_conv=hmpn_d_conv, expand=hmpn_expand,
            headdim=32, ngroups=1, num_heads=hmpn_num_heads,
        )

        # ====================================================================
        # [创新2] SCI 反事实推断 (Selective Counterfactual Inference)
        # 位置: HMPN 融合之后，分类之前
        # 功能: 双通道 Mamba2 构建反事实分支，量化跨模态因果效应
        # ====================================================================
        if use_counterfactual:
            # text←audio 反事实跨模态 Mamba2
            self.cf_attn_ta = CounterfactualCrossAttention(
                d_model=d_model,
                num_layers=counterfactual_num_layers,
                d_state=counterfactual_d_state,
                dim_feedforward=4 * d_model,
                dropout=dropout,
                headdim=counterfactual_headdim,
            )
            # text←video 反事实跨模态 Mamba2
            self.cf_attn_tv = CounterfactualCrossAttention(
                d_model=d_model,
                num_layers=counterfactual_num_layers,
                d_state=counterfactual_d_state,
                dim_feedforward=4 * d_model,
                dropout=dropout,
                headdim=counterfactual_headdim,
            )
            # 反事实融合 MLP
            self.cf_fusion_mlp = nn.Sequential(
                nn.Linear(2 * d_model, d_model), nn.Tanh(),
                nn.Dropout(dropout), nn.Linear(d_model, d_model),
            )
            # 反事实分类头
            self.cf_head = nn.Sequential(
                nn.Linear(d_model, d_model // 2), nn.ReLU(),
                nn.Dropout(dropout), nn.Linear(d_model // 2, num_classes),
            )

        # ====================================================================
        # [创新3] 互信息约束 (MMILB + CPC)
        # 功能: 训练时计算互信息相关损失，架构无关
        # ====================================================================
        if use_mutual_info:
            self.mutual_info = MutualInfoConstraint(
                d_text=d_model, d_audio=d_model, d_video=d_model,
                d_fusion=d_model, mmilb_activation='ReLU',
                cpc_layers=cpc_layers, cpc_activation='Tanh',
                add_va=add_va_mi,
            )

        # ====================================================================
        # 4. 分层分类器 (Hierarchical Classifiers)
        # ====================================================================
        self.pooling = nn.AdaptiveAvgPool1d(1)

        # 单模态分类头
        self.head_uni_text = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(d_model // 2, num_classes),
        )
        self.head_uni_audio = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(d_model // 2, num_classes),
        )
        self.head_uni_video = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(d_model // 2, num_classes),
        )

        # 双模态分类头
        self.head_bi_ta = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(d_model // 2, num_classes),
        )
        self.head_bi_tv = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(d_model // 2, num_classes),
        )
        self.head_bi_av = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(d_model // 2, num_classes),
        )

        # 全模态分类头
        self.fusion_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.Tanh(),
        )
        self.head_multi = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(d_model // 2, num_classes),
        )

    def forward(
        self,
        x_text: torch.Tensor,
        x_audio: torch.Tensor,
        x_video: torch.Tensor,
        return_all: bool = True,
        labels: torch.Tensor = None,
        mem: dict = None,
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            x_text:  文本特征 [B, L_t, 768]
            x_audio: 音频特征 [B, L_a, 74]
            x_video: 视频特征 [B, L_v, 35]
            return_all: 是否返回所有中间结果 (训练时需要)
            labels: 样本标签 [B] (用于 MMILB 正负样本分离)
            mem: MMILB 的 memory 字典 (用于熵估计)

        Returns:
            字典包含所有输出 (logits, features, 创新点输出等)
        """
        batch_size = x_text.size(0)

        # ====================================================================
        # 步骤1: 特征投影 (Feature Projection)
        # 将不同维度的原始特征统一映射到 d_model 维空间
        # ====================================================================
        X_t = self.text_projection(x_text)    # [B, L_t, d_model]
        X_a = self.audio_projection(x_audio)  # [B, L_a, d_model]
        X_v = self.video_projection(x_video)  # [B, L_v, d_model]

        # ====================================================================
        # [创新1] 步骤1.5: SS-CD 因果去偏
        # 在投影后、解耦前，对每个模态执行双路径因果去偏
        # IS路径: BiMamba2 捕获时序依赖; CS路径: ConditionalMamba2 边缘化混杂因子
        # ====================================================================
        if self.use_causal_debias:
            X_t, X_a, X_v = self.causal_debias(X_t, X_a, X_v)

        # ====================================================================
        # 步骤2: 解耦编码 (Decouple Encoder)
        # GRL 对抗训练将特征解耦为模态特有 (Specific) 和共享 (Common) 部分
        # ====================================================================
        decouple_outputs = self.decouple_encoder(
            X_t, X_a, X_v, return_disc=self.training,
        )

        s_text = decouple_outputs['s_text']
        s_audio = decouple_outputs['s_audio']
        s_video = decouple_outputs['s_video']
        c_text = decouple_outputs['c_text']
        c_audio = decouple_outputs['c_audio']
        c_video = decouple_outputs['c_video']

        # ====================================================================
        # 步骤3: 双流并行处理
        # 先对齐序列长度，再送入 HMNF (异构融合) 和 HMPN (同构感知)
        # ====================================================================
        min_len = min(c_text.size(1), c_audio.size(1), c_video.size(1))

        def align_seq(x, length):
            """自适应平均池化对齐序列长度"""
            if x.size(1) == length:
                return x
            return F.adaptive_avg_pool1d(
                x.transpose(1, 2), length
            ).transpose(1, 2)

        c_text_aligned = align_seq(c_text, min_len)
        c_audio_aligned = align_seq(c_audio, min_len)
        c_video_aligned = align_seq(c_video, min_len)

        # --- 流A: HMNF (异构多模态 Mamba2 融合) ---
        hmnf_out_a, hmnf_out_v, hmnf_out_l = self.hmnf(
            c_audio_aligned, c_video_aligned, c_text_aligned,
        )
        hmnf_pool_a = torch.mean(hmnf_out_a, dim=1)  # [B, d_model]
        hmnf_pool_v = torch.mean(hmnf_out_v, dim=1)  # [B, d_model]
        hmnf_pool_l = torch.mean(hmnf_out_l, dim=1)  # [B, d_model]
        hmnf_fused_feat = self.hmnf_fusion(
            torch.cat([hmnf_pool_a, hmnf_pool_v, hmnf_pool_l], dim=-1)
        )  # [B, d_model]

        # --- 流B: HMPN (同构 Mamba2 感知网络) ---
        hmpn_final = self.hmpn(
            c_text_aligned, c_audio_aligned, c_video_aligned
        )  # [B, d_model]

        # ====================================================================
        # [创新2] 步骤3.5: SCI 反事实推断
        # 训练时: 双通道 Mamba2 生成 factual + counterfactual 融合特征
        # 因果效应 = factual_fusion - counterfactual_fusion
        # ====================================================================
        counterfactual_preds = None
        counterfactual_fusion = None

        if self.use_counterfactual and self.training:
            # text←audio: 反事实 Mamba2 跨模态扫描
            # 返回 (factual_out, counterfactual_out)
            factual_ta, cf_ta = self.cf_attn_ta(
                c_text_aligned, c_audio_aligned,
                counterfactual_type=self.counterfactual_type,
            )  # 各 [B, min_len, d_model]

            # text←video: 反事实 Mamba2 跨模态扫描
            factual_tv, cf_tv = self.cf_attn_tv(
                c_text_aligned, c_video_aligned,
                counterfactual_type=self.counterfactual_type,
            )  # 各 [B, min_len, d_model]

            # 反事实通道: 池化 + 拼接 + MLP 融合
            if cf_ta is not None and cf_tv is not None:
                cf_ta_pooled = torch.mean(cf_ta, dim=1)  # [B, d_model]
                cf_tv_pooled = torch.mean(cf_tv, dim=1)  # [B, d_model]
                counterfactual_fusion = self.cf_fusion_mlp(
                    torch.cat([cf_ta_pooled, cf_tv_pooled], dim=-1)
                )  # [B, d_model]

                # 反事实预测 (用于计算反事实损失)
                counterfactual_preds = self.cf_head(counterfactual_fusion)

        # ====================================================================
        # 步骤4: 分层分类
        # ====================================================================
        # 4.1 单模态分类 (使用未对齐的 common 特征)
        c_text_pooled = self.pooling(c_text.transpose(1, 2)).squeeze(-1)    # [B, d_model]
        c_audio_pooled = self.pooling(c_audio.transpose(1, 2)).squeeze(-1)  # [B, d_model]
        c_video_pooled = self.pooling(c_video.transpose(1, 2)).squeeze(-1)  # [B, d_model]

        pred_uni_text = self.head_uni_text(c_text_pooled)    # [B, num_classes]
        pred_uni_audio = self.head_uni_audio(c_audio_pooled)  # [B, num_classes]
        pred_uni_video = self.head_uni_video(c_video_pooled)  # [B, num_classes]

        # 4.2 双模态分类 (使用 HMNF 池化特征相加)
        pred_bi_ta = self.head_bi_ta(hmnf_pool_l + hmnf_pool_a)  # [B, num_classes]
        pred_bi_tv = self.head_bi_tv(hmnf_pool_l + hmnf_pool_v)  # [B, num_classes]
        pred_bi_av = self.head_bi_av(hmnf_pool_a + hmnf_pool_v)  # [B, num_classes]

        # 4.3 全模态分类 (融合 HMNF 和 HMPN 双流)
        concat_feat = torch.cat([hmnf_fused_feat, hmpn_final], dim=-1)  # [B, 2*d_model]
        fused_final = self.fusion_gate(concat_feat)  # [B, d_model]

        # [创新2] 因果效应: 真实融合 - 反事实融合
        if self.use_counterfactual and self.training and counterfactual_fusion is not None:
            fused_final_causal = fused_final - counterfactual_fusion
        else:
            fused_final_causal = fused_final

        pred_multi = self.head_multi(fused_final_causal)  # [B, num_classes]

        # ====================================================================
        # [创新3] 步骤4.5: 互信息约束计算 (训练时)
        # MMILB: 估计模态间互信息下界
        # CPC: 对比预测编码约束单模态-融合一致性
        # ====================================================================
        mi_outputs = {}
        if self.use_mutual_info and self.training:
            lld, H, pn_dic = self.mutual_info.compute_mmilb(
                c_text_pooled, c_audio_pooled, c_video_pooled,
                labels=labels, mem=mem,
            )
            nce = self.mutual_info.compute_cpc(
                c_text_pooled, c_audio_pooled, c_video_pooled,
                fused_final,
            )
            mi_outputs = {
                'lld': lld,        # 互信息下界
                'nce': nce,        # NCE 对比损失
                'H': H,            # 熵估计
                'pn_dic': pn_dic,  # 正负样本字典
            }

        # ====================================================================
        # 步骤5: 组织输出
        # ====================================================================
        outputs = {
            'logits_uni': [pred_uni_text, pred_uni_audio, pred_uni_video],
            'logits_bi': [pred_bi_ta, pred_bi_tv, pred_bi_av],
            'logits_multi': pred_multi,
            'features_contrast': {
                'hmnf': hmnf_fused_feat,
                'hmpn': hmpn_final,
            },
        }

        if return_all:
            outputs.update({
                'decouple_items': {
                    's_text': s_text, 's_audio': s_audio, 's_video': s_video,
                    'c_text': c_text, 'c_audio': c_audio, 'c_video': c_video,
                    'recon_text': decouple_outputs['recon_text'],
                    'recon_audio': decouple_outputs['recon_audio'],
                    'recon_video': decouple_outputs['recon_video'],
                    'original_text': X_t, 'original_audio': X_a, 'original_video': X_v,
                },
                'adv_logits': decouple_outputs.get('disc_logits', None),
                # === 创新点输出 ===
                'counterfactual_preds': counterfactual_preds,
                'counterfactual_fusion': counterfactual_fusion,
                'mi_outputs': mi_outputs,
            })

        return outputs