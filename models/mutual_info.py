"""
多模态互信息约束模块 (Multi-Modal Mutual Information Constraint)

从 AtCAF 项目移植到 H-DCD 架构的核心创新模块之三。

包含两个核心组件:
1. MMILB (Modality Mutual Information Lower Bound):
   - 估计模态间互信息的下界
   - 使用高斯分布建模条件概率 p(y|x)
   - 通过memory机制累积正/负样本，计算熵估计用于训练阶段0
   - 训练阶段0: 最大化lld (maximize mutual information lower bound)
   - 训练阶段1: 最小化lld (作为正则项约束主模型)

2. CPC (Contrastive Predictive Coding):
   - 通过对比学习约束单模态表示与融合表示的一致性
   - 使用反向预测网络G: fusion → modal_pred
   - NCE loss: 鼓励配对的(modal, fusion)相似度高于非配对的

适配说明:
    - 输入均为batch-first格式的特征向量 [B, D]（池化后的序列级特征）
    - MMILB的memory机制需要在trainer中维护
    - CPC不依赖特定架构，可直接使用
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MMILB(nn.Module):
    """
    模态互信息下界 (Modality Mutual Information Lower Bound)
    
    通过高斯分布建模两个模态表示之间的互信息下界。
    
    数学原理:
        给定模态x和模态y的表示，假设条件分布 q(y|x) 是高斯分布:
        q(y|x) = N(μ(x), σ²(x))
        其中 μ(x) = MLP_μ(x), log σ²(x) = MLP_logvar(x)
        
        互信息下界 lld = E[log q(y|x)]
                       = E[-0.5 * (μ(x) - y)² / σ²(x)]
        
        熵估计 H 使用正/负样本的协方差矩阵行列式近似
    
    Args:
        x_size (int): 模态x的特征维度
        y_size (int): 模态y的特征维度
        mid_activation (str): MLP中间层激活函数
        last_activation (str): MLP最后一层激活函数 (用于logvar)
    """
    
    def __init__(self, x_size, y_size, mid_activation='ReLU', last_activation='Tanh'):
        super(MMILB, self).__init__()
        
        try:
            self.mid_activation = getattr(nn, mid_activation)
            self.last_activation = getattr(nn, last_activation)
        except AttributeError:
            raise ValueError(f"激活函数未找到: {mid_activation} 或 {last_activation}")
        
        # μ(x): 预测y的条件均值
        self.mlp_mu = nn.Sequential(
            nn.Linear(x_size, y_size),
            self.mid_activation(),
            nn.Linear(y_size, y_size)
        )
        
        # log σ²(x): 预测y的条件方差的对数
        self.mlp_logvar = nn.Sequential(
            nn.Linear(x_size, y_size),
            self.mid_activation(),
            nn.Linear(y_size, y_size),
        )
        
        # 熵估计投影（将y映射到低维空间以估计熵）
        self.entropy_prj = nn.Sequential(
            nn.Linear(y_size, y_size // 4),
            nn.Tanh()
        )
    
    def forward(self, x, y, labels=None, mem=None):
        """
        前向传播
        
        Args:
            x: 模态x的特征 [B, x_size]
            y: 模态y的特征 [B, y_size]
            labels: 样本标签 [B] 或 [B, 1]，用于正负样本分离（可选）
            mem: 历史memory字典 {'pos': list, 'neg': list}，用于熵估计（可选）
        
        Returns:
            lld: 互信息下界标量 (log-likelihood lower bound)
            sample_dict: {'pos': pos_y, 'neg': neg_y} 正负样本（用于更新memory）
            H: 熵估计值标量
        """
        mu = self.mlp_mu(x)        # [B, y_size]
        logvar = self.mlp_logvar(x)  # [B, y_size]
        
        # 计算互信息下界 lld
        # lld = E[-0.5 * (μ - y)² / σ²]
        positive = -(mu - y) ** 2 / 2.0 / torch.exp(logvar)  # [B, y_size]
        lld = torch.mean(torch.sum(positive, dim=-1))  # 标量
        
        # 正负样本分离与熵估计
        sample_dict = {'pos': None, 'neg': None}
        H = 0.0
        
        if labels is not None:
            # 将y投影到低维空间
            y_proj = self.entropy_prj(y)
            
            # 根据标签分离正负样本
            # 兼容多分类场景: 将标签二分为"积极"和"非积极"
            # - 回归任务: >0 为正, <0 为负 (原始AtCAF逻辑)
            # - 分类任务: 使用前半类vs后半类，或偶数类vs奇数类
            #   这里采用通用策略: 使用中位数划分
            labels_flat = labels.view(-1)
            
            # 自动检测标签类型并划分正/负
            unique_labels = torch.unique(labels_flat)
            if unique_labels.min() < 0:
                # 回归或含负标签: 原始 >0 / <0 划分
                pos_mask = labels_flat > 0
                neg_mask = labels_flat < 0
            elif len(unique_labels) <= 2:
                # 二分类: 使用 ==1 / ==0
                pos_mask = labels_flat == 1
                neg_mask = labels_flat == 0
            else:
                # 多分类(如IEMOCAP 4类, MELD 7类):
                # 使用中位数划分 —— 标签 >= median 为正, < median 为负
                median_label = torch.median(labels_flat.float())
                pos_mask = labels_flat.float() >= median_label
                neg_mask = labels_flat.float() < median_label
            
            pos_y = y_proj[pos_mask] if pos_mask.sum() > 0 else y_proj[:0]
            neg_y = y_proj[neg_mask] if neg_mask.sum() > 0 else y_proj[:0]
            
            sample_dict['pos'] = pos_y
            sample_dict['neg'] = neg_y
            
            # 使用历史memory估计熵
            if mem is not None and mem.get('pos', None) is not None:
                pos_history = mem['pos']
                neg_history = mem['neg']
                
                # 拼接历史和当前样本
                pos_all = torch.cat(pos_history + [pos_y], dim=0) if len(pos_history) > 0 else pos_y
                neg_all = torch.cat(neg_history + [neg_y], dim=0) if len(neg_history) > 0 else neg_y
                
                if pos_all.size(0) > 1 and neg_all.size(0) > 1:
                    # 计算协方差矩阵
                    mu_pos = pos_all.mean(dim=0)
                    mu_neg = neg_all.mean(dim=0)
                    
                    sigma_pos = torch.mean(
                        torch.bmm(
                            (pos_all - mu_pos).unsqueeze(-1),
                            (pos_all - mu_pos).unsqueeze(1)
                        ), dim=0
                    )
                    sigma_neg = torch.mean(
                        torch.bmm(
                            (neg_all - mu_neg).unsqueeze(-1),
                            (neg_all - mu_neg).unsqueeze(1)
                        ), dim=0
                    )
                    
                    # 熵估计: H ≈ 0.25 * (log|Σ_pos| + log|Σ_neg|)
                    H = 0.25 * (torch.logdet(sigma_pos) + torch.logdet(sigma_neg))
        
        return lld, sample_dict, H


class CPC(nn.Module):
    """
    对比预测编码 (Contrastive Predictive Coding)
    
    通过对比学习约束单模态表示与融合表示之间的一致性。
    
    数学原理:
        给定模态表示 h_m 和融合表示 z:
        1. 使用网络G将z投影到h_m空间: x_pred = G(z)
        2. 归一化 x_pred 和 h_m
        3. 计算NCE损失:
           正样本对: pos = Σ(h_m_i · x_pred_i)
           负样本对: neg = log Σ_j exp(h_m_i · x_pred_j)
           NCE = -(pos - neg).mean()
    
    Args:
        x_size (int): 单模态特征维度（目标预测维度）
        y_size (int): 融合特征维度（输入维度）
        n_layers (int): 预测网络G的层数
        activation (str): 激活函数类型
    """
    
    def __init__(self, x_size, y_size, n_layers=1, activation='Tanh'):
        super(CPC, self).__init__()
        
        self.x_size = x_size
        self.y_size = y_size
        
        activation_fn = getattr(nn, activation)
        
        # 预测网络G: 从融合表示反向预测单模态表示
        if n_layers == 1:
            self.net = nn.Linear(y_size, x_size)
        else:
            layers = []
            for i in range(n_layers):
                if i == 0:
                    layers.append(nn.Linear(y_size, x_size))
                    layers.append(activation_fn())
                else:
                    layers.append(nn.Linear(x_size, x_size))
            self.net = nn.Sequential(*layers)
    
    def forward(self, x, y):
        """
        计算NCE loss
        
        Args:
            x: 单模态特征 [B, x_size] (例如: text/audio/video)
            y: 融合特征 [B, y_size] (例如: 融合后的表示)
        
        Returns:
            nce: NCE loss标量（越小越好，表示x和y越一致）
        """
        # 从融合表示预测单模态表示
        x_pred = self.net(y)  # [B, x_size]
        
        # 归一化到单位球面
        x_pred = F.normalize(x_pred, p=2, dim=1, eps=1e-8)
        x = F.normalize(x, p=2, dim=1, eps=1e-8)
        
        # 正样本对: 对角线上的相似度
        pos = torch.sum(x * x_pred, dim=-1)  # [B]
        
        # 负样本对: 所有配对的logsumexp
        neg = torch.logsumexp(torch.matmul(x, x_pred.t()), dim=-1)  # [B]
        
        # NCE loss
        nce = -(pos - neg).mean()
        
        return nce


class MutualInfoConstraint(nn.Module):
    """
    多模态互信息约束框架（整合MMILB和CPC）
    
    为H-DCD提供完整的互信息约束:
    - MMILB用于模态间互信息估计（text-visual, text-audio）
    - CPC用于单模态-融合一致性约束（text-fusion, audio-fusion, video-fusion）
    
    Args:
        d_text (int): 文本特征维度
        d_audio (int): 音频特征维度
        d_video (int): 视频特征维度
        d_fusion (int): 融合特征维度
        mmilb_activation (str): MMILB中间层激活函数
        cpc_layers (int): CPC预测网络层数
        cpc_activation (str): CPC激活函数
        add_va (bool): 是否添加 visual-audio MMILB
    """
    
    def __init__(
        self,
        d_text: int = 128,
        d_audio: int = 128,
        d_video: int = 128,
        d_fusion: int = 128,
        mmilb_activation: str = 'ReLU',
        cpc_layers: int = 1,
        cpc_activation: str = 'Tanh',
        add_va: bool = True,
    ):
        super(MutualInfoConstraint, self).__init__()
        
        self.add_va = add_va
        
        # ====================================================================
        # MMILB: 模态间互信息下界估计
        # text→visual, text→audio, (可选)visual→audio
        # ====================================================================
        self.mi_tv = MMILB(
            x_size=d_text,
            y_size=d_video,
            mid_activation=mmilb_activation,
        )
        self.mi_ta = MMILB(
            x_size=d_text,
            y_size=d_audio,
            mid_activation=mmilb_activation,
        )
        if add_va:
            self.mi_va = MMILB(
                x_size=d_video,
                y_size=d_audio,
                mid_activation=mmilb_activation,
            )
        
        # ====================================================================
        # CPC: 单模态-融合一致性约束
        # text←fusion, audio←fusion, video←fusion
        # ====================================================================
        self.cpc_text = CPC(
            x_size=d_text,
            y_size=d_fusion,
            n_layers=cpc_layers,
            activation=cpc_activation,
        )
        self.cpc_audio = CPC(
            x_size=d_audio,
            y_size=d_fusion,
            n_layers=cpc_layers,
            activation=cpc_activation,
        )
        self.cpc_video = CPC(
            x_size=d_video,
            y_size=d_fusion,
            n_layers=cpc_layers,
            activation=cpc_activation,
        )
    
    def compute_mmilb(self, text_feat, audio_feat, video_feat, 
                      labels=None, mem=None):
        """
        计算MMILB损失
        
        Args:
            text_feat:  [B, d_text] 池化后的文本特征
            audio_feat: [B, d_audio] 池化后的音频特征
            video_feat: [B, d_video] 池化后的视频特征
            labels:     [B] 标签（用于正负样本分离）
            mem:        memory字典 {'tv': {'pos':..., 'neg':...}, 'ta':..., 'va':...}
        
        Returns:
            lld: 总互信息下界（lld_tv + lld_ta + lld_va）
            H: 总熵估计
            pn_dic: 正负样本字典（用于更新memory）
        """
        mem_tv = mem.get('tv', None) if mem else None
        mem_ta = mem.get('ta', None) if mem else None
        mem_va = mem.get('va', None) if mem else None
        
        lld_tv, tv_pn, H_tv = self.mi_tv(text_feat, video_feat, labels, mem_tv)
        lld_ta, ta_pn, H_ta = self.mi_ta(text_feat, audio_feat, labels, mem_ta)
        
        if self.add_va:
            lld_va, va_pn, H_va = self.mi_va(video_feat, audio_feat, labels, mem_va)
        else:
            lld_va, va_pn, H_va = 0.0, {'pos': None, 'neg': None}, 0.0
        
        lld = lld_tv + lld_ta + (lld_va if isinstance(lld_va, torch.Tensor) else 0.0)
        H = H_tv + H_ta + H_va
        pn_dic = {'tv': tv_pn, 'ta': ta_pn, 'va': va_pn}
        
        return lld, H, pn_dic
    
    def compute_cpc(self, text_feat, audio_feat, video_feat, fusion_feat):
        """
        计算CPC损失
        
        Args:
            text_feat:   [B, d_text]
            audio_feat:  [B, d_audio]
            video_feat:  [B, d_video]
            fusion_feat: [B, d_fusion]
        
        Returns:
            nce: 总NCE损失 (nce_text + nce_audio + nce_video)
        """
        nce_text = self.cpc_text(text_feat, fusion_feat)
        nce_audio = self.cpc_audio(audio_feat, fusion_feat)
        nce_video = self.cpc_video(video_feat, fusion_feat)
        
        nce = nce_text + nce_audio + nce_video
        
        return nce