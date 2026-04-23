"""
Loss functions for H-DCD model

Complete implementation of the four-module loss system:
1. L_dec: Adversarial Disentanglement Loss (对抗性解耦损失)
2. L_hierarchical: Hierarchical Deep Supervision Loss (分层监督损失)
3. L_distill: Contrastive Knowledge Distillation Loss (对比知识蒸馏损失)
4. L_task: Main Task Loss (最终任务损失)

Total Loss: L_TOTAL = L_task + λ_dec·L_dec + λ_hierarchical·L_hierarchical + λ_distill·L_distill
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class H_DCD_Loss(nn.Module):
    """
    Composite Multi-Task Loss for H-DCD
    复合多任务损失函数
    
    整合四个核心学习任务:
    1. L_dec (对抗性解耦损失): 特征净化引擎
       - L_rec: 重构损失
       - L_adv: 对抗损失
       - L_mar': 情感感知的间隔损失
       - L_ort: 正交损失
    
    2. L_hierarchical (分层监督损失): 训练稳定器
       - L_yuni: 单模态级监督
       - L_ybi: 双模态级监督
       - L_ymul: 三模态级监督
    
    3. L_distill (对比知识蒸馏损失): 协同进化引擎
       - KL散度对比学习
    
    4. L_task (最终任务损失): 最终使命
       - 主任务预测损失
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.task_type = args.get('task_type', 'classification')
        
        # ================================================================
        # 总损失权重 (Grand Objective Weights)
        # ================================================================
        self.lambda_dec = args.get('lambda_dec', 0.5)              # L_dec 权重
        self.lambda_hierarchical = args.get('lambda_hierarchical', 1.0)  # L_hierarchical 权重
        self.lambda_distill = args.get('lambda_distill', 0.5)      # L_distill 权重
        
        # ================================================================
        # === AtCAF 创新点损失权重 ===
        # ================================================================
        # [创新2] 反事实效应损失权重 (η in AtCAF)
        self.lambda_counterfactual = args.get('lambda_counterfactual', 0.5)
        # [创新3] 互信息约束损失权重
        self.alpha_nce = args.get('alpha_nce', 0.1)    # CPC NCE损失权重 (α in AtCAF)
        self.beta_lld = args.get('beta_lld', 0.1)      # MMILB lld损失权重 (β in AtCAF)
        
        # ================================================================
        # L_dec 子损失权重 (Decouple Loss Sub-weights)
        # ================================================================
        self.lambda_adv = args.get('lambda_adv', 0.1)              # 对抗损失权重
        self.gamma_mar = args.get('gamma_mar', 0.1)                # 间隔损失权重
        self.gamma_ort = args.get('gamma_ort', 0.01)               # 正交损失权重
        # L_rec 权重为1.0 (基准)
        
        # ================================================================
        # L_hierarchical 子损失权重 (Hierarchical Loss Sub-weights)
        # ================================================================
        self.w_uni = args.get('w_uni', 1.0)                        # 单模态级权重
        self.w_bi = args.get('w_bi', 1.0)                          # 双模态级权重
        self.w_mul = args.get('w_mul', 1.0)                        # 三模态级权重
        
        # ================================================================
        # L_mar' 动态间隔参数 (Dynamic Margin Parameters)
        # ================================================================
        self.alpha_base = args.get('alpha_base', 0.2)              # 基础间隔
        self.beta = args.get('beta', 0.5)                          # 动态调节系数
        
        # VA空间距离矩阵 (预计算，根据数据集)
        self.register_buffer('va_distances', self._init_va_distances(args))
        
        # ================================================================
        # L_distill 温度参数 (Distillation Temperature)
        # ================================================================
        self.tau = args.get('tau', 2.0)                            # 温度系数
        
        # ================================================================
        # 任务损失函数 (Task-specific Loss Criterion)
        # ================================================================
        if self.task_type == 'classification':
            self.task_criterion = nn.CrossEntropyLoss()
        else:  # regression
            self.task_criterion = nn.L1Loss()  # MAE for regression

        # ================================================================
        # [P0-2] 不确定性加权多任务损失 (Kendall et al., 2018)
        # 每个损失项学习一个 log(σ²) 参数, 总损失为:
        #     Σ_i [ exp(-log_σ²_i) * L_i + 0.5 * log_σ²_i ]
        # 等价于自适应权重 1/(2σ²_i), 数值大的 L_i 会自动降低权重,
        # 解决 6+ 项损失量级差异大、手工调参困难的问题.
        # ================================================================
        self.use_uncertainty_weighting = args.get('use_uncertainty_weighting', True)
        if self.use_uncertainty_weighting:
            # 管理的损失项: L_task, L_dec, L_hier, L_distill, L_cf, L_nce, L_lld(注意符号)
            # 初始 log_sigma²=0 → σ²=1 → 权重=1, 与等权起点一致
            self.log_vars = nn.ParameterDict({
                'task':    nn.Parameter(torch.zeros(1)),
                'dec':     nn.Parameter(torch.zeros(1)),
                'hier':    nn.Parameter(torch.zeros(1)),
                'distill': nn.Parameter(torch.zeros(1)),
                'cf':      nn.Parameter(torch.zeros(1)),
                'nce':     nn.Parameter(torch.zeros(1)),
                'lld':     nn.Parameter(torch.zeros(1)),
            })
    
    def _init_va_distances(self, args):
        """
        初始化 Valence-Arousal 空间的情感距离矩阵
        
        根据数据集的情感类别，预计算它们在VA空间中的欧几里得距离。
        这个距离矩阵用于 L_mar' 中的动态间隔计算。
        
        Returns:
            va_distances: [num_classes, num_classes] 距离矩阵
        """
        num_classes = args.get('num_classes', 4)
        dataset_name = args.get('dataset_name', '').lower()
        
        # 不同数据集的VA坐标定义 (Valence, Arousal)
        # 参考文献: Russell's Circumplex Model of Affect
        if dataset_name == 'iemocap' or num_classes == 4:
            # IEMOCAP: happy, sad, angry, neutral
            va_coords = torch.tensor([
                [0.8, 0.6],   # happy: high valence, high arousal
                [-0.6, -0.4], # sad: low valence, low arousal
                [-0.7, 0.7],  # angry: low valence, high arousal
                [0.0, 0.0]    # neutral: middle
            ])
        elif dataset_name == 'meld' or num_classes == 7:
            # MELD: neutral, joy, sadness, anger, surprise, fear, disgust
            va_coords = torch.tensor([
                [0.0, 0.0],   # neutral
                [0.8, 0.6],   # joy
                [-0.6, -0.4], # sadness
                [-0.7, 0.7],  # anger
                [0.3, 0.8],   # surprise
                [-0.6, 0.5],  # fear
                [-0.7, 0.3]   # disgust
            ])
        else:
            # 默认: 均匀分布在单位圆上
            angles = torch.linspace(0, 2*np.pi, num_classes+1)[:-1]
            va_coords = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)
        
        # 计算成对距离矩阵 D_VA(c_i, c_k)
        # 使用欧几里得距离
        num_classes = va_coords.size(0)
        distances = torch.zeros(num_classes, num_classes)
        for i in range(num_classes):
            for k in range(num_classes):
                distances[i, k] = torch.norm(va_coords[i] - va_coords[k], p=2)
        
        return distances
    
    def forward(self, outputs, labels):
        """
        计算总损失
        
        Args:
            outputs: dict from H_DCD forward, 包含:
                - logits_uni: dict {'text': [B, C], 'audio': [B, C], 'video': [B, C]}
                - logits_bi: dict {'ta': [B, C], 'tv': [B, C], 'av': [B, C]}
                - logits_multi: [B, C] 最终预测
                - features_contrast: dict {'hmnf': [B, d], 'hmpn': [B, d]}
                - decouple_items: dict 包含解耦相关的所有中间变量
                - adv_logits: [B, 3] 模态判别器输出
            labels: [B] for classification or [B, 1] for regression
        
        Returns:
            total_loss: scalar tensor
            loss_dict: dict of individual losses for logging
        """
        device = labels.device
        
        # Prepare labels
        if self.task_type == 'classification':
            if labels.dim() > 1:
                labels = labels.squeeze(-1)
            labels = labels.long()
        else:  # regression
            if labels.dim() == 1:
                labels = labels.unsqueeze(-1)
            labels = labels.float()
        
        loss_dict = {}
        
        # ================================================================
        # 模块一: L_dec (对抗性解耦损失)
        # Adversarial Disentanglement Loss
        # ================================================================
        loss_dec, dec_dict = self._compute_L_dec(outputs, labels, device)
        loss_dict.update(dec_dict)
        
        # ================================================================
        # 模块二: L_hierarchical (分层监督损失)
        # Hierarchical Deep Supervision Loss
        # ================================================================
        loss_hierarchical, hier_dict = self._compute_L_hierarchical(outputs, labels)
        loss_dict.update(hier_dict)
        
        # ================================================================
        # 模块三: L_distill (对比知识蒸馏损失)
        # Contrastive Knowledge Distillation Loss
        # ================================================================
        loss_distill, distill_dict = self._compute_L_distill(outputs)
        loss_dict.update(distill_dict)
        
        # ================================================================
        # 模块四: L_task (最终任务损失)
        # Main Task Loss
        # ================================================================
        loss_task = self.task_criterion(outputs['logits_multi'], labels)
        loss_dict['L_task'] = loss_task.item()
        
        # ================================================================
        # === [创新2] L_counterfactual (因果效应损失) ===
        # 对齐 AtCAF 原始设计: 在预测值层面计算因果效应
        # L_cf = criterion(pred_multi - counterfactual_preds, labels)
        # 含义: 真实预测 - 反事实预测 = 因果效应, 因果效应应能正确预测标签
        # ================================================================
        loss_cf = torch.tensor(0.0, device=device)
        if 'counterfactual_preds' in outputs and outputs['counterfactual_preds'] is not None:
            causal_effect_preds = outputs['logits_multi_for_causal'] - outputs['counterfactual_preds']
            loss_cf = self.task_criterion(causal_effect_preds, labels)
            loss_dict['L_counterfactual'] = loss_cf.item()
        
        # ================================================================
        # === [创新3] L_mi (互信息约束损失) ===
        # 包含 NCE (CPC) 和 lld (MMILB) 两部分
        # 总损失中: + alpha_nce * nce - beta_lld * lld
        # ================================================================
        loss_nce = torch.tensor(0.0, device=device)
        loss_lld = torch.tensor(0.0, device=device)
        if 'mi_outputs' in outputs and outputs['mi_outputs']:
            mi = outputs['mi_outputs']
            if 'nce' in mi and isinstance(mi['nce'], torch.Tensor):
                loss_nce = mi['nce']
                loss_dict['L_nce'] = loss_nce.item()
            if 'lld' in mi and isinstance(mi['lld'], torch.Tensor):
                loss_lld = mi['lld']
                loss_dict['L_lld'] = loss_lld.item()
        
        # ================================================================
        # 总损失: L_TOTAL
        # [P0-2] 不确定性加权 (Kendall et al., 2018) 或手工权重, 二选一
        # Grand Objective Function (手工权重版本):
        # L = L_task + λ_dec·L_dec + λ_hier·L_hier + λ_distill·L_distill
        #   + η·L_cf + α·L_nce - β·L_lld
        # 不确定性加权版本:
        # L = Σ_i [ exp(-log_σ²_i) * L_i + 0.5 * log_σ²_i ]  (L_lld 取负号)
        # ================================================================
        if self.use_uncertainty_weighting:
            def _uw(name, loss_val):
                """不确定性加权单项: exp(-log_var) * L + 0.5 * log_var"""
                log_var = self.log_vars[name]
                precision = torch.exp(-log_var)
                return (precision * loss_val + 0.5 * log_var).squeeze()

            # lld 是要最大化的下界, 作为损失时取负号
            total_loss = (
                _uw('task', loss_task) +
                _uw('dec', loss_dec) +
                _uw('hier', loss_hierarchical) +
                _uw('distill', loss_distill) +
                _uw('cf', loss_cf) +
                _uw('nce', loss_nce) +
                _uw('lld', -loss_lld)
            )
            # 记录当前学到的权重 (exp(-log_var)) 用于监控
            with torch.no_grad():
                for k in self.log_vars:
                    loss_dict[f'w_{k}'] = torch.exp(-self.log_vars[k]).item()
        else:
            total_loss = (
                loss_task +
                self.lambda_dec * loss_dec +
                self.lambda_hierarchical * loss_hierarchical +
                self.lambda_distill * loss_distill +
                self.lambda_counterfactual * loss_cf +
                self.alpha_nce * loss_nce -
                self.beta_lld * loss_lld
            )
        
        loss_dict['L_TOTAL'] = total_loss.item()
        
        return total_loss, loss_dict
    
    # ========================================================================
    # 模块一: L_dec (对抗性解耦损失)
    # ========================================================================
    
    def _compute_L_dec(self, outputs, labels, device):
        """
        计算对抗性解耦损失
        
        L_dec = L_rec + λ_adv·L_adv + γ_mar·L_mar' + γ_ort·L_ort
        
        四个子损失从信息完整性、模态不变性、语义结构、表示独立性四个维度
        全方位约束解耦过程。
        """
        decouple_items = outputs['decouple_items']
        loss_dict = {}
        
        # ----------------------------------------------------------------
        # 2.1 L_rec (重构损失)
        # Reconstruction Loss
        # ----------------------------------------------------------------
        loss_rec = self._compute_L_rec(decouple_items)
        loss_dict['L_rec'] = loss_rec.item()
        
        # ----------------------------------------------------------------
        # 2.2 L_adv (对抗损失)
        # Adversarial Modality Classification Loss
        # ----------------------------------------------------------------
        loss_adv = self._compute_L_adv(outputs, device)
        loss_dict['L_adv'] = loss_adv.item()
        
        # ----------------------------------------------------------------
        # 2.3 L_mar' (情感感知的间隔损失)
        # Emotion Structure-Aware Contrastive Loss
        # ----------------------------------------------------------------
        loss_mar = self._compute_L_mar_prime(decouple_items, labels, device)
        loss_dict['L_mar_prime'] = loss_mar.item()
        
        # ----------------------------------------------------------------
        # 2.4 L_ort (正交损失)
        # Representation Orthogonality Constraint
        # ----------------------------------------------------------------
        loss_ort = self._compute_L_ort(decouple_items)
        loss_dict['L_ort'] = loss_ort.item()
        
        # ----------------------------------------------------------------
        # L_dec 总和
        # ----------------------------------------------------------------
        loss_dec = (
            loss_rec +
            self.lambda_adv * loss_adv +
            self.gamma_mar * loss_mar +
            self.gamma_ort * loss_ort
        )
        loss_dict['L_dec'] = loss_dec.item()
        
        return loss_dec, loss_dict
    
    def _compute_L_rec(self, decouple_items):
        """
        L_rec = 𝔼[||X_m - D_rec(Concat(X_com, X_prt))||²]
        
        重构损失: 保证解耦过程信息无损
        计算原始投影特征与重构特征之间的MSE
        """
        loss_rec = 0.0
        count = 0
        
        # 模态名到原始特征键的映射
        original_key_map = {
            'text': 'original_text',
            'audio': 'original_audio',
            'video': 'original_video',
        }
        
        for modality in ['text', 'audio', 'video']:
            recon_feat = decouple_items[f'recon_{modality}']  # [B, L, d] reconstructed
            
            # 使用真正的原始投影特征作为重构目标（如果可用）
            orig_key = original_key_map[modality]
            if orig_key in decouple_items and decouple_items[orig_key] is not None:
                original = decouple_items[orig_key]  # [B, L, d] 解耦前的投影特征
            else:
                # 降级方案: 使用 specific + common 作为近似目标
                s_feat = decouple_items[f's_{modality}']  # [B, L, d]
                c_feat = decouple_items[f'c_{modality}']  # [B, L, d]
                original = s_feat + c_feat
            
            # MSE loss
            loss_rec += F.mse_loss(recon_feat, original)
            count += 1
        
        # 平均
        loss_rec = loss_rec / count if count > 0 else loss_rec
        
        return loss_rec
    
    def _compute_L_adv(self, outputs, device):
        """
        L_adv = CrossEntropyLoss(D_modality(GRL(X_com)), label_m)
        
        对抗性模态分类损失: 强迫通用特征模态不变
        通过最小-最大博弈，使判别器无法从X_com中识别模态来源
        """
        if 'adv_logits' not in outputs or outputs['adv_logits'] is None:
            return torch.tensor(0.0, device=device)
        
        adv_logits = outputs['adv_logits']  # [B, 3] 模态判别器输出
        
        # 目标: 使判别器输出趋向均匀分布 (无法判断模态)
        # 使用KL散度让判别器的输出接近uniform distribution
        batch_size = adv_logits.size(0)
        uniform_target = torch.full((batch_size, 3), 1.0/3.0, device=device)
        
        loss_adv = F.kl_div(
            F.log_softmax(adv_logits, dim=-1),
            uniform_target,
            reduction='batchmean'
        )
        
        return loss_adv
    
    def _compute_L_mar_prime(self, decouple_items, labels, device):
        """
        L_mar' = 𝔼[(α_dynamic - cos(X_com^i, X_com^j) + cos(X_com^i, X_com^k))₊]
        
        情感结构感知的对比损失: 核心微创新
        - 正样本对 (i,j): 同情感、不同模态
        - 负样本对 (i,k): 不同情感
        - 动态间隔: α_dynamic = α_base + β·D_VA(c_i, c_k)
        
        构建与人类情感认知模型几何同构的语义结构
        """
        # 收集所有模态的通用特征
        c_text = decouple_items['c_text']    # [B, L_t, d]
        c_audio = decouple_items['c_audio']  # [B, L_a, d]
        c_video = decouple_items['c_video']  # [B, L_v, d]
        
        # 全局平均池化到 [B, d]
        c_text_pooled = c_text.mean(dim=1)   # [B, d]
        c_audio_pooled = c_audio.mean(dim=1) # [B, d]
        c_video_pooled = c_video.mean(dim=1) # [B, d]
        
        # 归一化
        c_text_norm = F.normalize(c_text_pooled, p=2, dim=-1)
        c_audio_norm = F.normalize(c_audio_pooled, p=2, dim=-1)
        c_video_norm = F.normalize(c_video_pooled, p=2, dim=-1)
        
        # 构建所有特征的列表 [3B, d]
        all_features = torch.cat([c_text_norm, c_audio_norm, c_video_norm], dim=0)
        
        # 扩展标签 [3B]
        # Flatten labels first if they have extra dimensions
        labels_flat = labels.view(-1)  # [B]
        if self.task_type == 'classification':
            all_labels = labels_flat.repeat(3)  # [3B]
        else:
            # 回归任务没有离散标签,使用连续值的分箱
            # 将回归值分成若干个bins作为伪类别
            all_labels = self._discretize_regression_labels(labels_flat.repeat(3))
        
        batch_size = all_features.size(0)
        loss_triplet = torch.tensor(0.0, device=device)
        num_triplets = 0
        
        # 构建三元组 (anchor, positive, negative)
        for i in range(batch_size):
            anchor = all_features[i]       # [d]
            anchor_label = all_labels[i]
            
            # 找正样本: 同标签但不同索引 (优先不同模态)
            pos_mask = (all_labels == anchor_label) & (torch.arange(batch_size, device=device) != i)
            if pos_mask.sum() == 0:
                continue
            
            # 找负样本: 不同标签
            neg_mask = (all_labels != anchor_label)
            if neg_mask.sum() == 0:
                continue
            
            # 随机选择一个正样本
            pos_indices = torch.where(pos_mask)[0]
            pos_idx = pos_indices[torch.randint(len(pos_indices), (1,))].item()
            positive = all_features[pos_idx]
            
            # 随机选择一个负样本
            neg_indices = torch.where(neg_mask)[0]
            neg_idx = neg_indices[torch.randint(len(neg_indices), (1,))].item()
            negative = all_features[neg_idx]
            neg_label = all_labels[neg_idx]
            
            # 计算余弦相似度
            cos_pos = F.cosine_similarity(anchor.unsqueeze(0), positive.unsqueeze(0))
            cos_neg = F.cosine_similarity(anchor.unsqueeze(0), negative.unsqueeze(0))
            
            # 动态间隔: α_dynamic = α_base + β·D_VA(c_i, c_k)
            if anchor_label.item() < self.va_distances.size(0) and neg_label.item() < self.va_distances.size(1):
                va_distance = self.va_distances[anchor_label.item(), neg_label.item()]
                alpha_dynamic = self.alpha_base + self.beta * va_distance
            else:
                alpha_dynamic = self.alpha_base
            
            # Triplet loss with dynamic margin（保持计算图连接）
            loss = torch.relu(alpha_dynamic - cos_pos + cos_neg)
            loss_triplet = loss_triplet + loss
            num_triplets += 1
        
        # 平均
        if num_triplets > 0:
            loss_triplet = loss_triplet / num_triplets
            return loss_triplet
        else:
            return torch.tensor(0.0, device=device)
    
    def _discretize_regression_labels(self, labels, num_bins=5):
        """
        将回归标签离散化为伪类别 (用于回归任务的三元组损失)
        """
        min_val = labels.min()
        max_val = labels.max()
        if max_val - min_val < 1e-6:
            return torch.zeros_like(labels, dtype=torch.long)
        
        bins = torch.linspace(min_val, max_val, num_bins+1, device=labels.device)
        discretized = torch.bucketize(labels, bins[1:-1])
        return discretized
    
    def _compute_L_ort(self, decouple_items):
        """
        L_ort = 𝔼[|cos(X_com, X_prt)|]
        
        表示正交性约束: 鼓励通用和特有特征线性无关
        通过最小化余弦相似度的绝对值，使两个特征空间正交
        """
        loss_ort = 0.0
        count = 0
        
        for modality in ['text', 'audio', 'video']:
            s_feat = decouple_items[f's_{modality}']  # [B, L, d] specific
            c_feat = decouple_items[f'c_{modality}']  # [B, L, d] common
            
            # 展平到 [B*L, d]
            s_flat = s_feat.reshape(-1, s_feat.size(-1))
            c_flat = c_feat.reshape(-1, c_feat.size(-1))
            
            # 归一化
            s_norm = F.normalize(s_flat, p=2, dim=-1)
            c_norm = F.normalize(c_flat, p=2, dim=-1)
            
            # 余弦相似度
            cos_sim = (s_norm * c_norm).sum(dim=-1)  # [B*L]
            
            # 最小化绝对值
            loss_ort += torch.abs(cos_sim).mean()
            count += 1
        
        loss_ort = loss_ort / count if count > 0 else loss_ort
        
        return loss_ort
    
    # ========================================================================
    # 模块二: L_hierarchical (分层监督损失)
    # ========================================================================
    
    def _compute_L_hierarchical(self, outputs, labels):
        """
        计算分层监督损失
        
        L_hierarchical = w_uni·L_yuni + w_bi·L_ybi + w_mul·L_ymul
        
        为模型的不同深度提供直接的监督信号，解决梯度消失问题
        """
        loss_dict = {}
        
        # ----------------------------------------------------------------
        # L_yuni: 单模态级监督 (解耦后)
        # ----------------------------------------------------------------
        loss_yuni = 0.0
        for i, modality in enumerate(['text', 'audio', 'video']):
            logit = outputs['logits_uni'][i]  # [B, C]
            loss_yuni += self.task_criterion(logit, labels)
        loss_yuni /= 3.0
        loss_dict['L_yuni'] = loss_yuni.item()
        
        # ----------------------------------------------------------------
        # L_ybi: 双模态级监督 (中间交互)
        # ----------------------------------------------------------------
        loss_ybi = 0.0
        for i, pair in enumerate(['ta', 'tv', 'av']):
            logit = outputs['logits_bi'][i]  # [B, C]
            loss_ybi += self.task_criterion(logit, labels)
        loss_ybi /= 3.0
        loss_dict['L_ybi'] = loss_ybi.item()
        
        # ----------------------------------------------------------------
        # L_ymul: 三模态级监督 (最终融合)
        # ----------------------------------------------------------------
        loss_ymul = self.task_criterion(outputs['logits_multi'], labels)
        loss_dict['L_ymul'] = loss_ymul.item()
        
        # ----------------------------------------------------------------
        # L_hierarchical 总和
        # ----------------------------------------------------------------
        loss_hierarchical = (
            self.w_uni * loss_yuni +
            self.w_bi * loss_ybi +
            self.w_mul * loss_ymul
        )
        loss_dict['L_hierarchical'] = loss_hierarchical.item()
        
        return loss_hierarchical, loss_dict
    
    # ========================================================================
    # 模块三: L_distill (对比知识蒸馏损失)
    # ========================================================================
    
    def _compute_L_distill(self, outputs):
        """
        计算对比知识蒸馏损失
        
        L_distill = D_KL(P_Teacher || P_Student)
        
        让"教师"轨道(HMNF-处理通用特征)将结构化知识
        迁移给"学生"轨道(HMPN-处理特有特征)
        
        不直接匹配输出值，而是匹配样本间的关系分布
        """
        feat_teacher = outputs['features_contrast']['hmnf']  # [B, d] 教师: HMNF
        feat_student = outputs['features_contrast']['hmpn']  # [B, d] 学生: HMPN
        
        batch_size = feat_teacher.size(0)
        
        if batch_size < 2:
            # Batch太小，无法计算关系矩阵
            loss_distill = torch.tensor(0.0, device=feat_teacher.device)
            loss_dict = {'L_distill': 0.0}
            return loss_distill, loss_dict
        
        # ----------------------------------------------------------------
        # 步骤1: 计算成对相似度矩阵
        # ----------------------------------------------------------------
        # 归一化特征
        feat_teacher_norm = F.normalize(feat_teacher, p=2, dim=-1)  # [B, d]
        feat_student_norm = F.normalize(feat_student, p=2, dim=-1)  # [B, d]
        
        # 相似度矩阵 (余弦相似度)
        sim_teacher = torch.matmul(feat_teacher_norm, feat_teacher_norm.T)  # [B, B]
        sim_student = torch.matmul(feat_student_norm, feat_student_norm.T)  # [B, B]
        
        # ----------------------------------------------------------------
        # 步骤2: 转化为概率分布 (带温度的Softmax)
        # ----------------------------------------------------------------
        # 除以温度系数 τ (temperature)，得到更软的分布
        # 较高的τ会产生更平滑的概率分布，有助于传递更丰富的信息
        P_teacher = F.softmax(sim_teacher / self.tau, dim=1)  # [B, B]
        P_student = F.log_softmax(sim_student / self.tau, dim=1)  # [B, B] (log space)
        
        # ----------------------------------------------------------------
        # 步骤3: 计算KL散度
        # ----------------------------------------------------------------
        # KL(P_teacher || P_student) = Σ P_teacher * log(P_teacher / P_student)
        #                             = Σ P_teacher * (log P_teacher - log P_student)
        loss_distill = F.kl_div(
            P_student,
            P_teacher,
            reduction='batchmean'
        )
        
        loss_dict = {'L_distill': loss_distill.item()}
        
        return loss_distill, loss_dict


class WarmupScheduler:
    """
    Learning rate warmup scheduler
    """
    def __init__(self, optimizer, warmup_epochs, base_lr, target_lr):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.base_lr = base_lr
        self.target_lr = target_lr
        self.current_epoch = 0
    
    def step(self):
        if self.current_epoch < self.warmup_epochs:
            lr = self.base_lr + (self.target_lr - self.base_lr) * \
                 (self.current_epoch / self.warmup_epochs)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        self.current_epoch += 1
    
    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']

