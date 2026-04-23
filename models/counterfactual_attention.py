"""
选择性反事实推断模块 (Selective Counterfactual Inference Module, SCI)

============================================================================
整体架构说明
============================================================================
本模块是 H-DCD 框架的核心创新模块之二，实现"反事实推断"(Counterfactual
Inference) 理论的 Mamba2 化版本。

核心洞察 — "选择性即因果性" (Selectivity as Causality):
    Mamba2 的选择性参数 Δ 在功能上等价于 Transformer 的注意力权重:
    - Transformer: attention_weights 控制"关注哪些位置"
    - Mamba2: Δ(x) 控制"每步保留多少旧状态、吸收多少新输入"
    两者均是控制信息流"选择性"的核心参数。

    因此，将反事实干预点从"注意力权重空间"迁移到"选择性门控空间"
    是一个自然且有理论支撑的映射。

架构设计 — 双通道 Mamba2:
    1. Factual 通道: 正常 Mamba2 跨模态扫描 → 真实融合特征
    2. Counterfactual 通道 (仅训练时): 对输入施加反事实干预 → 反事实融合特征
    3. 因果效应: factual_fusion - counterfactual_fusion

跨模态交互方式 — 序列拼接扫描法:
    x_cross = [x_kv; x_query]  (先 KV 后 Query 拼接)
    → Mamba2 因果扫描: KV 信息通过隐状态 h 传播到 Query 位置
    → 截取 Query 位置的输出作为跨模态融合结果

四种反事实策略 (在 Mamba2 输入层面实施干预):
    1. random:   用随机特征替代输入 → "如果输入信息完全随机"
    2. shuffle:  在 batch 维度打乱输入 → "如果输入属于其他样本"
    3. reversed: 时间反转输入序列 → "如果时序依赖完全反转"
    4. uniform:  用均值替代所有位置 → "如果失去一切位置差异"

数据流:
    输入: x_query [B, L_q, D], x_kv [B, L_kv, D]
      ├── Factual: cat([x_kv, x_query]) → Mamba2 → 截取 → FFN → factual_out
      ├── Counterfactual: 干预输入 → cat → Mamba2 → 截取 → cf_out
      └── 返回: factual_out, counterfactual_out (由上层计算因果效应)

调用关系:
    CounterfactualCrossAttention (多层包装)
      └── CounterfactualCrossAttentionLayer × num_layers (单层)
            └── CounterfactualMamba2 (核心双通道 Mamba2)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================================
# 导入 Mamba2 和公共组件
# ============================================================================
try:
    from mamba_ssm import Mamba2
    MAMBA2_AVAILABLE = True
except ImportError:
    MAMBA2_AVAILABLE = False

from common import RMSNorm


# ============================================================================
# CounterfactualMamba2: 支持反事实干预的 Mamba2 跨模态扫描
# ============================================================================
class CounterfactualMamba2(nn.Module):
    """
    支持反事实干预的 Mamba2 跨模态扫描模块

    功能定位:
        替代原始 CounterfactualMultiheadAttention (手动 QKV + softmax + 反事实干预)。
        通过序列拼接扫描实现跨模态交互，通过输入层面的反事实干预生成
        反事实世界的融合特征。

    跨模态交互原理:
        将 KV 模态序列拼接在 Query 模态前面: x_cross = [x_kv; x_query]
        Mamba2 的因果扫描特性确保 Query 位置能"看到" KV 位置的信息
        (信息通过 SSM 隐状态 h 从 KV 位置传播到 Query 位置)

    反事实干预原理:
        在 Transformer 中，干预 softmax 后的注意力权重;
        在 Mamba2 中，干预送入 SSM 的输入序列，从而间接改变
        Mamba2 内部的选择性参数 Δ(x) 和隐状态传播路径。

    Args:
        d_model (int): 模型维度
        d_state (int): Mamba2 SSM 状态维度
        d_conv (int): Mamba2 内部卷积核大小
        expand (int): Mamba2 扩展因子
        dropout (float): Dropout 概率
        headdim (int): Mamba2 头维度
    """

    def __init__(
        self,
        d_model: int = 128,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        headdim: int = 32,
    ):
        super().__init__()

        if not MAMBA2_AVAILABLE:
            raise ImportError(
                "CounterfactualMamba2 需要 mamba_ssm.Mamba2，"
                "请安装: pip install mamba-ssm"
            )

        self.d_model = d_model

        # ====================================================================
        # Factual 通道: 正常的 Mamba2 跨模态扫描
        # ====================================================================
        self.factual_norm = RMSNorm(d_model)
        self.factual_mamba = Mamba2(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            headdim=headdim,
        )

        # ====================================================================
        # Counterfactual 通道: 干预后的 Mamba2 扫描
        # 使用独立的 Mamba2 实例，确保 factual 和 counterfactual
        # 通道有不同的参数，使因果效应的计算更加准确
        # ====================================================================
        self.counterfactual_norm = RMSNorm(d_model)
        self.counterfactual_mamba = Mamba2(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            headdim=headdim,
        )

        # 输出投影 (对应原始的 out_proj)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _ensure_mamba2_stride(tensor: torch.Tensor) -> torch.Tensor:
        """确保张量满足 Mamba2 的 stride 对齐要求"""
        tensor = tensor.contiguous()
        if tensor.stride(0) % 8 != 0 or tensor.stride(2) % 8 != 0:
            tensor = tensor.clone()
        return tensor

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        counterfactual_type: str = None,
    ):
        """
        前向传播

        Args:
            query:    Query 模态特征 [B, L_q, d_model]
            key_value: Key/Value 模态特征 [B, L_kv, d_model]
            counterfactual_type: 反事实策略
                ('random'/'shuffle'/'reversed'/'uniform'/None)

        Returns:
            factual_output: 真实融合特征 [B, L_q, d_model]
            counterfactual_output: 反事实融合特征 [B, L_q, d_model] 或 None
        """
        B, L_q, D = query.shape
        L_kv = key_value.shape[1]

        # ====================================================================
        # Factual 通道: 正常跨模态扫描
        # ====================================================================
        # 序列拼接: 先 KV 后 Query，利用 Mamba2 的因果扫描特性
        # 使 Query 位置能通过隐状态 h 接收 KV 位置的信息
        factual_cross = torch.cat(
            [key_value, query], dim=1
        )  # [B, L_kv + L_q, D]

        factual_normed = self.factual_norm(factual_cross)  # [B, L_kv + L_q, D]
        factual_normed = self._ensure_mamba2_stride(factual_normed)
        factual_scanned = self.factual_mamba(factual_normed)  # [B, L_kv + L_q, D]

        # 截取 Query 位置的输出 (KV 位置仅作为上下文提供信息)
        factual_output = factual_scanned[:, L_kv:, :]  # [B, L_q, D]
        factual_output = self.out_proj(factual_output)  # [B, L_q, D]
        factual_output = self.dropout(factual_output)  # [B, L_q, D]

        # ====================================================================
        # Counterfactual 通道: 仅在指定策略时激活
        # ====================================================================
        counterfactual_output = None

        if counterfactual_type is not None:
            # 对输入施加反事实干预
            cf_key_value = self._apply_counterfactual_intervention(
                key_value, counterfactual_type
            )  # [B, L_kv, D]

            # 使用干预后的 KV 进行拼接扫描
            cf_cross = torch.cat(
                [cf_key_value, query], dim=1
            )  # [B, L_kv + L_q, D]

            cf_normed = self.counterfactual_norm(cf_cross)  # [B, L_kv + L_q, D]
            cf_normed = self._ensure_mamba2_stride(cf_normed)
            cf_scanned = self.counterfactual_mamba(cf_normed)  # [B, L_kv + L_q, D]

            # 截取 Query 位置的输出
            counterfactual_output = cf_scanned[:, L_kv:, :]  # [B, L_q, D]

        return factual_output, counterfactual_output

    def _apply_counterfactual_intervention(
        self,
        key_value: torch.Tensor,
        cf_type: str,
    ) -> torch.Tensor:
        """
        对 KV 输入施加反事实干预

        通过修改送入 Mamba2 的 KV 输入，间接改变 Mamba2 内部的
        选择性参数 Δ(x) = softplus(W_Δ · x_cf)，从而产生不同的
        隐状态传播路径，实现反事实世界的信息流。

        Args:
            key_value: 原始 KV 特征 [B, L_kv, D]
            cf_type: 反事实策略类型

        Returns:
            cf_key_value: 干预后的 KV 特征 [B, L_kv, D]
        """
        B, L_kv, D = key_value.shape

        if cf_type == 'random':
            # ============================================================
            # Random 策略: 用随机特征替代 KV
            # 效果: Mamba2 内部 Δ(x_random) 变为随机值，
            #       SSM 在每步随机决定保留/遗忘多少信息
            # 语义: "如果跨模态信息完全随机，融合结果会怎样?"
            # ============================================================
            cf_key_value = torch.randn_like(key_value)

        elif cf_type == 'shuffle':
            # ============================================================
            # Shuffle 策略: 在 batch 维度打乱 KV
            # 效果: 破坏样本与 KV 信息的对应关系
            # 语义: "如果跨模态信息属于其他样本，融合结果会怎样?"
            # ============================================================
            perm_indices = torch.randperm(B, device=key_value.device)
            cf_key_value = key_value[perm_indices]

        elif cf_type == 'reversed':
            # ============================================================
            # Reversed 策略: 时间反转 KV 序列
            # 效果: Mamba2 按相反的时间顺序处理 KV，
            #       隐状态传播路径完全反转
            # 语义: "如果跨模态的时序依赖完全反转，融合结果会怎样?"
            # ============================================================
            cf_key_value = torch.flip(key_value, dims=[1])

        elif cf_type == 'uniform':
            # ============================================================
            # Uniform 策略: 用序列均值替代所有位置
            # 效果: Δ(x_uniform) 在所有位置相同，
            #       SSM 以相同速率更新状态 (退化为线性 RNN)
            # 语义: "如果失去一切位置差异，融合结果会怎样?"
            # ============================================================
            # 计算序列维度的均值并广播
            kv_mean = key_value.mean(dim=1, keepdim=True)  # [B, 1, D]
            cf_key_value = kv_mean.expand_as(key_value)  # [B, L_kv, D]

        else:
            # 未知策略: 不干预 (回退到 factual)
            cf_key_value = key_value

        return cf_key_value


# ============================================================================
# CounterfactualCrossAttentionLayer: 单层反事实 Mamba2 跨模态交互
# ============================================================================
class CounterfactualCrossAttentionLayer(nn.Module):
    """
    单层选择性反事实跨模态交互层

    功能定位:
        替代原始的 CounterfactualCrossAttentionLayer (CrossAttn+LN+FFN)。
        使用 CounterfactualMamba2 进行跨模态扫描 + FFN 进行非线性变换。

    结构:
        CrossMamba2 → Add & Norm → FFN → Add & Norm

    Args:
        d_model (int): 模型维度
        d_state (int): Mamba2 SSM 状态维度
        d_conv (int): Mamba2 内部卷积核大小
        expand (int): Mamba2 扩展因子
        dim_feedforward (int): FFN 中间维度
        dropout (float): Dropout 概率
        headdim (int): Mamba2 头维度
    """

    def __init__(
        self,
        d_model: int = 128,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        headdim: int = 32,
    ):
        super().__init__()

        # 核心: 支持反事实干预的 Mamba2 跨模态扫描
        self.cross_mamba = CounterfactualMamba2(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
            headdim=headdim,
        )

        # Factual 通道的归一化和 FFN
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self,
        x_query: torch.Tensor,
        x_kv: torch.Tensor,
        kv_padding_mask=None,
        counterfactual_type: str = None,
    ):
        """
        前向传播

        Args:
            x_query: Query 模态特征 [B, L_q, d_model]
            x_kv: Key/Value 模态特征 [B, L_kv, d_model]
            kv_padding_mask: 保留接口兼容性 (Mamba2 不使用)
            counterfactual_type: 反事实策略

        Returns:
            factual_output: 真实融合输出 [B, L_q, d_model]
            counterfactual_output: 反事实融合输出 [B, L_q, d_model] 或 None
        """
        # ====================================================================
        # 步骤1: Mamba2 跨模态扫描 (factual + counterfactual)
        # ====================================================================
        factual_attn_out, cf_raw_out = self.cross_mamba(
            query=x_query,
            key_value=x_kv,
            counterfactual_type=counterfactual_type,
        )

        # ====================================================================
        # 步骤2: Factual 通道的残差 + LayerNorm + FFN
        # ====================================================================
        # 跨模态扫描输出 + 残差 + 归一化
        factual_output = self.norm1(x_query + factual_attn_out)  # [B, L_q, D]

        # FFN + 残差 + 归一化
        residual = factual_output
        ffn_out = self.ffn(factual_output)  # [B, L_q, D]
        factual_output = self.norm2(residual + ffn_out)  # [B, L_q, D]

        # ====================================================================
        # 步骤3: Counterfactual 通道 (仅传递原始输出，不加 FFN)
        # 反事实通道保持轻量: 仅需提供与 factual 的对比基线
        # ====================================================================
        counterfactual_output = cf_raw_out  # [B, L_q, D] 或 None

        return factual_output, counterfactual_output


# ============================================================================
# CounterfactualCrossAttention: 多层选择性反事实推断模块
# ============================================================================
class CounterfactualCrossAttention(nn.Module):
    """
    选择性反事实推断模块 (Selective Counterfactual Inference, SCI)

    功能定位:
        H-DCD 框架的创新模块二。在 HMPN 融合之后构建反事实分支，
        通过 "factual - counterfactual" 量化跨模态交互的因果效应。

    在整体架构中的位置:
        解耦编码 → 双流融合 (HMNF+HMPN) → 【SCI 反事实推断】 → 分层分类

    完整数据流:
        输入: x_query [B, L_q, D], x_kv [B, L_kv, D]
          │
          ├── Factual 通道 (始终激活):
          │     逐层: cat([x_kv, x_query]) → Mamba2 → 截取 → Res+LN → FFN → Res+LN
          │     → FinalNorm → factual_out [B, L_q, D]
          │
          └── Counterfactual 通道 (仅训练时):
                逐层: 干预输入 → cat → Mamba2 → 截取
                → cf_out [B, L_q, D]

    Args:
        d_model (int): 模型维度
        num_layers (int): 跨模态交互层数
        d_state (int): Mamba2 SSM 状态维度
        d_conv (int): Mamba2 内部卷积核大小
        expand (int): Mamba2 扩展因子
        dim_feedforward (int): FFN 中间维度
        dropout (float): Dropout 概率
        headdim (int): Mamba2 头维度
    """

    def __init__(
        self,
        d_model: int = 128,
        num_layers: int = 2,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        headdim: int = 32,
    ):
        super().__init__()

        self.num_layers = num_layers

        # 堆叠多层反事实跨模态交互层
        self.layers = nn.ModuleList([
            CounterfactualCrossAttentionLayer(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                headdim=headdim,
            )
            for _ in range(num_layers)
        ])

        # 最终归一化 (仅用于 factual 通道)
        self.final_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x_query: torch.Tensor,
        x_kv: torch.Tensor,
        query_padding_mask=None,
        kv_padding_mask=None,
        counterfactual_type: str = None,
    ):
        """
        前向传播

        Args:
            x_query: Query 模态特征 [B, L_q, d_model]
            x_kv: Key/Value 模态特征 [B, L_kv, d_model]
            query_padding_mask: 保留接口兼容性
            kv_padding_mask: 保留接口兼容性
            counterfactual_type: 反事实策略 (仅训练时使用)

        Returns:
            factual_output: 真实融合输出 [B, L_q, d_model]
            counterfactual_output: 反事实融合输出 [B, L_q, d_model] 或 None
        """
        factual_hidden = x_query
        counterfactual_hidden = None

        for layer in self.layers:
            factual_hidden, cf_layer_out = layer(
                x_query=factual_hidden,
                x_kv=x_kv,
                counterfactual_type=counterfactual_type,
            )
            # 累积最后一层的反事实输出
            if cf_layer_out is not None:
                counterfactual_hidden = cf_layer_out

        # Factual 通道最终归一化
        factual_output = self.final_norm(factual_hidden)  # [B, L_q, D]

        return factual_output, counterfactual_hidden