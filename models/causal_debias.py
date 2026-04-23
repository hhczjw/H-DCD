"""
状态空间因果去偏模块 (State-Space Causal Debiasing Module, SS-CD)

============================================================================
整体架构说明
============================================================================
本模块是 H-DCD 框架的核心创新模块之一，实现"前门调整"(Front-Door Adjustment)
因果去偏理论的 Mamba2 化版本。

原理:
    对于每个模态，通过双路径机制消除混杂因子的影响:
    - IS路径 (Individual SSM Path):
        使用双向 Mamba2 (BiMamba2Block) 捕获模态内部的时序依赖，
        以线性复杂度 O(L) 替代原始 Transformer 自注意力的 O(L²)。
    - CS路径 (Confounder-aware SSM Path):
        设计"条件 Mamba2" (ConditionalMamba2Block) 机制，将混杂因子字典的
        全局统计信息通过三阶段条件注入(聚合→调制→扫描)调制 SSM 输入，
        替代原始 Transformer 跨注意力 (Q=模态, K/V=字典)。
    最终将两条路径的输出拼接，再通过 MLP 映射回原始维度。

数据流:
    输入 x [B, L, D]
      ├── IS路径: BiMamba2Block × N层 → is_output [B, L, D]
      ├── CS路径: ConditionalMamba2Block × N层 → cs_output [B, L, D]
      └── 融合: cat([is_output, cs_output]) → MLP → output + x (残差)

调用关系:
    MultiModalDebiasWrapper
      └── UnimodalDebiasModule (每个模态一个)
            ├── BiMamba2Block × num_layers  (IS路径)
            ├── ConditionalMamba2Block × num_layers  (CS路径)
            ├── confounder_dict  (可学习混杂因子字典)
            └── fusion_mlp  (双路径融合)
"""

import os
import numpy as np
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
# BiMamba2Block: IS路径的双向 Mamba2 自编码块
# ============================================================================
class BiMamba2Block(nn.Module):
    """
    双向 Mamba2 自编码块 (Bidirectional Mamba2 Self-Encoding Block)

    功能定位:
        替代原始 TransformerEncoderLayer，作为 IS路径 的核心组件。
        使用双向 Mamba2 + 门控融合实现模态内部时序依赖的捕获，
        计算复杂度从 O(L²D) 降至 O(LD)。

    架构流程 (参考 H-DCD 已有的 HMNFBlock):
        1. 门控分支: Linear → SiLU → gate
        2. 正向路径: Conv1d → 残差 → Linear → Mamba2_fwd
        3. 反向路径: Flip → Conv1d → 残差 → Linear → Mamba2_bwd → Flip
        4. 融合输出: (fwd ⊙ gate + bwd ⊙ gate) → Linear → RMSNorm → 残差

    数学形式:
        h_fwd = Mamba2_fwd(Linear(Conv1d(x) + x))
        h_bwd = Flip(Mamba2_bwd(Linear(Conv1d(Flip(x)) + Flip(x))))
        gate = SiLU(W_g · x)
        output = RMSNorm(W_o · (h_fwd ⊙ gate + h_bwd ⊙ gate)) + x

    Args:
        d_model (int): 输入输出特征维度
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
                "BiMamba2Block 需要 mamba_ssm.Mamba2，请安装: pip install mamba-ssm"
            )

        self.d_model = d_model

        # ====================================================================
        # 1. 门控分支 (Gating Branch)
        #    生成门控信号，用于自适应融合正向和反向路径的输出
        #    SiLU 激活函数提供平滑的门控值 ∈ (-∞, +∞) × sigmoid 特性
        # ====================================================================
        self.gate_linear = nn.Linear(d_model, d_model)
        self.gate_act = nn.SiLU()

        # ====================================================================
        # 2. 正向路径 (Forward Path)
        #    Conv1d(k=3) 提供局部上下文感知，Mamba2 提供长程依赖建模
        # ====================================================================
        # Conv1d: kernel_size=3, padding=1 保持序列长度不变
        self.fwd_conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.fwd_linear = nn.Linear(d_model, d_model)
        self.fwd_mamba = Mamba2(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            headdim=headdim,
        )

        # ====================================================================
        # 3. 反向路径 (Backward Path)
        #    与正向路径结构对称，处理时间反转的序列以捕获反向依赖
        # ====================================================================
        self.bwd_conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.bwd_linear = nn.Linear(d_model, d_model)
        self.bwd_mamba = Mamba2(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            headdim=headdim,
        )

        # ====================================================================
        # 4. 融合与输出 (Fusion & Output)
        # ====================================================================
        self.out_linear = nn.Linear(d_model, d_model)
        self.norm = RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _ensure_mamba2_stride(tensor: torch.Tensor) -> torch.Tensor:
        """
        确保张量的 stride 满足 Mamba2 的硬件对齐要求。
        Mamba2 要求 stride(0) 和 stride(2) 必须是 8 的倍数。
        """
        tensor = tensor.contiguous()
        if tensor.stride(0) % 8 != 0 or tensor.stride(2) % 8 != 0:
            tensor = tensor.clone()
        return tensor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入特征 [B, L, d_model] (batch-first 格式)

        Returns:
            output: 编码后的特征 [B, L, d_model]
        """
        # B=batch_size, L=序列长度, D=特征维度
        B, L, D = x.shape
        residual_global = x  # 保存用于全局残差连接

        # ====================================================================
        # 步骤1: 生成门控信号
        # gate ∈ [B, L, D]，通过 SiLU 激活控制正/反向信息的融合比例
        # ====================================================================
        gate = self.gate_act(self.gate_linear(x))  # [B, L, D]

        # ====================================================================
        # 步骤2: 正向路径
        # Conv1d 捕获 ±1 步的局部上下文，加残差后经 Linear+Mamba2 建模长程依赖
        # ====================================================================
        # Conv1d 需要 (B, D, L) 格式
        x_transposed = x.transpose(1, 2)  # [B, D, L]
        fwd_conv_out = self.fwd_conv(x_transposed)  # [B, D, L]
        fwd_conv_out = fwd_conv_out.transpose(1, 2)  # [B, L, D]

        # 残差连接: 保留原始信息 + 局部上下文增强
        x_fwd = x + fwd_conv_out  # [B, L, D]

        # Linear 投影后送入 Mamba2 进行选择性状态空间扫描
        x_fwd = self.fwd_linear(x_fwd)  # [B, L, D]
        x_fwd = self._ensure_mamba2_stride(x_fwd)
        out_fwd = self.fwd_mamba(x_fwd)  # [B, L, D]

        # ====================================================================
        # 步骤3: 反向路径
        # 先翻转序列再处理，使 Mamba2 的因果扫描方向变为反向
        # 处理完后翻转回原始顺序，实现反向时序依赖建模
        # ====================================================================
        x_flip = torch.flip(x, dims=[1])  # [B, L, D] 时间反转
        x_flip_transposed = x_flip.transpose(1, 2)  # [B, D, L]
        bwd_conv_out = self.bwd_conv(x_flip_transposed)  # [B, D, L]
        bwd_conv_out = bwd_conv_out.transpose(1, 2)  # [B, L, D]

        x_bwd = x_flip + bwd_conv_out  # [B, L, D]
        x_bwd = self.bwd_linear(x_bwd)  # [B, L, D]
        x_bwd = self._ensure_mamba2_stride(x_bwd)
        out_bwd = self.bwd_mamba(x_bwd)  # [B, L, D]

        # 翻转回原始时间顺序
        out_bwd = torch.flip(out_bwd, dims=[1])  # [B, L, D]

        # ====================================================================
        # 步骤4: 门控融合 + 输出
        # 正向和反向输出分别与门控信号逐元素相乘后求和
        # 通过 Linear → RMSNorm 标准化后加全局残差
        # ====================================================================
        fwd_gated = out_fwd * gate  # [B, L, D]
        bwd_gated = out_bwd * gate  # [B, L, D]
        summed = fwd_gated + bwd_gated  # [B, L, D]

        out = self.out_linear(summed)  # [B, L, D]
        out = self.norm(out)  # [B, L, D]
        out = self.dropout(out)  # [B, L, D]

        # 全局残差连接: 确保梯度直通路径
        return residual_global + out  # [B, L, D]


# ============================================================================
# ConditionalMamba2Block: CS路径的条件 Mamba2 跨模态扫描块
# ============================================================================
class ConditionalMamba2Block(nn.Module):
    """
    条件 Mamba2 混杂因子交互块 (Conditional Mamba2 Confounder Interaction Block)

    功能定位:
        替代原始 nn.MultiheadAttention 跨注意力层，作为 CS路径 的核心组件。
        通过"三阶段条件注入"将混杂因子字典的信息注入 Mamba2 的处理流程，
        实现对混杂因子的逐步边缘化。

    三阶段条件注入:
        1. 位置感知字典查询: 每个位置独立查询字典 [K,D]
           x_query = W_q · x  → [B, L, D]
           α = softmax(x_query · (W_k · Dict)^T / √D)  → [B, L, K]
           c_pos = α · Dict  → [B, L, D]
        2. 位置级条件调制: 每个位置获得独立的 FiLM 参数
           x' = x ⊙ sigmoid(W_scale · c_pos) + W_shift · c_pos
        3. 选择性扫描: 调制后的序列经 Mamba2 处理
           output = Mamba2(x') + x  (残差连接)

    理论等价性:
        - 原始跨注意力: 逐位置查询字典 → 注意力权重 → 加权求和
        - 条件Mamba2: 全局条件注入 → Mamba2 选择性门控 Δ(x) 动态决定
          在每个时间步保留/遗忘多少字典条件信息

    Args:
        d_model (int): 特征维度
        confounder_size (int): 混杂因子字典大小 K
        d_state (int): Mamba2 SSM 状态维度
        d_conv (int): Mamba2 内部卷积核大小
        expand (int): Mamba2 扩展因子
        dropout (float): Dropout 概率
        headdim (int): Mamba2 头维度
    """

    def __init__(
        self,
        d_model: int = 128,
        confounder_size: int = 50,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        headdim: int = 32,
    ):
        super().__init__()

        if not MAMBA2_AVAILABLE:
            raise ImportError(
                "ConditionalMamba2Block 需要 mamba_ssm.Mamba2，请安装: pip install mamba-ssm"
            )

        self.d_model = d_model
        self.confounder_size = confounder_size
        self.scaling = d_model ** -0.5  # 缩放因子用于注意力池化

        # ====================================================================
        # [P1-4] 阶段1: 位置感知字典查询 (Position-Aware Dictionary Query)
        # 改进: 每个位置独立查询字典, 获得位置级条件向量 [B, L, D]
        # 替代原来的全局均值池化 → 广播, 使去偏操作具有位置感知能力
        # ====================================================================
        self.dict_key_proj = nn.Linear(d_model, d_model)   # 字典 Key 投影
        self.query_proj = nn.Linear(d_model, d_model)      # 位置 Query 投影

        # ====================================================================
        # 阶段2: 条件调制层 (Conditional Modulation)
        # 使用 FiLM (Feature-wise Linear Modulation) 机制:
        # x' = x ⊙ scale + shift
        # [P1-4] scale/shift 现在是位置级的 [B, L, D], 不再广播
        # ====================================================================
        self.modulation_scale = nn.Linear(d_model, d_model)
        self.modulation_shift = nn.Linear(d_model, d_model)

        # ====================================================================
        # 阶段3: 选择性状态空间扫描 (Selective SSM Scan)
        # 条件调制后的序列送入 Mamba2 进行选择性建模
        # ====================================================================
        self.norm = RMSNorm(d_model)
        self.mamba = Mamba2(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            headdim=headdim,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        confounder_dict: torch.Tensor,
    ) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 模态输入特征 [B, L, d_model]
            confounder_dict: 混杂因子字典 [K, d_model] (可学习参数)

        Returns:
            output: 混杂因子交互后的特征 [B, L, d_model]
        """
        B, L, D = x.shape

        # ====================================================================
        # [P1-4] 阶段1: 位置感知字典查询
        # 每个位置独立查询字典, 获得位置级条件向量 [B, L, D]
        # 替代原来的均值池化 + 全局广播
        # ====================================================================
        # 投影字典为 Key: [K, D]
        dict_keys = self.dict_key_proj(confounder_dict)  # [K, D]

        # 每个位置独立作为 Query: [B, L, D]
        x_query = self.query_proj(x)  # [B, L, D]

        # 位置级注意力: [B, L, D] × [D, K] → [B, L, K]
        attn_logits = torch.matmul(
            x_query, dict_keys.t()
        ) * self.scaling  # [B, L, K]

        # softmax 得到每个位置的聚合权重
        attn_weights = F.softmax(attn_logits, dim=-1)  # [B, L, K]

        # 加权求和: 每个位置获得独立的条件向量
        # c_pos: [B, L, K] × [K, D] → [B, L, D]
        c_pos = torch.matmul(
            attn_weights, confounder_dict
        )  # [B, L, D]

        # ====================================================================
        # [P1-4] 阶段2: 位置感知条件调制
        # scale/shift 现在是 [B, L, D], 每个位置有不同的去偏强度
        # ====================================================================
        scale = torch.sigmoid(
            self.modulation_scale(c_pos)
        )  # [B, L, D]

        shift = self.modulation_shift(c_pos)  # [B, L, D]

        # 位置级 FiLM 调制 (无需广播, 维度已匹配)
        x_modulated = x * scale + shift  # [B, L, D]

        # ====================================================================
        # 阶段3: Mamba2 选择性扫描 + 残差连接
        # ====================================================================
        x_normed = self.norm(x_modulated)  # [B, L, D]

        # 确保满足 Mamba2 stride 要求
        x_normed = x_normed.contiguous()
        if x_normed.stride(0) % 8 != 0 or x_normed.stride(2) % 8 != 0:
            x_normed = x_normed.clone()

        mamba_out = self.mamba(x_normed)  # [B, L, D]
        mamba_out = self.dropout(mamba_out)  # [B, L, D]

        # 残差连接: 保持原始输入的信息通路
        return x + mamba_out  # [B, L, D]


# ============================================================================
# UnimodalDebiasModule: 单模态因果去偏模块 (整合 IS 和 CS 路径)
# ============================================================================
class UnimodalDebiasModule(nn.Module):
    """
    单模态状态空间因果去偏模块 (Unimodal State-Space Causal Debiasing Module)

    功能定位:
        H-DCD 框架中的创新模块一 (SS-CD)。对单个模态的特征执行因果去偏，
        通过前门调整的双路径机制消除混杂偏差。

    在整体架构中的位置:
        特征投影层 → 【SS-CD 因果去偏】 → 解耦编码器 → 双流融合

    Args:
        d_model (int): 输入特征维度 (H-DCD 统一维度, 如 128)
        num_layers (int): IS/CS 路径的层数
        confounder_size (int): 混杂因子字典大小 (KMeans 聚类数)
        d_state (int): Mamba2 SSM 状态维度
        d_conv (int): Mamba2 内部卷积核大小
        expand (int): Mamba2 扩展因子
        dropout (float): Dropout 概率
        headdim (int): Mamba2 头维度
        confounder_npy_path (str, optional): KMeans 初始化的 .npy 文件路径
    """

    def __init__(
        self,
        d_model: int = 128,
        num_layers: int = 2,
        confounder_size: int = 50,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        headdim: int = 32,
        confounder_npy_path: str = None,
    ):
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self.confounder_size = confounder_size

        # ====================================================================
        # IS路径: 双向 Mamba2 自编码 × N层 (Individual SSM Path)
        # 功能: 捕获模态内部的时序依赖，不受混杂因子影响
        # ====================================================================
        self.is_blocks = nn.ModuleList([
            BiMamba2Block(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                dropout=dropout,
                headdim=headdim,
            )
            for _ in range(num_layers)
        ])

        # ====================================================================
        # CS路径: 条件 Mamba2 × N层 (Confounder-aware SSM Path)
        # 功能: 通过字典条件注入机制边缘化混杂因子
        # ====================================================================
        self.cs_blocks = nn.ModuleList([
            ConditionalMamba2Block(
                d_model=d_model,
                confounder_size=confounder_size,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                dropout=dropout,
                headdim=headdim,
            )
            for _ in range(num_layers)
        ])

        # ====================================================================
        # 混杂因子字典 (Confounder Dictionary)
        # 可学习参数 [K, d_model]，支持 KMeans 或随机初始化
        # ====================================================================
        self._init_confounder_dictionary(confounder_npy_path)

        # ====================================================================
        # 融合 MLP: 将 IS 和 CS 路径的输出拼接后映射回 d_model
        # [B, L, 2D] → MLP → [B, L, D]
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

    def _init_confounder_dictionary(self, npy_path: str):
        """
        初始化混杂因子字典

        支持两种初始化方式:
        1. KMeans 初始化: 使用预计算的聚类中心 (推荐用于正式训练)
        2. 随机初始化: 值除以 100 保持较小初始值 (与 AtCAF 保持一致)

        Args:
            npy_path: .npy 文件路径, 包含 [K, D] 的聚类中心
        """
        if npy_path is not None and os.path.exists(npy_path):
            center_data = np.load(npy_path)  # [K, D]
            if center_data.ndim == 1:
                center_data = center_data.reshape(self.confounder_size, -1)
            if center_data.shape != (self.confounder_size, self.d_model):
                raise RuntimeError(
                    f"混杂因子字典维度不匹配: "
                    f"期望 ({self.confounder_size}, {self.d_model}), "
                    f"实际 {center_data.shape}"
                )
            self.confounder_dict = nn.Parameter(
                torch.from_numpy(center_data).float()
            )
            print(f"[SS-CD] 使用 KMeans 初始化混杂因子字典: {npy_path}")
        else:
            # 随机初始化，小值防止训练初期梯度爆炸
            self.confounder_dict = nn.Parameter(
                torch.rand(self.confounder_size, self.d_model) / 100.0
            )
            print(
                f"[SS-CD] 随机初始化混杂因子字典: "
                f"size={self.confounder_size}, dim={self.d_model}"
            )

    def forward(self, x: torch.Tensor, src_key_padding_mask=None) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入特征 [B, L, d_model] (batch-first 格式)
            src_key_padding_mask: 保留接口兼容性 (Mamba2 不需要显式 padding mask,
                因为 SSM 的递归特性天然处理变长序列)

        Returns:
            output: 去偏后的特征 [B, L, d_model]
        """
        # ====================================================================
        # IS路径: 双向 Mamba2 逐层处理
        # 每层 BiMamba2Block 内部包含残差连接，支持深层堆叠
        # ====================================================================
        is_output = x
        for block in self.is_blocks:
            is_output = block(is_output)  # [B, L, D]

        # ====================================================================
        # CS路径: 条件 Mamba2 逐层处理
        # 每层 ConditionalMamba2Block 使用相同的混杂因子字典进行条件注入
        # ====================================================================
        cs_output = x
        for block in self.cs_blocks:
            cs_output = block(cs_output, self.confounder_dict)  # [B, L, D]

        # ====================================================================
        # 双路径融合: 拼接 → MLP → 残差
        # 拼接 IS(捕获纯时序依赖) 和 CS(边缘化混杂因子) 的输出，
        # 通过 MLP 学习两条路径信息的最优融合权重
        # ====================================================================
        combined = torch.cat([is_output, cs_output], dim=-1)  # [B, L, 2D]
        output = self.fusion_mlp(combined)  # [B, L, D]

        # 全局残差连接: 保持与原始输入的信息通路
        output = output + x  # [B, L, D]

        return output


# ============================================================================
# MultiModalDebiasWrapper: 多模态去偏包装器
# ============================================================================
class MultiModalDebiasWrapper(nn.Module):
    """
    多模态状态空间去偏包装器 (Multi-Modal SS-CD Wrapper)

    功能定位:
        为 H-DCD 的三个模态 (text, audio, video) 分别创建独立的
        UnimodalDebiasModule 实例，统一管理和调用。

    在整体架构中的位置:
        H_DCD.forward() 中:
        特征投影 → 【MultiModalDebiasWrapper】 → 解耦编码器

    Args:
        d_model (int): 统一特征维度
        num_layers (int): 每个去偏模块的 IS/CS 层数
        confounder_size (int): 混杂因子字典大小
        d_state (int): Mamba2 SSM 状态维度
        d_conv (int): Mamba2 内部卷积核大小
        expand (int): Mamba2 扩展因子
        dropout (float): Dropout 概率
        headdim (int): Mamba2 头维度
        debias_text (bool): 是否对文本去偏
        debias_audio (bool): 是否对音频去偏
        debias_video (bool): 是否对视频去偏
        confounder_npy_dir (str, optional): 包含 KMeans .npy 文件的目录路径
        dataset_name (str): 数据集名称 (用于定位 .npy 文件)
    """

    def __init__(
        self,
        d_model: int = 128,
        num_layers: int = 2,
        confounder_size: int = 50,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        headdim: int = 32,
        debias_text: bool = True,
        debias_audio: bool = True,
        debias_video: bool = True,
        confounder_npy_dir: str = None,
        dataset_name: str = 'mosi',
    ):
        super().__init__()

        self.debias_text = debias_text
        self.debias_audio = debias_audio
        self.debias_video = debias_video

        def _get_npy_path(modal_name):
            """根据模态名称构建 KMeans .npy 文件路径"""
            if confounder_npy_dir is None:
                return None
            path = os.path.join(
                confounder_npy_dir,
                f"kmeans_{dataset_name}-{confounder_size}_{modal_name}.npy"
            )
            return path if os.path.exists(path) else None

        # 共用的 Mamba2 参数字典
        mamba_kwargs = dict(
            d_model=d_model,
            num_layers=num_layers,
            confounder_size=confounder_size,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
            headdim=headdim,
        )

        # 为启用去偏的模态创建独立的去偏模块
        if debias_text:
            self.text_debias = UnimodalDebiasModule(
                **mamba_kwargs,
                confounder_npy_path=_get_npy_path('text'),
            )

        if debias_audio:
            self.audio_debias = UnimodalDebiasModule(
                **mamba_kwargs,
                confounder_npy_path=_get_npy_path('audio'),
            )

        if debias_video:
            self.video_debias = UnimodalDebiasModule(
                **mamba_kwargs,
                confounder_npy_path=_get_npy_path('visual'),
            )

    def forward(
        self,
        x_text: torch.Tensor,
        x_audio: torch.Tensor,
        x_video: torch.Tensor,
        text_mask=None,
        audio_mask=None,
        video_mask=None,
    ):
        """
        对三个模态分别执行状态空间因果去偏

        Args:
            x_text:  文本特征 [B, L_t, d_model]
            x_audio: 音频特征 [B, L_a, d_model]
            x_video: 视频特征 [B, L_v, d_model]
            text_mask:  保留接口兼容性 (Mamba2 不使用)
            audio_mask: 保留接口兼容性
            video_mask: 保留接口兼容性

        Returns:
            debiased_text:  去偏后的文本特征 [B, L_t, d_model]
            debiased_audio: 去偏后的音频特征 [B, L_a, d_model]
            debiased_video: 去偏后的视频特征 [B, L_v, d_model]
        """
        # 文本去偏
        if self.debias_text:
            debiased_text = self.text_debias(x_text)
        else:
            debiased_text = x_text

        # 音频去偏
        if self.debias_audio:
            debiased_audio = self.audio_debias(x_audio)
        else:
            debiased_audio = x_audio

        # 视频去偏
        if self.debias_video:
            debiased_video = self.video_debias(x_video)
        else:
            debiased_video = x_video

        return debiased_text, debiased_audio, debiased_video