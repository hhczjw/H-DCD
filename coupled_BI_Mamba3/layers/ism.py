"""
ISM — Intra-modal Sequence Modeling Block
==========================================
参考 MSAmba/models/mamba_block.py::Block_GLCE 移植，去除对 timm / bimamba_inner_fn 的依赖，
完全用标准 PyTorch + mamba_ssm.Mamba(SISO) 实现，可无缝嵌入现有流水线。

结构（完全对应论文图）:

    输入 x  (B, L, D)
      ↓
    [LN]                            -- pre-norm
      ↓
    [GLCE]
        x_global = Linear(L→L) 作用在时间维 (转置操作)   -- 全局分支 (MLP on time)
        x_local  = Conv1d(kernel=3, padding=1)            -- 局部分支
        x = x_global + x_local + shortcut                -- ⊕ 三路加和
      ↓
    [LN]                            -- GLCE 后的第二个 LN
      ↓
    [BSSM]  双向选择性扫描
        z  = in_proj(x) / 2         -- 门控分支
        xf = Mamba_fwd(x_half)      -- 正向 SSM
        xb = flip(Mamba_bwd(flip(x_half))) -- 反向 SSM
        y  = (xf ⊗ z) ⊕ (xb ⊗ z)  -- 门控融合
        out = out_proj(y)           -- MLP 输出投影
      ↓
    ⊕ 残差 (+ 原始输入 x)
      ↓
    输出  (B, L, D)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# 优先与 MSAmba 保持一致：使用 mamba_simple.Mamba。
# 若项目内 mamba_simple 因环境/补丁问题在导入阶段抛出非 ImportError（如 NameError），
# 则自动回退到 mamba2_simple；再失败则回退到纯 PyTorch BiGRU。
_MAMBA_AVAILABLE = False
_MAMBA_BACKEND = "none"
MambaSSM = None

try:
    from mamba_ssm.modules.mamba_simple import Mamba as MambaSSM
    _MAMBA_AVAILABLE = True
    _MAMBA_BACKEND = "mamba_simple"
except Exception:
    try:
        from mamba_ssm.modules.mamba2_simple import Mamba2Simple as MambaSSM
        _MAMBA_AVAILABLE = True
        _MAMBA_BACKEND = "mamba2_simple"
    except Exception:
        MambaSSM = None
        _MAMBA_AVAILABLE = False
        _MAMBA_BACKEND = "bigru"


# ---------------------------------------------------------------------------
# GLCE — Global-Local Context Extractor
# ---------------------------------------------------------------------------
class GLCE(nn.Module):
    """
    全局-局部上下文提取器 (纯 PyTorch, 无额外依赖).

    输入:  x  (B, L, D)   已经过 pre-norm
    输出:  x' (B, L, D)   三路融合后再经第二个 LN
    """

    def __init__(self, d_model: int, seq_len: int):
        super().__init__()
        # 全局分支: 在时间步维度做全局线性混合 (等价于 MLP on time axis)
        self.global_extractor = nn.Linear(seq_len, seq_len)
        # 局部分支: 在特征序列上做 kernel=3 的因果/对称卷积 (same padding)
        self.local_extractor = nn.Conv1d(
            in_channels=seq_len, out_channels=seq_len,
            kernel_size=3, stride=1, padding=1
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, D)"""
        # 全局分支: (B,L,D) -> (B,D,L) -> Linear(L) -> (B,D,L) -> (B,L,D)
        x_t = x.permute(0, 2, 1)                      # (B, D, L)
        x_global = self.global_extractor(x_t)          # (B, D, L)
        x_global = x_global.permute(0, 2, 1)           # (B, L, D)

        # 局部分支: Conv1d 操作在 (B, L, D) 上，把 L 当 in_channels
        x_local = self.local_extractor(x)              # (B, L, D)

        # 三路加和 + 第二层 LN
        out = self.norm2(x_global + x_local + x)
        return out


# ---------------------------------------------------------------------------
# UniModalBSSM — 单模态双向 SSM (Bi-directional Selective Scanning)
# ---------------------------------------------------------------------------
class UniModalBSSM(nn.Module):
    """
    双向选择性扫描模块.

    实现方式: 复用 MambaSSM (SISO kernel) 做正向扫描，
              flip 序列后再扫一次，得到反向扫描结果。
    两路结果分别用门控向量 z 相乘后相加，最后接 out_proj。

    若 mamba_ssm 不可用，fallback 为双向 GRU (纯 PyTorch)。
    """

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4,
                 expand: int = 2):
        super().__init__()
        self.d_model = d_model

        self.use_mamba = False
        if _MAMBA_AVAILABLE and MambaSSM is not None:
            try:
                # 正向/反向各一套参数（与 ISM 双向扫描逻辑一致）
                self.mamba_fwd = MambaSSM(
                    d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand
                )
                self.mamba_bwd = MambaSSM(
                    d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand
                )
                self.use_mamba = True
            except Exception:
                self.use_mamba = False

        # Fallback: 双向 GRU（当 mamba_simple/mamba2_simple 不可用或运行失败时启用）
        self.bigru = None
        self.gru_proj = None
        if not self.use_mamba:
            self.bigru = nn.GRU(
                d_model, d_model, num_layers=1,
                batch_first=True, bidirectional=True
            )
            self.gru_proj = nn.Linear(d_model * 2, d_model)

        # 门控向量投影 (共享 z)
        self.gate_proj = nn.Linear(d_model, d_model)
        # 输出投影
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, D)"""
        z = torch.sigmoid(self.gate_proj(x))           # (B, L, D) 门控

        if self.use_mamba:
            # 正向扫描
            y_fwd = self.mamba_fwd(x)                  # (B, L, D)
            # 反向扫描: flip -> Mamba -> flip back
            x_flip = x.flip(dims=[1])
            y_bwd = self.mamba_bwd(x_flip).flip(dims=[1])  # (B, L, D)
        else:
            out_gru, _ = self.bigru(x)                 # (B, L, 2D)
            out_gru = self.gru_proj(out_gru)           # (B, L, D)
            y_fwd = y_bwd = out_gru * 0.5              # 均分模拟双向

        # 门控融合: (y_fwd ⊗ z) ⊕ (y_bwd ⊗ z)
        y = y_fwd * z + y_bwd * z                      # (B, L, D)
        return self.out_proj(y)                        # (B, L, D)


# ---------------------------------------------------------------------------
# ISMBlock — 完整 ISM 模块
# ---------------------------------------------------------------------------
class ISMBlock(nn.Module):
    """
    单个 ISM block:  LN → GLCE → BSSM → residual

    可堆叠多层 (ism_depth 控制), 每个模态独立一套权重。

    Args:
        d_model:  特征维度 D
        seq_len:  序列长度 L (GLCE 用到, 需与输入一致)
        d_state:  Mamba SSM 状态维度
        d_conv:   Mamba conv 卷积核大小
        expand:   Mamba 扩展比
        dropout:  Dropout 概率
    """

    def __init__(
        self,
        d_model: int,
        seq_len: int = 50,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.glce = GLCE(d_model, seq_len)
        self.bssm = UniModalBSSM(d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, D)
        返回: (B, L, D)
        """
        # Pre-norm
        h = self.norm1(x)
        # GLCE: 全局-局部上下文
        h = self.glce(h)
        # BSSM: 双向 SSM 扫描
        h = self.bssm(h)
        h = self.drop(h)
        # 残差
        return x + h


# ---------------------------------------------------------------------------
# ISMEncoder — 对单路模态堆叠 ism_depth 层 ISMBlock
# ---------------------------------------------------------------------------
class ISMEncoder(nn.Module):
    """
    堆叠 depth 层 ISMBlock，用于单模态的序列建模。

    Args:
        d_model:  特征维度
        seq_len:  序列长度
        depth:    堆叠层数 (对应 ism_depth)
        d_state / d_conv / expand / dropout: 透传给 ISMBlock
    """

    def __init__(
        self,
        d_model: int,
        seq_len: int = 50,
        depth: int = 1,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            ISMBlock(d_model, seq_len=seq_len, d_state=d_state,
                     d_conv=d_conv, expand=expand, dropout=dropout)
            for _ in range(depth)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, D) -> (B, L, D)"""
        for blk in self.blocks:
            x = blk(x)
        return x