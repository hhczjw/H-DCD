import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

try:
    from mamba_ssm import Mamba2
except ImportError:
    print("Warning: mamba_ssm not installed. HMNFBlock will not work.")
    Mamba2 = None

# ============================================================================
# RMSNorm (参考 DepMamba/Mamba 官方实现)
# ============================================================================
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return output * self.weight

class HMNFBlock(nn.Module):
    """
    HMNFBlock: Hierarchical Multimodal Network Fusion Block
    
    参考 DepMamba 的设计理念，结合 Mamba2 的高效序列建模能力。
    
    架构流程:
    1. 门控分支 (Gating Branch): Linear -> SiLU -> Gate
    2. 双向处理分支 (Bidirectional Processing Branches):
       - Forward: Conv1d -> Residual -> Linear -> Mamba2
       - Backward: Flip -> Conv1d -> Residual -> Linear -> Mamba2 -> Flip Back
    3. 融合输出 (Fusion): (Fwd * Gate + Bwd * Gate) -> Linear -> RMSNorm -> Global Residual
    """
    def __init__(
        self, 
        d_model: int, 
        d_state: int = 64, 
        d_conv: int = 4, 
        expand: int = 2,
        dropout: float = 0.1,
        headdim: int = 64,
        ngroups: int = 1,
        chunk_size: int = 256,
        layer_idx: Optional[int] = None,
        device=None,
        dtype=None,
    ):
        """
        Args:
            d_model: 输入特征维度
            d_state: Mamba SSM 状态维度
            d_conv: Mamba 内部卷积核大小
            expand: Mamba 扩展因子
            dropout: Dropout 比率
            headdim: Mamba2 head dimension
            ngroups: Mamba2 groups
            chunk_size: Mamba2 chunk size
            layer_idx: 层索引 (用于初始化)
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        
        if Mamba2 is None:
            raise ImportError("mamba_ssm.Mamba2 is required. Please install it.")

        self.d_model = d_model
        self.layer_idx = layer_idx

        # ====================================================================
        # 1. 门控分支 (Gating Branch)
        # ====================================================================
        self.gate_linear = nn.Linear(d_model, d_model, **factory_kwargs)
        self.gate_act = nn.SiLU()

        # ====================================================================
        # 2. 双向处理分支 (Bidirectional Processing Branches)
        # ====================================================================
        
        # --- 正向路径 (Forward Path) ---
        # Conv1d: kernel_size=3, padding=1 保持序列长度不变
        self.fwd_conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, **factory_kwargs)
        self.fwd_linear = nn.Linear(d_model, d_model, **factory_kwargs)
        self.fwd_mamba = Mamba2(
            d_model=d_model, 
            d_state=d_state, 
            d_conv=d_conv, 
            expand=expand,
            headdim=headdim,
            ngroups=ngroups,
            chunk_size=chunk_size,
            **factory_kwargs
        )

        # --- 反向路径 (Backward Path) ---
        self.bwd_conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, **factory_kwargs)
        self.bwd_linear = nn.Linear(d_model, d_model, **factory_kwargs)
        self.bwd_mamba = Mamba2(
            d_model=d_model, 
            d_state=d_state, 
            d_conv=d_conv, 
            expand=expand,
            headdim=headdim,
            ngroups=ngroups,
            chunk_size=chunk_size,
            **factory_kwargs
        )

        # ====================================================================
        # 3. 融合与输出 (Fusion & Output)
        # ====================================================================
        self.out_linear = nn.Linear(d_model, d_model, **factory_kwargs)
        self.norm = RMSNorm(d_model) # 使用 RMSNorm 替代 LayerNorm
        self.dropout = nn.Dropout(dropout)
        
        # 初始化权重
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        参考 DepMamba/Mamba 的权重初始化策略
        """
        if isinstance(module, nn.Linear):
            if module.bias is not None:
                if not getattr(module.bias, "_no_reinit", False):
                    nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, x: torch.Tensor, fwd_context: Optional[torch.Tensor] = None, bwd_context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: 输入张量 (Batch, Seq_Len, Dim)
            fwd_context: 正向全局上下文 (Batch, Seq_Len, Dim), 可选
            bwd_context: 反向全局上下文 (Batch, Seq_Len, Dim), 必须是已翻转的, 可选
        Returns:
            out: 输出张量 (Batch, Seq_Len, Dim)
        """
        B, L, D = x.shape
        residual_global = x

        # ====================================================================
        # 1. 门控信号生成
        # ====================================================================
        # x -> Linear -> SiLU -> gate
        gate = self.gate_act(self.gate_linear(x))

        # 准备 Conv1d 输入: (B, L, D) -> (B, D, L)
        x_transposed = x.transpose(1, 2)

        # ====================================================================
        # 2. 正向路径 (Forward Path)
        # ====================================================================
        # Conv1d
        fwd_conv_out = self.fwd_conv(x_transposed)
        fwd_conv_out = fwd_conv_out.transpose(1, 2) # (B, D, L) -> (B, L, D)
        
        # Residual: x + conv_out
        x_fwd = x + fwd_conv_out
        
        # Linear -> Mamba2
        x_fwd = self.fwd_linear(x_fwd)
        
        # --- Adaptive Input Coupling (Forward) ---
        if fwd_context is not None:
            x_fwd = x_fwd + fwd_context
        
        # Ensure contiguous for Mamba2
        x_fwd = x_fwd.contiguous()
             
        out_fwd = self.fwd_mamba(x_fwd)

        # ====================================================================
        # 3. 反向路径 (Backward Path)
        # ====================================================================
        # Flip input
        x_flip = torch.flip(x, dims=[1])
        x_flip_transposed = x_flip.transpose(1, 2)
        
        # Conv1d
        bwd_conv_out = self.bwd_conv(x_flip_transposed)
        bwd_conv_out = bwd_conv_out.transpose(1, 2) # (B, L, D)
        
        # Residual: x_flip + conv_out
        x_bwd = x_flip + bwd_conv_out
        
        # Linear -> Mamba2
        x_bwd = self.bwd_linear(x_bwd)
        
        # --- Adaptive Input Coupling (Backward) ---
        if bwd_context is not None:
            x_bwd = x_bwd + bwd_context
        
        # Ensure contiguous for Mamba2
        x_bwd = x_bwd.contiguous()
            
        out_bwd = self.bwd_mamba(x_bwd)
        
        # Flip back to normal order
        out_bwd = torch.flip(out_bwd, dims=[1])

        # ====================================================================
        # 4. 融合与输出 (Fusion & Output)
        # ====================================================================
        # Element-wise Multiply with Gate
        fwd_gated = out_fwd * gate
        bwd_gated = out_bwd * gate
        
        # Sum
        summed = fwd_gated + bwd_gated
        
        # Linear -> RMSNorm
        out = self.out_linear(summed)
        out = self.norm(out)
        out = self.dropout(out)
        
        # Global Residual Connection
        return residual_global + out

if __name__ == "__main__":
    # 简单测试
    print("Testing HMNFBlock...")
    if Mamba2 is not None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        block = HMNFBlock(d_model=64).to(device)
        x = torch.randn(2, 32, 64).to(device)
        y = block(x)
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {y.shape}")
        print("Test Passed!")
    else:
        print("Skipping test because Mamba2 is not installed.")
