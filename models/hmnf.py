import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from models.hmnf_block import HMNFBlock

class CoupledHMNFLayer(nn.Module):
    """
    CoupledHMNFLayer: 单层耦合 HMNF 模块
    
    负责执行一次完整的“自适应输入耦合”和并行 HMNFBlock 处理。
    """
    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        headdim: int = 32,
        ngroups: int = 1,
        chunk_size: int = 256,
    ):
        super().__init__()
        self.d_model = d_model
        
        # ====================================================================
        # 1. 全局上下文融合层 (Global Context Fusion Layer)
        # ====================================================================
        # 输入: [B, L, 3*D] -> 输出: [B, L, D]
        self.context_fusion = nn.Linear(3 * d_model, d_model)
        
        # ====================================================================
        # 2. 三个并行的 HMNFBlock
        # ====================================================================
        self.block_a = HMNFBlock(
            d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand, 
            dropout=dropout, headdim=headdim, ngroups=ngroups, chunk_size=chunk_size
        )
        
        self.block_v = HMNFBlock(
            d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand, 
            dropout=dropout, headdim=headdim, ngroups=ngroups, chunk_size=chunk_size
        )
        
        self.block_l = HMNFBlock(
            d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand, 
            dropout=dropout, headdim=headdim, ngroups=ngroups, chunk_size=chunk_size
        )

    def forward(
        self, 
        x_a: torch.Tensor, 
        x_v: torch.Tensor, 
        x_l: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        # 1. 生成全局上下文
        concat_features = torch.cat([x_a, x_v, x_l], dim=-1)
        global_context = self.context_fusion(concat_features)
        
        # 2. 准备正向和反向上下文
        fwd_context = global_context
        bwd_context = torch.flip(global_context, dims=[1])
        
        # 3. 并行处理
        out_a = self.block_a(x_a, fwd_context=fwd_context, bwd_context=bwd_context)
        out_v = self.block_v(x_v, fwd_context=fwd_context, bwd_context=bwd_context)
        out_l = self.block_l(x_l, fwd_context=fwd_context, bwd_context=bwd_context)
        
        return out_a, out_v, out_l


class CoupledHMNF(nn.Module):
    """
    CoupledHMNF: Hierarchical Multimodal Mamba Fusion (Multi-layer)
    
    核心机制: Stacked Coupled Bidirectional Mamba2
    堆叠多个 CoupledHMNFLayer，实现深度的多模态交互。
    """
    def __init__(
        self,
        d_model: int,
        num_layers: int = 1,  # 新增层数参数
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        headdim: int = 32,
        ngroups: int = 1,
        chunk_size: int = 256,
    ):
        super().__init__()
        self.num_layers = num_layers
        
        # 堆叠多个层
        self.layers = nn.ModuleList([
            CoupledHMNFLayer(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                dropout=dropout,
                headdim=headdim,
                ngroups=ngroups,
                chunk_size=chunk_size
            ) for _ in range(num_layers)
        ])

    def forward(
        self, 
        x_a: torch.Tensor, 
        x_v: torch.Tensor, 
        x_l: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x_a: Audio input (B, L, D)
            x_v: Visual input (B, L, D)
            x_l: Language input (B, L, D)
            
        Returns:
            out_a, out_v, out_l: Processed outputs (B, L, D)
        """
        # 确保序列长度一致
        assert x_a.shape[1] == x_v.shape[1] == x_l.shape[1], "Sequence lengths must be equal for coupling."
        
        current_a, current_v, current_l = x_a, x_v, x_l
        
        # 逐层处理
        for layer in self.layers:
            current_a, current_v, current_l = layer(current_a, current_v, current_l)
            
        return current_a, current_v, current_l

if __name__ == "__main__":
    # 测试代码
    print("Testing Multi-layer CoupledHMNF...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 参数
    B, L, D = 2, 32, 64
    num_layers = 3
    
    model = CoupledHMNF(d_model=D, num_layers=num_layers, headdim=16).to(device)
    
    x_a = torch.randn(B, L, D).to(device)
    x_v = torch.randn(B, L, D).to(device)
    x_l = torch.randn(B, L, D).to(device)
    
    try:
        out_a, out_v, out_l = model(x_a, x_v, x_l)
        print(f"Input shapes: {x_a.shape}")
        print(f"Output shapes: {out_a.shape}, {out_v.shape}, {out_l.shape}")
        print(f"Layers: {len(model.layers)}")
        print("Test Passed!")
    except Exception as e:
        print(f"Test Failed: {e}")
