"""
GCMN Fusion: 三流门控跨模态融合 (CAGMamba 对齐)
================================================

来源: CAGMamba (Jiao et al., 2026) 公式 (17)-(23)

在 CoupledMamba3Fork 的跨模态输出基础上, 增加:
  1. 单模态保留路径 (Unimodal BSSM)
  2. Cross-Modal BSSM (拼接三模态 → 联合扫描)
  3. Gated Fusion: F_final = F_uni + Gate ⊙ F_cross

设计思想:
  当模态间一致时, Gate→1, 充分融合;
  当模态冲突时, Gate→0, 退回到单模态特征, 防止污染.
"""

from __future__ import annotations

from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class UnimodalBSSM(nn.Module):
    """
    轻量级单模态 BSSM: 保留模态特有信息, 防止被跨模态融合覆盖.

    结构: LayerNorm → Linear(expand) → SiLU → Linear(back) → Residual
    """

    def __init__(self, d_model: int, expand: int = 2):
        super().__init__()
        hidden = d_model * expand
        self.norm = nn.LayerNorm(d_model)
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.SiLU(),
            nn.Linear(hidden, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D) 或 (B, D)
        return x + self.net(self.norm(x))


class CrossModalBSSM(nn.Module):
    """
    跨模态 BSSM: 拼接三模态特征后做联合扫描, 然后解耦回各自模态.

    结构: Concat → LayerNorm → Linear → SiLU → Linear → Split → 3×Project
    """

    def __init__(self, d_model: int, expand: int = 2):
        super().__init__()
        fused_dim = 3 * d_model
        hidden = fused_dim * expand
        self.norm = nn.LayerNorm(fused_dim)
        self.shared = nn.Sequential(
            nn.Linear(fused_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, fused_dim),
        )
        # 解耦投影: fused → L/A/V
        self.proj_L = nn.Linear(fused_dim, d_model)
        self.proj_A = nn.Linear(fused_dim, d_model)
        self.proj_V = nn.Linear(fused_dim, d_model)

    def forward(
        self, x_l: torch.Tensor, x_a: torch.Tensor, x_v: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        fused = torch.cat([x_l, x_a, x_v], dim=-1)        # (B,*,3D)
        enhanced = self.shared(self.norm(fused)) + fused   # Residual
        return (
            self.proj_L(enhanced),
            self.proj_A(enhanced),
            self.proj_V(enhanced),
        )


class GatedCrossModalFusion(nn.Module):
    """
    三流门控融合: unimodal + cross-modal → gated residual.

    公式 (21)-(23):
        G_t = σ(W_g [F_t^cross ∥ F_a^cross ∥ F_v^cross])
        F_t^final = F_t^uni + G_t ⊙ F_t^cross
    """

    def __init__(self, d_model: int, gate_expand: int = 2):
        super().__init__()
        gate_in = 3 * d_model
        gate_hidden = d_model * gate_expand
        self.gate_mlp = nn.Sequential(
            nn.Linear(gate_in, gate_hidden),
            nn.SiLU(),
            nn.Linear(gate_hidden, 3 * d_model),  # 同时出三模态门控
            nn.Sigmoid(),                            # (0,1) 门控
        )

    def forward(
        self,
        uni_l: torch.Tensor, uni_a: torch.Tensor, uni_v: torch.Tensor,
        cross_l: torch.Tensor, cross_a: torch.Tensor, cross_v: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # 1) 联合门控: 所有跨模态特征一起决定每个模态的门
        gate_input = torch.cat([cross_l, cross_a, cross_v], dim=-1)
        gates = self.gate_mlp(gate_input)           # (B,*,3D)
        g_l, g_a, g_v = gates.chunk(3, dim=-1)      # 各 (B,*,D)

        # 2) 残差门控融合: F_final = F_uni + Gate ⊙ F_cross
        out_l = uni_l + g_l * cross_l
        out_a = uni_a + g_a * cross_a
        out_v = uni_v + g_v * cross_v

        return out_l, out_a, out_v


class GCMNFusionModule(nn.Module):
    """
    GCMN 三流门控融合模块 (CAGMamba 对齐).

    用法:
        gcmn = GCMNFusionModule(d_model=128)
        out_l, out_a, out_v = gcmn(
            cross_l, cross_a, cross_v,    # CoupledMamba3Fork 的输出
            x_l, x_a, x_v,                # ISM 的原始输出 (池化前)
        )
    """

    def __init__(
        self,
        d_model: int = 128,
        unimodal_expand: int = 2,
        cross_expand: int = 2,
        gate_expand: int = 2,
    ):
        super().__init__()
        # 单模态保留路径
        self.uni_L = UnimodalBSSM(d_model, unimodal_expand)
        self.uni_A = UnimodalBSSM(d_model, unimodal_expand)
        self.uni_V = UnimodalBSSM(d_model, unimodal_expand)

        # 跨模态 BSSM
        self.cross_bssm = CrossModalBSSM(d_model, cross_expand)

        # 门控融合
        self.gated_fusion = GatedCrossModalFusion(d_model, gate_expand)

        # 残差 norm
        self.norm_L = nn.LayerNorm(d_model)
        self.norm_A = nn.LayerNorm(d_model)
        self.norm_V = nn.LayerNorm(d_model)

    def forward(
        self,
        # CoupledMamba3Fork 的跨模态输出 (已含残差+LN)
        cross_l: torch.Tensor,
        cross_a: torch.Tensor,
        cross_v: torch.Tensor,
        # ISM 输出 (池化前), 用于单模态保留
        raw_l: Optional[torch.Tensor] = None,
        raw_a: Optional[torch.Tensor] = None,
        raw_v: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            cross_*: CoupledMamba3Fork 输出的跨模态特征 (B, L, D)
            raw_*: ISM 输出的原始特征, 用于 unimodal 路径.
                   如果为 None, 用 cross_* 自身做单模态 (残差值).
        Returns:
            out_l, out_a, out_v: 门控融合后的特征 (B, L, D)
        """
        # 回退: 没有单模态路径时用 cross 自身
        if raw_l is None:
            raw_l, raw_a, raw_v = cross_l, cross_a, cross_v

        # 1) 单模态保留: 防止跨模态信息覆盖
        uni_l = self.uni_L(raw_l)
        uni_a = self.uni_A(raw_a)
        uni_v = self.uni_V(raw_v)

        # 2) 跨模态增强: BSSM 联合扫描
        enh_l, enh_a, enh_v = self.cross_bssm(cross_l, cross_a, cross_v)

        # 3) 门控融合
        out_l, out_a, out_v = self.gated_fusion(
            uni_l, uni_a, uni_v,
            enh_l, enh_a, enh_v,
        )

        # 4) 最终 norm
        return self.norm_L(out_l), self.norm_A(out_a), self.norm_V(out_v)
