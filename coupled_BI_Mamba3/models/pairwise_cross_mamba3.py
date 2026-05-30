"""
Pairwise Cross-Mamba3: 基于 Mamba-3 类的细粒度成对跨模态 Q/K/V 融合
=====================================================

本模块受经典 MulT (Pairwise Crossmodal Attention) 启发。
    * 对每个目标模态 tgt (如 Text):
        1) T 接受 Audio (A -> T):
           T 作为 Query, Audio 作为 Key/Value。经过 PairwiseCrossMamba3Cell 产生 T_A
        2) T 接受 Video (V -> T):
           T 作为 Query, Video 作为 Key/Value。经过 PairwiseCrossMamba3Cell 产生 T_V
        3) 融合:
           T_new = LayerNorm( T + Linear(concat(T_A, T_V)) )  (或其他轻量级融合)
           
    总共包含 3x2 = 6 个 PairwiseCrossMamba3Cell。

接口:
    输入  x_audio / x_visual / x_lexical: (B, L, d_model)
    输出  out_audio / out_visual / out_lexical: (B, L, d_model)
"""

from __future__ import annotations

import math
from typing import Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# ---------------------------------------------------------------------------
# Mamba-3 依赖 (与官方 mamba3.py 完全一致的导入)
# ---------------------------------------------------------------------------
try:
    from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated
except ImportError:
    RMSNormGated = None

try:
    from mamba_ssm.ops.tilelang.mamba3.mamba3_mimo import mamba3_mimo as mamba3_mimo_combined
except ImportError:
    mamba3_mimo_combined = None

try:
    from mamba_ssm.ops.triton.mamba3.mamba3_siso_combined import mamba3_siso_combined
    MAMBA3_AVAILABLE = True
except ImportError:
    mamba3_siso_combined = None
    MAMBA3_AVAILABLE = False


# ============================================================================
# PairwiseCrossMamba3Cell: fork 自 Mamba3, 一对一跨模态
# ============================================================================
class PairwiseCrossMamba3Cell(nn.Module):
    """
    一对一的 cross-modal Mamba-3 cell (Src -> Tgt)。

    与 Coupled 版本的差异:
        - src_key 只有一个。不再做 src 的加权融合。
        - B 和 V (x) 的投影直接从这个单一的 u_src 产生。
    """
    def __init__(
        self,
        d_model: int,
        d_state: int = 128,
        expand: int = 2,
        headdim: int = 64,
        ngroups: int = 1,
        rope_fraction: float = 0.5,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init_floor: float = 1e-4,
        A_floor: float = 1e-4,
        is_outproj_norm: bool = False,
        is_mimo: bool = False,
        mimo_rank: int = 4,
        chunk_size: int = 64,
        v_self_ratio: float = 0.0,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        assert 0.0 <= float(v_self_ratio) <= 1.0, \
            f"v_self_ratio 必须 ∈ [0,1], got {v_self_ratio}"
        self.v_self_ratio = float(v_self_ratio)

        # ---------- 与 Mamba3.__init__ 完全一致的超参 ----------
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.headdim = headdim
        self.chunk_size = chunk_size
        self.A_floor = A_floor
        self.is_outproj_norm = is_outproj_norm
        self.is_mimo = is_mimo
        self.mimo_rank = mimo_rank if is_mimo else 1
        if is_mimo:
            assert mamba3_mimo_combined is not None, "MIMO kernel unavailable"

        self.d_inner = int(expand * d_model)
        assert self.d_inner % headdim == 0, \
            f"d_inner ({self.d_inner}) 必须能被 headdim ({headdim}) 整除"
        self.nheads = self.d_inner // headdim
        self.num_bc_heads = ngroups

        # RoPE
        assert rope_fraction in [0.5, 1.0]
        self.rotary_dim_divisor = int(2 / rope_fraction)
        self.split_tensor_size = int(d_state * rope_fraction)
        if self.split_tensor_size % 2 != 0:
            self.split_tensor_size -= 1
        self.num_rope_angles = self.split_tensor_size // 2
        assert self.num_rope_angles > 0

        # ---------- in_proj 拆分: tgt 部分 (z + ctrl) ----------
        d_tgt_proj = (
            2 * self.d_inner            # z + x_tgt_default
            + 3 * self.nheads           # dd_dt + dd_A + trap
            + self.num_rope_angles      # angles
        )
        self.in_proj_tgt = nn.Linear(d_model, d_tgt_proj, bias=False, **factory_kwargs)

        # ---------- C (=Q) 投影: 来自 tgt ----------
        d_c = d_state * self.num_bc_heads * self.mimo_rank
        self.c_proj_tgt = nn.Linear(d_model, d_c, bias=False, **factory_kwargs)

        # ---------- B (=K) 和 V (=x) 投影: 来自单一源 src ----------
        d_b = d_state * self.num_bc_heads * self.mimo_rank
        self.b_proj_src = nn.Linear(d_model, d_b, bias=False, **factory_kwargs)
        self.v_proj_src = nn.Linear(d_model, self.d_inner, bias=False, **factory_kwargs)

        # ---------- dt_bias (与 Mamba3.__init__ 完全一致) ----------
        _dt = torch.exp(
            torch.rand(self.nheads, device=device, dtype=torch.float32)
            * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        _dt = torch.clamp(_dt, min=dt_init_floor)
        _dt_bias = _dt + torch.log(-torch.expm1(-_dt))
        self.dt_bias = nn.Parameter(_dt_bias, requires_grad=True)
        self.dt_bias._no_weight_decay = True

        # ---------- B / C bias (照抄 Mamba3) ----------
        self.B_bias = nn.Parameter(
            1 + torch.zeros((self.nheads, self.mimo_rank, d_state),
                            dtype=torch.float32, device=device),
            requires_grad=True,
        )
        self.C_bias = nn.Parameter(
            1 + torch.zeros((self.nheads, self.mimo_rank, d_state),
                            dtype=torch.float32, device=device),
            requires_grad=True,
        )

        # ---------- B / C RMSNormGated ----------
        assert RMSNormGated is not None, "请安装 mamba_ssm 以获取 RMSNormGated"
        self.B_norm = RMSNormGated(d_state, eps=1e-5, **factory_kwargs)
        self.C_norm = RMSNormGated(d_state, eps=1e-5, **factory_kwargs)

        # ---------- MIMO 参数 (照抄 Mamba3) ----------
        if self.is_mimo:
            mimo_x_init = torch.ones(self.nheads, self.mimo_rank, headdim, device=device) / self.mimo_rank
            mimo_z_init = torch.ones(self.nheads, self.mimo_rank, headdim, device=device)
            mimo_o_init = torch.ones(self.nheads, self.mimo_rank, headdim, device=device) / self.mimo_rank
            self.mimo_x = nn.Parameter(mimo_x_init, requires_grad=True)
            self.mimo_z = nn.Parameter(mimo_z_init, requires_grad=True)
            self.mimo_o = nn.Parameter(mimo_o_init, requires_grad=True)

        # ---------- D 跳连 ----------
        self.D = nn.Parameter(torch.ones(self.nheads, device=device))
        self.D._no_weight_decay = True

        # ---------- 输出 RMSNormGated (可选) ----------
        if self.is_outproj_norm:
            self.norm = RMSNormGated(
                self.d_inner, eps=1e-5, norm_before_gate=True,
                group_size=headdim, **factory_kwargs,
            )

        # ---------- out_proj ----------
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False, **factory_kwargs)

    def forward(
        self,
        u_tgt: torch.Tensor,
        u_src: torch.Tensor,
        cu_seqlens: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            u_tgt: (B, L, d_model)  目标模态特征 -> z, ctrl, C(Q)
            u_src: (B, L, d_model)  单一源模态特征 -> B(K), x(V)
        Returns:
            out:   (B, L, d_model)
        """
        batch, seqlen, _ = u_tgt.shape

        # ---------------- 1) tgt 出 z + 默认 x + 控制信号 ----------------
        proj_t = self.in_proj_tgt(u_tgt)
        z, x_default, dd_dt, dd_A, trap, angles = torch.split(
            proj_t,
            [self.d_inner, self.d_inner,
             self.nheads, self.nheads, self.nheads,
             self.num_rope_angles],
            dim=-1,
        )
        z = rearrange(z, "b l (h p) -> b l h p", p=self.headdim)
        if self.v_self_ratio > 0.0:
            x_default = rearrange(x_default, "b l (h p) -> b l h p", p=self.headdim)

        # ---------------- 2) tgt 出 C (Q) ----------------
        C = self.c_proj_tgt(u_tgt)
        C = rearrange(C, "b l (r g n) -> b l r g n",
                      r=self.mimo_rank, g=self.num_bc_heads)

        # ---------------- 3) src 出 B (K) 和 V (x) ----------------
        B = self.b_proj_src(u_src)
        B = rearrange(B, "b l (r g n) -> b l r g n",
                      r=self.mimo_rank, g=self.num_bc_heads)
        
        x = self.v_proj_src(u_src)
        x = rearrange(x, "b l (h p) -> b l h p", p=self.headdim)
        
        if self.v_self_ratio > 0.0:
            x = self.v_self_ratio * x_default + (1.0 - self.v_self_ratio) * x

        # ---------------- 4) ADT / DT (照抄 Mamba3) ----------------
        _A = -F.softplus(dd_A.to(torch.float32))
        _A = torch.clamp(_A, max=-self.A_floor)
        DT = F.softplus(dd_dt + self.dt_bias)
        ADT = _A * DT
        DT = rearrange(DT, "b l n -> b n l")
        ADT = rearrange(ADT, "b l n -> b n l")
        trap = rearrange(trap, "b l h -> b h l")
        angles = angles.unsqueeze(-2).expand(-1, -1, self.nheads, -1).to(torch.float32)

        # ---------------- 5) RMSNormGated on B and C ----------------
        B = self.B_norm(B)
        C = self.C_norm(C)

        # ---------------- 6) Mamba-3 kernel (照抄 Mamba3.forward) ----------------
        if self.is_mimo:
            y = mamba3_mimo_combined(
                Q=C, K=B, V=x,
                ADT=ADT, DT=DT, Trap=trap,
                Q_bias=self.C_bias, K_bias=self.B_bias,
                MIMO_V=self.mimo_x, MIMO_Z=self.mimo_z,
                MIMO_Out=self.mimo_o if not self.is_outproj_norm else None,
                Angles=angles, D=self.D,
                Z=z if not self.is_outproj_norm else None,
                chunk_size=self.chunk_size,
                rotary_dim_divisor=self.rotary_dim_divisor,
                dtype=x.dtype,
                return_state=False,
                cu_seqlens=cu_seqlens,
            )
            if self.is_outproj_norm:
                z_e = torch.einsum("blhp,hrp->blrhp", z.float(), self.mimo_z)
                z_e = rearrange(z_e, "b l r h p -> b l r (h p)")
                y = rearrange(y, "b l r h p -> b l r (h p)").float()
                y = self.norm(y, z_e)
                y = rearrange(y, "b l r (h p) -> b l r h p", p=self.headdim)
                y = torch.einsum("blrhp,hrp->blhp", y, self.mimo_o)
            y = rearrange(y, "b l h p -> b l (h p)")
        else:
            y = mamba3_siso_combined(
                Q=C.squeeze(2), K=B.squeeze(2), V=x,
                ADT=ADT, DT=DT, Trap=trap,
                Q_bias=self.C_bias.squeeze(1), K_bias=self.B_bias.squeeze(1),
                Angles=angles, D=self.D,
                Z=z if not self.is_outproj_norm else None,
                chunk_size=self.chunk_size,
                Input_States=None,
                return_final_states=False,
                cu_seqlens=cu_seqlens,
            )
            y = rearrange(y, "b l h p -> b l (h p)")
            if self.is_outproj_norm:
                z_e = rearrange(z, "b l h p -> b l (h p)")
                y = self.norm(y, z_e)

        # ---------------- 7) out_proj ----------------
        out = self.out_proj(y.to(x.dtype))
        return out


# ============================================================================
# PairwiseCrossMamba3Fork: 三模态 两两成对 跨模态融合主模块
# ============================================================================
class PairwiseCrossMamba3Fork(nn.Module):
    """
    三模态细粒度成对跨模态融合主类。每层包含 6 个 PairwiseCrossMamba3Cell:
    A->T, V->T
    T->A, V->A
    T->V, A->V
    """
    def __init__(
        self,
        d_model: int,
        d_state: int = 128,
        expand: int = 2,
        headdim: int = 64,
        ngroups: int = 1,
        rope_fraction: float = 0.5,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init_floor: float = 1e-4,
        A_floor: float = 1e-4,
        is_outproj_norm: bool = False,
        is_mimo: bool = False,
        mimo_rank: int = 4,
        chunk_size: int = 64,
        v_self_ratio: float = 0.0,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.d_model = d_model
        self.modalities = ("audio", "visual", "lexical")
        self.num_modalities = len(self.modalities)

        cell_kwargs = dict(
            d_model=d_model, d_state=d_state, expand=expand, headdim=headdim,
            ngroups=ngroups, rope_fraction=rope_fraction,
            dt_min=dt_min, dt_max=dt_max, dt_init_floor=dt_init_floor, A_floor=A_floor,
            is_outproj_norm=is_outproj_norm, is_mimo=is_mimo, mimo_rank=mimo_rank,
            chunk_size=chunk_size, 
            v_self_ratio=v_self_ratio,
            device=device, dtype=dtype,
        )

        # 6 个独立的一对一 Cell
        self.cells = nn.ModuleDict()
        for tgt in self.modalities:
            for src in self._src_keys_for(tgt):
                self.cells[f"{src}_to_{tgt}"] = PairwiseCrossMamba3Cell(**cell_kwargs)

        # 融合网络: 将2个源的成对输出 concat 起来，投影回 d_model
        self.fusion_nets = nn.ModuleDict({
            tgt: nn.Sequential(
                nn.Linear(2 * d_model, d_model),
                nn.Dropout(0.1) # 可选加入轻量级 dropout 防止过拟合
            )
            for tgt in self.modalities
        })

        # 残差 + LN
        self.layer_norms = nn.ModuleDict({
            m: nn.LayerNorm(d_model) for m in self.modalities
        })

    def _src_keys_for(self, tgt: str) -> Tuple[str, str]:
        return tuple(m for m in self.modalities if m != tgt)

    def forward(
        self,
        x_audio: torch.Tensor,
        x_visual: torch.Tensor,
        x_lexical: torch.Tensor,
        cu_seqlens: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        feats: Dict[str, torch.Tensor] = {
            "audio": x_audio, "visual": x_visual, "lexical": x_lexical,
        }
        outs: Dict[str, torch.Tensor] = {}

        for tgt in self.modalities:
            u_tgt = feats[tgt]
            s0_key, s1_key = self._src_keys_for(tgt)
            u_s0 = feats[s0_key]
            u_s1 = feats[s1_key]

            # 1. 分别提取两两交叉特征
            y_s0 = self.cells[f"{s0_key}_to_{tgt}"](u_tgt, u_s0, cu_seqlens=cu_seqlens)
            y_s1 = self.cells[f"{s1_key}_to_{tgt}"](u_tgt, u_s1, cu_seqlens=cu_seqlens)
            
            # 2. 拼接 + 线性融合
            y_fused = self.fusion_nets[tgt](torch.cat([y_s0, y_s1], dim=-1))
            
            # 3. 残差 + LN
            outs[tgt] = self.layer_norms[tgt](y_fused + u_tgt)

        return outs["audio"], outs["visual"], outs["lexical"]

__all__ = ["PairwiseCrossMamba3Cell", "PairwiseCrossMamba3Fork", "MAMBA3_AVAILABLE"]
