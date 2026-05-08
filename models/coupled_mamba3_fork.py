"""
Coupled-Mamba3 Fork: 基于 Mamba-3 类的跨模态 Q/K/V 融合
=====================================================

本模块直接 fork 自 [`Mamba3`](H-DCD/mamba/mamba_ssm/modules/mamba3.py:26),
保留其 *几乎所有* 工程优势:
    ✓ data-dependent A: -softplus(dd_A) + A_floor 截断
    ✓ Trapezoidal 离散化 (trap)
    ✓ RoPE 复数旋转位置编码 (rope_fraction / num_rope_angles)
    ✓ B / C 上的 RMSNormGated (B_norm / C_norm)
    ✓ B_bias / C_bias 学习偏置
    ✓ MIMO 支持 (mimo_rank > 1 时启用 mimo_x/mimo_z/mimo_o)
    ✓ is_outproj_norm 选项 (RMSNormGated 在 out_proj 前)
    ✓ D 跳连 + Q/K bias
    ✓ 兼容 Mamba-3 SISO / MIMO Triton kernel
    ✓ bf16 兼容 (kernel 内部处理)

改造点 (策略 A: 严格跨模态):
    对每个目标模态 tgt ∈ {audio, visual, lexical}:
        z_tgt, x_tgt, dd_dt, dd_A, trap, angles  ← in_proj_tgt(u_tgt)
        B_src, C_tgt                             ← 三套独立投影
            C_tgt = c_proj_tgt(u_tgt)            # Q 来自 tgt
            B_src = w_src[0] * b_proj_s0(u_s0) + w_src[1] * b_proj_s1(u_s1)
            V_src = w_src[0] * x_proj_s0(u_s0) + w_src[1] * x_proj_s1(u_s1)
        其中 w_src = softmax(weight_net(concat(u_tgt, u_s0, u_s1)))
        然后调用与 Mamba3.forward 完全相同的 kernel 调用与后处理流程。
    -> tgt 控制 "如何衰减/旋转/读出", src 提供 "写入什么/如何写入",
       完美对齐 cross-attention 语义, 同时复用 Mamba-3 的全部数值技巧。

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
# CrossMamba3Cell: fork 自 Mamba3, 改 in_proj 切分为跨模态版
# ============================================================================
class CrossMamba3Cell(nn.Module):
    """
    单个目标模态 tgt 的 cross-modal Mamba-3 cell。

    与官方 [`Mamba3`](H-DCD/mamba/mamba_ssm/modules/mamba3.py:26) 的差异:
        - in_proj 拆为 *3 份* (每模态一份), 而非合并到一个 Linear
            * tgt 模态的 in_proj_tgt:  z + x + dd_dt + dd_A + trap + angles
              (即 V 的来源部分, 但实际 V/B 会在 forward 里被替换)
            * 每个源模态 src 一套 b_proj_src + x_proj_src (用于产 K/B 和 V/x 候选)
            * 每个 tgt 模态一个 c_proj_tgt (Q/C, 因为 Q 始终来自 tgt)
        - forward 的语义: 接收 (u_tgt, u_s0, u_s1, w_src) 三份特征 + 源权重
        - 其余 RMSNormGated / kernel 调用 / MIMO / D-skip 完全照抄

    注意:
        - 不含 conv1d (Mamba-3 本来就没有, 与 Mamba-2 不同)
        - 不实现 step() (训练优先, 推理走 forward 即可)
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
        modality_keys: Tuple[str, str, str] = ("audio", "visual", "lexical"),
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

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

        self.modality_keys = modality_keys
        self.num_modalities = len(modality_keys)

        # ---------- in_proj 拆分: tgt 部分 (z + ctrl) ----------
        # tgt 出: z (d_inner) + dd_dt (nheads) + dd_A (nheads) + trap (nheads) + angles (num_rope_angles)
        # 注: x 也由 tgt 出, 但策略 A 中 x 会被 src 加权版本替换, 这里作为 *默认 fallback*
        d_tgt_proj = (
            2 * self.d_inner            # z + x_tgt_default
            + 3 * self.nheads           # dd_dt + dd_A + trap
            + self.num_rope_angles      # angles
        )
        self.in_proj_tgt = nn.Linear(d_model, d_tgt_proj, bias=False, **factory_kwargs)

        # ---------- C (=Q) 投影: 来自 tgt ----------
        # C 形状: (B, L, d_state * num_bc_heads * mimo_rank)
        d_c = d_state * self.num_bc_heads * self.mimo_rank
        self.c_proj_tgt = nn.Linear(d_model, d_c, bias=False, **factory_kwargs)

        # ---------- B (=K) 和 V (=x) 投影: 每个源模态各一套 ----------
        d_b = d_state * self.num_bc_heads * self.mimo_rank
        self.b_projs = nn.ModuleDict({
            m: nn.Linear(d_model, d_b, bias=False, **factory_kwargs)
            for m in modality_keys
        })
        self.v_projs = nn.ModuleDict({
            m: nn.Linear(d_model, self.d_inner, bias=False, **factory_kwargs)
            for m in modality_keys
        })

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

    # ------------------------------------------------------------------------
    def forward(
        self,
        u_tgt: torch.Tensor,
        u_src0: torch.Tensor,
        u_src1: torch.Tensor,
        w_src: torch.Tensor,
        src_keys: Tuple[str, str],
        cu_seqlens: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            u_tgt:   (B, L, d_model)    目标模态特征 -> z, ctrl, C(Q)
            u_src0:  (B, L, d_model)
            u_src1:  (B, L, d_model)
            w_src:   (B, L, 2)          softmax 后的源权重 (来自 weight_net)
            src_keys: ("src0_name", "src1_name")
            cu_seqlens: 与 Mamba3 一致 (可选, 用于变长序列)
        Returns:
            out_tgt: (B, L, d_model)
        """
        batch, seqlen, _ = u_tgt.shape
        s0_key, s1_key = src_keys

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
        # x_default 暂不使用 (策略 A 用 src 加权版本); 保留参数以便后续策略切换

        # ---------------- 2) tgt 出 C (Q) ----------------
        C = self.c_proj_tgt(u_tgt)
        C = rearrange(C, "b l (r g n) -> b l r g n",
                      r=self.mimo_rank, g=self.num_bc_heads)

        # ---------------- 3) src 加权出 B (K) 和 V (x) ----------------
        w0 = w_src[..., 0:1]                                 # (B, L, 1)
        w1 = w_src[..., 1:2]
        # B (K)
        B0 = self.b_projs[s0_key](u_src0)
        B1 = self.b_projs[s1_key](u_src1)
        B = w0 * B0 + w1 * B1
        B = rearrange(B, "b l (r g n) -> b l r g n",
                      r=self.mimo_rank, g=self.num_bc_heads)
        # V (x)
        V0 = self.v_projs[s0_key](u_src0)
        V1 = self.v_projs[s1_key](u_src1)
        x = w0 * V0 + w1 * V1
        x = rearrange(x, "b l (h p) -> b l h p", p=self.headdim)

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
# CoupledMamba3Fork: 三模态 cross-modal 主模块
# ============================================================================
class CoupledMamba3Fork(nn.Module):
    """
    三模态跨模态融合主类。每层包含 3 个 [`CrossMamba3Cell`](H-DCD/models/coupled_mamba3_fork.py:67),
    每个对应一个 tgt 模态。

    与 [`CrossMamba3`](H-DCD/models/cross_mamba3.py:415) 的核心差异:
        - CrossMamba3:        从零造一个简化版 SSM cell (没有 RMSNormGated/MIMO/RoPE 全套)
        - CoupledMamba3Fork:  直接 fork Mamba3, 享受 *全部* Mamba-3 工程优势

    与 [`CoupledMamba3`](H-DCD/models/coupled_mamba3.py) (状态层耦合版) 的差异:
        - 状态层耦合: h_tgt += w * P(h_src), 信息只在状态加性混合
        - cross Q/K/V (本模块): 信息直接在 d_state 维做线性注意力深度融合
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
            chunk_size=chunk_size, modality_keys=self.modalities,
            device=device, dtype=dtype,
        )

        # 每个 tgt 模态一个 cell
        self.cells = nn.ModuleDict({
            tgt: CrossMamba3Cell(**cell_kwargs) for tgt in self.modalities
        })

        # 动态源权重网络: concat(u_tgt, u_s0, u_s1) -> 2-dim softmax
        # 为每个 tgt 各一套 weight_net (因为权重语义因 tgt 而异)
        self.weight_nets = nn.ModuleDict({
            tgt: nn.Sequential(
                nn.Linear(self.num_modalities * d_model, d_model),
                nn.Tanh(),
                nn.Linear(d_model, 2),
            )
            for tgt in self.modalities
        })

        # 残差 + LN
        self.layer_norms = nn.ModuleDict({
            m: nn.LayerNorm(d_model) for m in self.modalities
        })

    def _src_keys_for(self, tgt: str) -> Tuple[str, str]:
        return tuple(m for m in self.modalities if m != tgt)  # type: ignore

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
            s0_key, s1_key = self._src_keys_for(tgt)
            u_tgt = feats[tgt]
            u_s0 = feats[s0_key]
            u_s1 = feats[s1_key]

            # 动态源权重
            w_logits = self.weight_nets[tgt](torch.cat([u_tgt, u_s0, u_s1], dim=-1))
            w_src = F.softmax(w_logits, dim=-1)              # (B, L, 2)

            # cross-modal cell
            y = self.cells[tgt](u_tgt, u_s0, u_s1, w_src, (s0_key, s1_key),
                                cu_seqlens=cu_seqlens)

            # 残差 + LN
            outs[tgt] = self.layer_norms[tgt](y + u_tgt)

        return outs["audio"], outs["visual"], outs["lexical"]


# ============================================================================
# Smoke Test
# ============================================================================
if __name__ == "__main__":
    import sys
    print("=" * 64)
    print("CoupledMamba3Fork (基于 Mamba3 类的跨模态 fork)")
    print("=" * 64)
    print(f"MAMBA3_AVAILABLE = {MAMBA3_AVAILABLE}")
    print(f"RMSNormGated     = {RMSNormGated is not None}")

    if not (MAMBA3_AVAILABLE and RMSNormGated is not None and torch.cuda.is_available()):
        print("\n[Skip] 需要 CUDA + 完整 mamba_ssm 安装才能跑 kernel; 这里只验证可导入。")
        sys.exit(0)

    device = "cuda"
    B, L, D = 2, 64, 128
    model = CoupledMamba3Fork(
        d_model=D,
        d_state=64,
        expand=2,
        headdim=32,           # d_inner = 256, nheads = 8
        ngroups=1,
        rope_fraction=0.5,
        is_mimo=False,
        chunk_size=64,
    ).to(device)

    x_a = torch.randn(B, L, D, device=device, requires_grad=True)
    x_v = torch.randn(B, L, D, device=device, requires_grad=True)
    x_l = torch.randn(B, L, D, device=device, requires_grad=True)

    out_a, out_v, out_l = model(x_a, x_v, x_l)
    print(f"\nInput  : {x_a.shape}")
    print(f"Output : audio={out_a.shape}, visual={out_v.shape}, lexical={out_l.shape}")

    loss = out_a.sum() + out_v.sum() + out_l.sum()
    loss.backward()
    print(f"\nGrad  audio  : {x_a.grad.abs().mean().item():.4e}")
    print(f"Grad  visual : {x_v.grad.abs().mean().item():.4e}")
    print(f"Grad  lexical: {x_l.grad.abs().mean().item():.4e}")

    assert not (torch.isnan(out_a).any() or torch.isnan(out_v).any() or torch.isnan(out_l).any())
    print("\nNumerical OK.")
    print("=" * 64)