"""
Cross-Mamba3: Cross-Modal Q/K/V Fusion via Mamba-3 SSD Kernel
==============================================================

本模块在 Coupled-Mamba3 的基础上, 进一步把 Mamba-3 SSD kernel 中的
Q / K / V 三个槽位用于 *跨模态* 融合, 而非传统 self-attention 的 "同源 QKV"。

核心思想:
    Mamba-3 的 SSD kernel 在数学上等价于 "线性注意力 + 时序状态衰减":
        - V (B, L, H, Pv): 每个 token 写入状态的内容    (= SSM 输入 x)
        - K (B, L, Hqk, P): 每个 token 写入状态的方式    (= 离散化的 B 矩阵)
        - Q (B, L, Hqk, P): 每个 token 从状态读取的方式  (= 离散化的 C 矩阵)
    kernel 本身不关心 Q/K/V 来自何处, 因此可把不同模态分别放入 Q / K / V,
    实现一个 "时序+跨模态" 融合的 cross-attention 变体, 同时享受:
        * 数据依赖 A (-softplus) + 梯形离散化 (trap)
        * RoPE 位置编码 (kernel 内置)
        * O(L * d_state) 线性复杂度
        * 三模态各自独立的梯度回传 (autograd 自动支持)

方案 A: 三路 SISO 调用
    对每个目标模态 tgt ∈ {audio, visual, lexical}:
        Q ← q_proj[tgt](x_tgt)                          # 目标模态自己出 Q
        # 动态权重决定其它两个源模态的相对贡献:
        w_src = softmax(weight_net(concat features))    # (B, L, 2)
        K ← w_src[0] * k_proj[s0](x_s0) + w_src[1] * k_proj[s1](x_s1)
        V ← w_src[0] * v_proj[s0](x_s0) + w_src[1] * v_proj[s1](x_s1)
        out_tgt = mamba3_siso_combined(Q, K, V, ADT, DT, Trap, ..., Angles)
    -> 三次 kernel 调用 (相互独立可并发), 保留了 Coupled 的动态权重精髓。

输入 / 输出:
    输入  x_audio / x_visual / x_lexical: (B, L, d_model)
    输出  out_audio / out_visual / out_lexical: (B, L, d_model)

依赖:
    - torch
    - mamba_ssm (Optional, 没有时自动回退到纯 PyTorch 线性注意力近似)
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# ---------------------------------------------------------------------------
# 尝试导入 Mamba-3 的底层 SSD kernel (仅 forward_kernel 路径需要)
# ---------------------------------------------------------------------------
try:
    from mamba_ssm.ops.triton.mamba3.mamba3_siso_combined import mamba3_siso_combined
    MAMBA3_KERNEL_AVAILABLE = True
except ImportError:
    mamba3_siso_combined = None
    MAMBA3_KERNEL_AVAILABLE = False


# ============================================================================
# 工具函数: SSM 控制信号生成 (ADT / DT / Trap / Angles)
# ============================================================================
class _SSMControlGen(nn.Module):
    """
    根据目标模态特征生成 SSD kernel 所需的非 Q/K/V 控制信号。

    Mamba-3 kernel 需要:
        ADT   (B, H, L)      = -softplus(dd_A) * softplus(dd_dt + dt_bias)
        DT    (B, H, L)      = softplus(dd_dt + dt_bias)
        Trap  (B, H, L)      = sigmoid(trap_proj)
        Angles(B, L, H, A)   = 旋转角度

    这些信号 *只依赖 tgt 模态* (与 Mamba-3 同源), 因为它们决定的是
    "目标模态的状态如何衰减 / 离散化 / 旋转",而非来源模态特征。
    """
    def __init__(
        self,
        d_model: int,
        nheads: int,
        num_rope_angles: int,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init_floor: float = 1e-4,
        A_floor: float = 1e-4,
    ):
        super().__init__()
        self.nheads = nheads
        self.num_rope_angles = num_rope_angles
        self.A_floor = A_floor

        # 一次性投影出所有控制信号: dd_dt(H) + dd_A(H) + trap(H) + angles(num_rope_angles)
        d_ctrl = 3 * nheads + num_rope_angles
        self.ctrl_proj = nn.Linear(d_model, d_ctrl, bias=False)

        # dt_bias 用 Mamba-3 同款 inverse-softplus 初始化 (mamba3.py:91)
        _dt = torch.exp(
            torch.rand(nheads) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        _dt = torch.clamp(_dt, min=dt_init_floor)
        _dt_bias = _dt + torch.log(-torch.expm1(-_dt))
        self.dt_bias = nn.Parameter(_dt_bias, requires_grad=True)
        self.dt_bias._no_weight_decay = True

    def forward(self, x_tgt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x_tgt: (B, L, d_model)
        Returns:
            ADT:    (B, H, L)
            DT:     (B, H, L)
            Trap:   (B, H, L)        sigmoid 之后的梯形权重 (kernel 接受 pre/post 都行,这里 post)
            Angles: (B, L, H, A)
        """
        ctrl = self.ctrl_proj(x_tgt)  # (B, L, 3H + A)
        dd_dt, dd_A, trap_proj, angles = torch.split(
            ctrl,
            [self.nheads, self.nheads, self.nheads, self.num_rope_angles],
            dim=-1,
        )
        # ADT, DT
        A = -F.softplus(dd_A.float())
        A = torch.clamp(A, max=-self.A_floor)
        DT = F.softplus(dd_dt + self.dt_bias)
        ADT = A * DT
        DT = rearrange(DT, "b l h -> b h l")
        ADT = rearrange(ADT, "b l h -> b h l")
        # Trap (kernel 内部会做 sigmoid 兼容,这里保持与 mamba3.py 一致传 post-sigmoid)
        Trap = torch.sigmoid(trap_proj)
        Trap = rearrange(Trap, "b l h -> b h l")
        # Angles: 与 mamba3.py:171 一致, 在 head 维 expand
        angles = angles.unsqueeze(-2).expand(-1, -1, self.nheads, -1).to(torch.float32)
        return ADT, DT, Trap, angles


# ============================================================================
# CrossMamba3Block: 单个目标模态的跨模态 SSD 融合块
# ============================================================================
class CrossMamba3Block(nn.Module):
    """
    给定一个目标模态 tgt 和两个源模态 src_list = [s0, s1],
    用 Mamba-3 SSD kernel 计算 cross-modal SSM-attention:

        Q ← q_proj[tgt](x_tgt)
        w = softmax(weight_net(concat(x_tgt, x_s0, x_s1)))     # (B, L, 2)
        K ← w[..., 0:1] * k_proj[s0](x_s0) + w[..., 1:2] * k_proj[s1](x_s1)
        V ← w[..., 0:1] * v_proj[s0](x_s0) + w[..., 1:2] * v_proj[s1](x_s1)
        out_tgt = mamba3_siso_combined(Q, K, V, ADT, DT, Trap, ..., Angles)
    """
    def __init__(
        self,
        d_model: int,
        d_state: int = 128,
        nheads: int = 4,
        nheads_qk: int = 1,
        rope_fraction: float = 0.5,
        chunk_size: int = 64,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init_floor: float = 1e-4,
        A_floor: float = 1e-4,
        modality_keys: Tuple[str, str, str] = ("audio", "visual", "lexical"),
    ):
        """
        Args:
            d_model:         每模态特征维度
            d_state:         相当于 Mamba-3 的 headdim_qk (Q/K 维度), 必须为偶数
            nheads:          V 的 head 数 (= Mamba-3 nheads)
            nheads_qk:       Q/K 的 head 数 (GQA), 要求 nheads % nheads_qk == 0
            rope_fraction:   RoPE 比例 (0.5 或 1.0)
            chunk_size:      SSD 分块大小
        """
        super().__init__()
        assert d_state % 2 == 0, f"d_state ({d_state}) 必须为偶数 (RoPE 要求)"
        assert nheads % nheads_qk == 0, f"nheads ({nheads}) 必须能被 nheads_qk ({nheads_qk}) 整除"
        assert d_model % nheads == 0, f"d_model ({d_model}) 必须能被 nheads ({nheads}) 整除"
        assert rope_fraction in (0.5, 1.0)

        self.d_model = d_model
        self.d_state = d_state                # = headdim_qk
        self.nheads = nheads
        self.nheads_qk = nheads_qk
        self.headdim_v = d_model // nheads    # V 的 head 维
        self.chunk_size = chunk_size
        self.modality_keys = modality_keys
        self.num_modalities = len(modality_keys)

        # RoPE 角度数 (与 mamba3.py:78-83 对齐)
        rotary_dim_divisor = int(2 / rope_fraction)
        split_tensor_size = int(d_state * rope_fraction)
        if split_tensor_size % 2 != 0:
            split_tensor_size -= 1
        num_rope_angles = max(split_tensor_size // 2, 1)
        self.rotary_dim_divisor = rotary_dim_divisor
        self.num_rope_angles = num_rope_angles

        # ----- Q 投影 (按目标模态准备 1 套) -----
        # 输出形状: (B, L, nheads_qk * d_state)
        self.q_proj = nn.Linear(d_model, nheads_qk * d_state, bias=False)

        # ----- K / V 投影 (按源模态各准备 1 套, 共 2 套) -----
        self.k_projs = nn.ModuleDict({
            m: nn.Linear(d_model, nheads_qk * d_state, bias=False)
            for m in modality_keys
        })
        self.v_projs = nn.ModuleDict({
            m: nn.Linear(d_model, nheads * self.headdim_v, bias=False)
            for m in modality_keys
        })

        # ----- 动态源权重网络 (保留 Coupled 思想) -----
        # 输入: concat(x_tgt, x_s0, x_s1) -> (B, L, 3*d_model)
        # 输出: (B, L, 2) 的源混合权重 (每个时间步独立)
        self.weight_net = nn.Sequential(
            nn.Linear(self.num_modalities * d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, 2),
        )

        # ----- SSM 控制信号生成器 (依赖 tgt 模态) -----
        self.ctrl_gen = _SSMControlGen(
            d_model=d_model,
            nheads=nheads,
            num_rope_angles=num_rope_angles,
            dt_min=dt_min,
            dt_max=dt_max,
            dt_init_floor=dt_init_floor,
            A_floor=A_floor,
        )

        # ----- Q/K bias (与 mamba3.py:101-102 一致) -----
        self.Q_bias = nn.Parameter(1.0 + torch.zeros(nheads_qk, d_state))
        self.K_bias = nn.Parameter(1.0 + torch.zeros(nheads_qk, d_state))

        # ----- D 跳连 (与 mamba3.py 同款) -----
        self.D = nn.Parameter(torch.ones(nheads))
        self.D._no_weight_decay = True

        # ----- 输出投影: (B, L, nheads, headdim_v) -> (B, L, d_model) -----
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    # ------------------------------------------------------------------------
    def forward(
        self,
        x_tgt: torch.Tensor,
        x_src0: torch.Tensor,
        x_src1: torch.Tensor,
        src_keys: Tuple[str, str],
    ) -> torch.Tensor:
        """
        Args:
            x_tgt:  (B, L, d_model)    目标模态特征 (出 Q + 控制信号)
            x_src0: (B, L, d_model)    源模态 0  (出部分 K/V)
            x_src1: (B, L, d_model)    源模态 1  (出部分 K/V)
            src_keys: ("src0_name", "src1_name")  用于索引 k_projs / v_projs
        Returns:
            out_tgt: (B, L, d_model)
        """
        B, L, _ = x_tgt.shape
        s0_key, s1_key = src_keys

        # ---------- 1) Q 投影 (来自目标模态) ----------
        Q = self.q_proj(x_tgt)                                   # (B, L, Hqk * P)
        Q = rearrange(Q, "b l (h p) -> b l h p", h=self.nheads_qk)

        # ---------- 2) 动态源权重 ----------
        w_src_logits = self.weight_net(torch.cat([x_tgt, x_src0, x_src1], dim=-1))  # (B, L, 2)
        w_src = F.softmax(w_src_logits, dim=-1)                  # (B, L, 2)
        w0 = w_src[..., 0:1]                                     # (B, L, 1)
        w1 = w_src[..., 1:2]

        # ---------- 3) K 投影 (源模态加权融合) ----------
        K0 = self.k_projs[s0_key](x_src0)                        # (B, L, Hqk * P)
        K1 = self.k_projs[s1_key](x_src1)
        K = w0 * K0 + w1 * K1                                    # (B, L, Hqk * P)
        K = rearrange(K, "b l (h p) -> b l h p", h=self.nheads_qk)

        # ---------- 4) V 投影 (源模态加权融合) ----------
        V0 = self.v_projs[s0_key](x_src0)                        # (B, L, H * Pv)
        V1 = self.v_projs[s1_key](x_src1)
        V = w0 * V0 + w1 * V1
        V = rearrange(V, "b l (h p) -> b l h p", h=self.nheads)  # (B, L, H, Pv)

        # ---------- 5) SSM 控制信号 (依赖 tgt) ----------
        ADT, DT, Trap, Angles = self.ctrl_gen(x_tgt)             # 形状见 _SSMControlGen

        # ---------- 6) 调用 Mamba-3 SSD kernel 或 PyTorch fallback ----------
        if MAMBA3_KERNEL_AVAILABLE and Q.is_cuda:
            y = mamba3_siso_combined(
                Q=Q,
                K=K,
                V=V,
                ADT=ADT,
                DT=DT,
                Trap=Trap,
                Q_bias=self.Q_bias,
                K_bias=self.K_bias,
                Angles=Angles,
                D=self.D,
                Z=None,                # 这里不用门控, 简化首版
                Input_States=None,
                chunk_size=self.chunk_size,
                return_final_states=False,
                cu_seqlens=None,
            )                          # (B, L, H, Pv)
        else:
            y = self._pytorch_fallback(Q, K, V, ADT, DT, Trap, Angles)

        # ---------- 7) 输出投影 ----------
        y = rearrange(y, "b l h p -> b l (h p)")                 # (B, L, d_model)
        out = self.out_proj(y.to(x_tgt.dtype))
        return out

    # ------------------------------------------------------------------------
    def _pytorch_fallback(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        ADT: torch.Tensor,
        DT: torch.Tensor,
        Trap: torch.Tensor,
        Angles: torch.Tensor,
    ) -> torch.Tensor:
        """
        纯 PyTorch 线性注意力 + 时序衰减 fallback。

        近似公式 (单 head, 单 group, 忽略 RoPE 简化版, 仅用于无 kernel 时调试):
            S_t = exp(ADT_t) * S_{t-1} + (K_t + K_bias) ⊗ V_t * DT_t
            y_t = (Q_t + Q_bias) · S_t + D * V_t
        其中 S_t ∈ R^{H, Pv, Pqk} 是 outer-product 状态, 数学上是 SSD 的离散化。
        与 kernel 数值不一致 (无 RoPE/chunk 优化), 但语义对齐, 可用于 CPU 调试。

        注意:
            - 没有 RoPE (Angles 被忽略)
            - 没有 trap 梯形权重精确实现, 用线性插值近似
            - 仅供 CPU/无 kernel 环境下跑通 forward + backward, 不建议训练
        """
        B, L, H, Pv = V.shape
        Hqk, P = Q.shape[2], Q.shape[3]
        device, dtype = V.device, V.dtype

        # 应用 bias
        Q_b = Q + self.Q_bias                                    # (B, L, Hqk, P)
        K_b = K + self.K_bias

        # 处理 GQA: 把 Hqk 广播到 H
        if Hqk != H:
            assert H % Hqk == 0
            repeat = H // Hqk
            Q_b = Q_b.repeat_interleave(repeat, dim=2)           # (B, L, H, P)
            K_b = K_b.repeat_interleave(repeat, dim=2)

        # 状态 S: (B, H, Pv, P), outer product 累积
        S = torch.zeros(B, H, Pv, P, device=device, dtype=torch.float32)
        out_list = []
        ADT_t = ADT.transpose(1, 2)                              # (B, L, H)
        DT_t = DT.transpose(1, 2)                                # (B, L, H)
        Trap_t = Trap.transpose(1, 2)                            # (B, L, H)

        for t in range(L):
            decay = torch.exp(ADT_t[:, t, :].float())            # (B, H)
            dt_t = DT_t[:, t, :].float()                         # (B, H)
            trap_t = Trap_t[:, t, :].float()                     # (B, H)
            v_t = V[:, t, :, :].float()                          # (B, H, Pv)
            k_t = K_b[:, t, :, :].float()                        # (B, H, P)
            q_t = Q_b[:, t, :, :].float()                        # (B, H, P)

            # 梯形权重: B_bar 在零阶 (DT*K) 与 ZOH-like ((1-decay)*DT*K) 间凸组合
            kv = torch.einsum("bhp,bhq->bhpq", v_t, k_t)         # (B, H, Pv, P)
            ZOH = (1.0 - decay).unsqueeze(-1).unsqueeze(-1) * dt_t.unsqueeze(-1).unsqueeze(-1) * kv
            ZERO = dt_t.unsqueeze(-1).unsqueeze(-1) * kv
            kv_disc = trap_t.unsqueeze(-1).unsqueeze(-1) * ZOH \
                    + (1.0 - trap_t).unsqueeze(-1).unsqueeze(-1) * ZERO

            # 状态更新
            S = decay.unsqueeze(-1).unsqueeze(-1) * S + kv_disc  # (B, H, Pv, P)

            # 读取
            y_t = torch.einsum("bhpq,bhq->bhp", S, q_t)          # (B, H, Pv)
            y_t = y_t + self.D.view(1, H, 1).float() * v_t       # skip
            out_list.append(y_t)

        y = torch.stack(out_list, dim=1).to(dtype)               # (B, L, H, Pv)
        return y


# ============================================================================
# CrossMamba3: 三模态 cross-modal SSD 融合主模块
# ============================================================================
class CrossMamba3(nn.Module):
    """
    多模态 (audio / visual / lexical) 融合, 通过 *三次* 跨模态 Mamba-3 SSD 调用,
    每次让一个模态做 Q-source, 另外两个模态加权后做 K/V-source。

    与 CoupledMamba3 的差异:
        - CoupledMamba3 在 *状态层* 做加性耦合 (h_tgt += w * P(h_src))
        - CrossMamba3   在 *Q/K/V 层* 做交互, 直接利用 Mamba-3 kernel 的
          线性注意力性质实现深度跨模态融合 (信息可在 d_state 维上充分混合)
    """
    def __init__(
        self,
        d_model: int,
        d_state: int = 128,
        nheads: int = 4,
        nheads_qk: int = 1,
        rope_fraction: float = 0.5,
        chunk_size: int = 64,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init_floor: float = 1e-4,
        A_floor: float = 1e-4,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.modalities = ("audio", "visual", "lexical")

        # 每个 tgt 模态一个独立的 CrossMamba3Block
        self.cross_blocks = nn.ModuleDict({
            tgt: CrossMamba3Block(
                d_model=d_model,
                d_state=d_state,
                nheads=nheads,
                nheads_qk=nheads_qk,
                rope_fraction=rope_fraction,
                chunk_size=chunk_size,
                dt_min=dt_min,
                dt_max=dt_max,
                dt_init_floor=dt_init_floor,
                A_floor=A_floor,
                modality_keys=self.modalities,
            )
            for tgt in self.modalities
        })

        # 残差 + LN 与原 CoupledMamba3 保持一致
        self.layer_norms = nn.ModuleDict({
            m: nn.LayerNorm(d_model) for m in self.modalities
        })

    def _src_keys_for(self, tgt: str) -> Tuple[str, str]:
        """返回除 tgt 之外的另两个模态键, 顺序固定保证可重复性。"""
        return tuple(m for m in self.modalities if m != tgt)  # type: ignore

    def forward(
        self,
        x_audio: torch.Tensor,
        x_visual: torch.Tensor,
        x_lexical: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x_audio / x_visual / x_lexical: (B, L, d_model)
        Returns:
            out_audio / out_visual / out_lexical: (B, L, d_model)
        """
        feats: Dict[str, torch.Tensor] = {
            "audio": x_audio,
            "visual": x_visual,
            "lexical": x_lexical,
        }
        outs: Dict[str, torch.Tensor] = {}

        # 三次独立的 cross-modal SSD 调用 (相互无依赖, 可并发)
        for tgt in self.modalities:
            src_keys = self._src_keys_for(tgt)
            x_tgt = feats[tgt]
            x_s0 = feats[src_keys[0]]
            x_s1 = feats[src_keys[1]]
            y = self.cross_blocks[tgt](x_tgt, x_s0, x_s1, src_keys)
            outs[tgt] = self.layer_norms[tgt](y + x_tgt)         # 残差 + LN

        return outs["audio"], outs["visual"], outs["lexical"]


# ============================================================================
# Smoke Test
# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Cross-Mamba3 Module Test")
    print("=" * 60)
    print(f"MAMBA3_KERNEL_AVAILABLE = {MAMBA3_KERNEL_AVAILABLE}")

    B, L, D = 2, 32, 64
    model = CrossMamba3(
        d_model=D,
        d_state=64,           # = headdim_qk, 偶数即可
        nheads=4,             # V 的 head 数
        nheads_qk=1,          # Q/K 的 head 数 (GQA)
        rope_fraction=0.5,
        chunk_size=64,
    )

    x_a = torch.randn(B, L, D, requires_grad=True)
    x_v = torch.randn(B, L, D, requires_grad=True)
    x_l = torch.randn(B, L, D, requires_grad=True)

    out_a, out_v, out_l = model(x_a, x_v, x_l)
    print(f"Input  shape : {x_a.shape}")
    print(f"Output audio : {out_a.shape}")
    print(f"Output visual: {out_v.shape}")
    print(f"Output lexical: {out_l.shape}")

    # 反向测试: 三路梯度都应回传
    loss = out_a.sum() + out_v.sum() + out_l.sum()
    loss.backward()
    print(f"grad x_audio   : {x_a.grad.abs().mean().item():.4e}")
    print(f"grad x_visual  : {x_v.grad.abs().mean().item():.4e}")
    print(f"grad x_lexical : {x_l.grad.abs().mean().item():.4e}")

    if torch.isnan(out_a).any() or torch.isnan(out_v).any() or torch.isnan(out_l).any():
        print("Warning: NaNs detected!")
    else:
        print("Output numerical check passed.")
    print("=" * 60)