"""
Coupled Mamba3: Multi-modal Fusion with Mamba-3 Backbone + Adaptive State Coupling
==================================================================================

本模块在原 Coupled Mamba 的基础上,将底层 SSM 内核从 Mamba / Mamba2 升级为
Mamba-3 (Dao AI Lab & Goombalab, 2026),同时完整保留跨模态自适应耦合机制。

相对原版的核心升级 (方案 C: d_state=128 + loop 模式同步升级):
    1. parallel 路径直接调用官方 `mamba_ssm.Mamba3` (融合 RoPE + SSD + 改进离散化,
       去除 conv1d)。
    2. loop 路径将简化版 SSM Cell 重写为 Mamba-3 风格的单步单元 `CoupledMambaCell3`:
       data-dependent A (-softplus(dd_A) clamp)、Δ (softplus(dd_dt + dt_bias))、
       梯形/二阶离散化 (trap = sigmoid(trap_proj)),与论文公式语义一致。
    3. 跨模态耦合机制 (自适应权重网络 + 跨模态状态投影 + 残差/LN) 完全保留,
       签名 (x_t, h_prev, coupled_influence) -> (y_t, h_t) 不变,
       因此原 forward_loop 的耦合扫描逻辑 0 改动。

输入 / 输出张量 shape 与原版完全一致:
    输入:  x_audio / x_visual / x_lexical, 各为 [Batch, Seq_Len, d_model]
    输出:  out_audio / out_visual / out_lexical, 同上

Classes:
    - CoupledMambaCell3: Mamba-3 风格单步 SSM 单元 (loop 路径用,支持外部状态耦合)
    - CoupledMamba3:     主模块,处理三模态序列并实现自适应跨模态耦合

Dependencies:
    - torch
    - mamba_ssm (Optional, 仅 parallel 模式需要,导入失败会自动回退到 loop)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

# [Mamba3 改造] 导入符号由 Mamba2 替换为 Mamba3;变量名同步改为 MAMBA3_AVAILABLE
# 以避免与旧代码中的 MAMBA_AVAILABLE 混淆。Mamba3 在 mamba_ssm/__init__.py 已暴露。
try:
    from mamba_ssm import Mamba3
    MAMBA3_AVAILABLE = True
except ImportError:
    MAMBA3_AVAILABLE = False


# ============================================================================
# CoupledMambaCell3: Mamba-3 风格单步 SSM 单元 (loop 路径专用)
# ============================================================================
class CoupledMambaCell3(nn.Module):
    """
    Mamba-3 风格的单时间步 SSM 单元,数学语义对齐论文公式:

        # 来自 token x_t 的数据依赖投影 (单一 in_proj 一次性切片):
        B_t, C_t, dd_dt, dd_A, trap_proj = split(in_proj(x_t))

        # 1) 数据依赖 A 与 Δ (Mamba-3 关键):
        A    = -softplus(dd_A)            # ≤ 0, 再 clamp(max=-A_floor)
        DT   = softplus(dd_dt + dt_bias)  # > 0
        ADT  = A * DT

        # 2) 梯形 / 二阶离散化 (trap 是 Mamba-3 标志):
        trap  = sigmoid(trap_proj)
        A_bar = exp(ADT)                                              # ∈ (0, 1]
        B_bar = (1 - A_bar) * (B_t * DT) * trap + (B_t * DT) * (1-trap)

        # 3) 状态更新 (耦合注入位置与原版完全一致):
        h_t = A_bar * h_{t-1} + B_bar * x_drive + coupled_influence

        # 4) 输出:
        y_t = out_proj( C_t * h_t ) + D * x_t

    其中 `x_drive` 取 x_t 在 d_model 上的均值标量, 复刻 Mamba-3 SSD 中
    "x 拆 head 后每 head 一个驱动力" 的简化形式 (loop 教学版,head=1)。

    与原 CoupledMambaCell 的差异:
        - A 由 Linear(d_state, d_state) 静态稠密 -> 数据依赖对角 (A_bar)
        - 新增 Δ、trap 数据依赖投影
        - 移除 SiLU(state_update),非线性由 in_proj/out_proj 与梯形权重承担
        - 接口 (x_t, h_prev, coupled_influence) -> (y_t, h_t) 完全保持,
          因此 forward_loop 调用方 0 改动。
    """

    def __init__(
        self,
        d_model: int,
        d_state: int,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init_floor: float = 1e-4,
        A_floor: float = 1e-4,
    ):
        """
        Args:
            d_model:        输入/输出特征维度。
            d_state:        SSM 隐状态维度 (Mamba-3 推荐 ≥ 64,默认外层 128)。
            dt_min/dt_max:  Δ 初始化区间 (与 Mamba3 官方一致, mamba3.py:91)。
            dt_init_floor:  Δ 下限,防止 softplus 后过小导致梯度消失。
            A_floor:        A 上限 (取负后是下限),防止 A_bar 过于接近 1。
        """
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.A_floor = A_floor

        # [Mamba3 改造] 单一 in_proj 一次性输出所有数据依赖参数:
        #   B_t (d_state) | C_t (d_state) | dd_dt (1) | dd_A (1) | trap_proj (1)
        # 这与 Mamba3.in_proj 的 split 风格完全对齐 (mamba3.py:84)。
        d_in_proj = 2 * d_state + 3
        self.in_proj = nn.Linear(d_model, d_in_proj, bias=False)

        # [Mamba3 改造] dt_bias 用 Mamba3 同款 inverse-softplus 初始化,
        # 保证 softplus(dd_dt + dt_bias) 初值落在 [dt_min, dt_max] 区间。
        # 公式来源: mamba3.py:91 (dt + log(-expm1(-dt))).
        _dt = torch.exp(
            torch.rand(1) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        )
        _dt = torch.clamp(_dt, min=dt_init_floor)
        _dt_bias = _dt + torch.log(-torch.expm1(-_dt))
        self.dt_bias = nn.Parameter(_dt_bias, requires_grad=True)
        self.dt_bias._no_weight_decay = True

        # [Mamba3 改造] D 是 skip 残差 (Mamba3 沿用), 形状为标量 (head=1 简化)。
        self.D = nn.Parameter(torch.ones(1))
        self.D._no_weight_decay = True

        # 输出投影: (C_t ⊙ h_t) ∈ [B, d_state] -> [B, d_model]
        # 这里把 Mamba3 中 "C 与 h 的 einsum" + out_proj 合并为一个 Linear,
        # 数学等价 (单 head 情形下 einsum 退化为逐元素乘后求和或线性变换)。
        self.out_proj = nn.Linear(d_state, d_model)

    def forward(
        self,
        x_t: torch.Tensor,
        h_prev: torch.Tensor,
        coupled_influence: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x_t:               [Batch, d_model]
            h_prev:            [Batch, d_state]
            coupled_influence: [Batch, d_state] 或 None  (跨模态加权状态注入)

        Returns:
            y_t: [Batch, d_model]
            h_t: [Batch, d_state]
        """
        # ---------- 1. 数据依赖投影 ----------
        proj = self.in_proj(x_t)  # [B, 2*d_state + 3]
        B_t, C_t, dd_dt, dd_A, trap_proj = torch.split(
            proj,
            [self.d_state, self.d_state, 1, 1, 1],
            dim=-1,
        )  # 形状: B_t/C_t [B, d_state]; 其余 [B, 1]

        # ---------- 2. 计算 A、Δ、trap (Mamba-3 核心) ----------
        # [Mamba3 改造] A 不再是独立 nn.Linear, 而是从 token 投影出 dd_A 后取 -softplus
        A = -F.softplus(dd_A.float())                       # [B, 1], ≤ 0
        A = torch.clamp(A, max=-self.A_floor)               # 防止 A 趋近 0
        DT = F.softplus(dd_dt + self.dt_bias)               # [B, 1], > 0
        trap = torch.sigmoid(trap_proj)                     # [B, 1], ∈ (0, 1)

        # ---------- 3. 梯形 / 二阶离散化 ----------
        # [Mamba3 改造] A_bar = exp(A * DT), 数值上始终 ∈ (0, 1], 无溢出风险
        A_bar = torch.exp(A * DT)                           # [B, 1]
        B_DT = B_t * DT                                     # [B, d_state]
        # 梯形权重凸组合: trap=1 退化为 ZOH-like, trap=0 退化为纯 Δ·B 注入
        B_bar = (1.0 - A_bar) * B_DT * trap + B_DT * (1.0 - trap)  # [B, d_state]

        # ---------- 4. 状态更新 (耦合注入点保持原版位置) ----------
        # x_drive: 复刻 Mamba-3 SSD 中 "每 head 一个驱动力标量" 的简化形式
        x_drive = x_t.mean(dim=-1, keepdim=True)            # [B, 1]
        h_t = A_bar * h_prev + B_bar * x_drive              # [B, d_state]

        # ⭐ 跨模态耦合注入 (与原 CoupledMambaCell.forward 第 81-83 行语义一致)
        if coupled_influence is not None:
            h_t = h_t + coupled_influence

        # ---------- 5. 输出 ----------
        # y_t = out_proj(C_t ⊙ h_t) + D * x_t (skip 残差)
        y_state = self.out_proj(C_t * h_t.to(C_t.dtype))    # [B, d_model]
        y_t = y_state + self.D * x_t                        # broadcast D over d_model
        return y_t, h_t


# ============================================================================
# CoupledMamba3: 多模态融合主模块
# ============================================================================
class CoupledMamba3(nn.Module):
    """
    Coupled Mamba3 -- 多模态 (audio / visual / lexical) 序列融合模块。

    特性:
        1. 三个独立的 Mamba-3 风格 SSM 通道 (loop 模式) 或 一个宽 Mamba-3 通道 (parallel 模式)
        2. 自适应权重网络: 在每个时间步动态生成 3x3 跨模态注意力权重
        3. 跨模态状态投影: 把源模态隐状态映射到目标模态状态空间
        4. 残差连接 + LayerNorm

    两种前向实现:
        - forward_loop:     显式按时间步循环, 严格实现耦合公式 (慢, 教学/小规模, 0 内核依赖)
        - forward_parallel: 调用官方 Mamba3, 享受融合的 RoPE + SSD kernel (快, 需安装 mamba_ssm)
    """

    def __init__(
        self,
        d_model: int,
        # [Mamba3 改造] d_state 默认从 64 提升到 128, 与 Mamba3 官方默认一致 (mamba3.py:30)
        d_state: int = 128,
        use_parallel: bool = False,
        # ↓↓↓ [Mamba3 改造] 透出 Mamba-3 的可选超参 (仅作用于 parallel 路径) ↓↓↓
        headdim: int = 64,
        ngroups: int = 1,
        rope_fraction: float = 0.5,
        is_mimo: bool = False,
        mimo_rank: int = 4,
        chunk_size: int = 64,
        A_floor: float = 1e-4,
        is_outproj_norm: bool = False,
        # ↓↓↓ loop 路径 Cell 的数值稳定性超参 ↓↓↓
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init_floor: float = 1e-4,
    ):
        """
        Args:
            d_model:         每个模态的特征维度。
            d_state:         SSM 状态维度 (推荐 ≥ 64;Mamba-3 默认 128)。
            use_parallel:    True 走 Mamba3 内核 (需 mamba_ssm + GPU + Triton);False 走显式循环。
            headdim:         Mamba-3 内部 head 维度。要求 d_inner_wide=2*3*d_model 能被 headdim 整除。
            ngroups:         B/C 的 group 数 (Mamba-3 默认 1)。
            rope_fraction:   RoPE 应用比例,只能取 0.5 或 1.0 (mamba3.py:78)。
            is_mimo:         是否启用 MIMO 架构 (需要 TileLang)。
            mimo_rank:       MIMO 秩 (仅 is_mimo=True 时生效)。
            chunk_size:      Mamba-3 SSD 分块大小,SISO 推荐 64;MIMO 推荐 64/mimo_rank。
            A_floor:         A 上限 (取负后是下限), loop 与 parallel 共享。
            is_outproj_norm: 是否在输出前对 y 做 RMSNormGated。
            dt_min/dt_max/dt_init_floor: loop 路径 Cell 的 Δ 初始化超参。
        """
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.use_parallel = use_parallel
        self.modalities = ['audio', 'visual', 'lexical']
        self.num_modalities = len(self.modalities)

        # ====================================================================
        # Loop 实现的组件 (显式跨模态耦合) -- 与原版结构 1:1 对齐
        # ====================================================================
        # 1) 三个独立的 Mamba-3 风格 Cell
        # [Mamba3 改造] cell 类从 CoupledMambaCell 升级为 CoupledMambaCell3
        self.mamba_cores = nn.ModuleDict({
            m: CoupledMambaCell3(
                d_model=d_model,
                d_state=d_state,
                dt_min=dt_min,
                dt_max=dt_max,
                dt_init_floor=dt_init_floor,
                A_floor=A_floor,
            )
            for m in self.modalities
        })

        # 2) 跨模态状态投影 (Coupled 机制核心 -- 与底层 SSM 解耦,原样保留)
        self.coupling_projections = nn.ModuleDict()
        for tgt in self.modalities:
            for src in self.modalities:
                if src == tgt:
                    continue
                layer_name = f"{src}_to_{tgt}"
                self.coupling_projections[layer_name] = nn.Linear(
                    d_state, d_state, bias=False
                )

        # 3) 自适应权重网络 (Coupled 机制核心 -- 原样保留)
        # 输入: 三个模态拼接的隐状态 [B, 3*d_state]
        # 输出: [B, 3, 3] 的跨模态注意力 logits (再经 softmax)
        self.weight_net = nn.Sequential(
            nn.Linear(self.num_modalities * d_state, d_state),
            nn.Tanh(),
            nn.Linear(d_state, self.num_modalities * self.num_modalities),
        )

        # 4) 每模态独立 LayerNorm (与底层 SSM 无关,原样保留)
        self.layer_norms = nn.ModuleDict({
            m: nn.LayerNorm(d_model) for m in self.modalities
        })

        # ====================================================================
        # Parallel 实现的组件 (Mamba-3 内核融合)
        # ====================================================================
        if MAMBA3_AVAILABLE:
            # [Mamba3 改造] 维度断言: Mamba-3 内部要求 d_inner = expand * d_model
            # 必须能被 headdim 整除 (mamba3.py:73)。这里 d_inner_wide = 2 * (3 * d_model)。
            d_inner_wide = 2 * d_model * self.num_modalities
            assert d_inner_wide % headdim == 0, (
                f"[Mamba3 改造] d_inner = 2 * (3 * d_model) = {d_inner_wide} "
                f"必须能被 headdim={headdim} 整除。请调整 d_model 或 headdim。"
            )

            # [Mamba3 改造] Mamba2 -> Mamba3:
            #   - 移除 d_conv=4 (Mamba-3 完全去除 conv1d 路径)
            #   - 新增 headdim/ngroups/rope_fraction/is_mimo/mimo_rank/chunk_size 等
            #   - 通道混合改由 in_proj + 数据依赖 A 的 SSD 扫描 + RoPE 完成
            self.parallel_mamba = Mamba3(
                d_model=d_model * self.num_modalities,  # 把 3 个模态拼接成宽序列
                d_state=d_state,
                expand=2,
                headdim=headdim,
                ngroups=ngroups,
                rope_fraction=rope_fraction,
                is_mimo=is_mimo,
                mimo_rank=mimo_rank,
                chunk_size=chunk_size,
                A_floor=A_floor,
                is_outproj_norm=is_outproj_norm,
            )

            # 拆分回每个模态: Mamba3.out_proj 输出 [B, L, d_model_wide]
            # 这里 Linear(3*d_model -> d_model) 形状与 Mamba3 输出严格匹配。
            self.parallel_proj = nn.ModuleDict({
                m: nn.Linear(d_model * self.num_modalities, d_model)
                for m in self.modalities
            })

    # ------------------------------------------------------------------------
    # Forward 调度
    # ------------------------------------------------------------------------
    def forward(
        self,
        x_audio: torch.Tensor,
        x_visual: torch.Tensor,
        x_lexical: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x_audio:   [Batch, Seq_Len, d_model]
            x_visual:  [Batch, Seq_Len, d_model]
            x_lexical: [Batch, Seq_Len, d_model]
        Returns:
            out_audio, out_visual, out_lexical: 形状与输入相同
        """
        if self.use_parallel and MAMBA3_AVAILABLE:
            return self.forward_parallel(x_audio, x_visual, x_lexical)
        else:
            if self.use_parallel and not MAMBA3_AVAILABLE:
                # [Mamba3 改造] 回退提示文案: Mamba2 -> Mamba3
                print("Warning: Mamba3 not available. Falling back to loop implementation.")
            return self.forward_loop(x_audio, x_visual, x_lexical)

    # ------------------------------------------------------------------------
    # Loop 实现: 显式跨模态状态耦合 (与原版逻辑 0 改动,仅底层 cell 已升级)
    # ------------------------------------------------------------------------
    def forward_loop(self, x_audio, x_visual, x_lexical):
        """
        显式按时间步实现自适应状态耦合,严格执行:
            h_t^{(i)} = SSM_i(x_t^{(i)}, h_{t-1}^{(i)}) + Σ_{j≠i} w_{ij}·P_{j→i}(h_{t-1}^{(j)})
        其中 w_{ij} 由 weight_net(concat(h_{t-1})) 在每步动态生成 (3x3 softmax)。

        注意: 内部 SSM_i 已升级为 CoupledMambaCell3 (Mamba-3 风格),
              但耦合扫描代码与原版完全一致。
        """
        batch_size, seq_len, _ = x_audio.shape
        device = x_audio.device

        # 初始化每模态隐状态
        h_states = {
            m: torch.zeros(batch_size, self.d_state, device=device)
            for m in self.modalities
        }
        outputs = {m: [] for m in self.modalities}

        for t in range(seq_len):
            # 快照上一步状态 (避免并行更新带来的状态污染)
            h_prev = {k: v.clone() for k, v in h_states.items()}

            # 1) 自适应权重生成: [B, 3*d_state] -> [B, 3, 3]
            h_concat = torch.cat(
                [h_prev['audio'], h_prev['visual'], h_prev['lexical']], dim=-1
            )
            raw_weights = self.weight_net(h_concat).view(
                batch_size, self.num_modalities, self.num_modalities
            )
            attn_weights = F.softmax(raw_weights, dim=-1)
            # attn_weights[b, i, j] = 在更新模态 i 时, 源模态 j 的重要性

            # 2) 逐模态更新
            for tgt_idx, tgt_modality in enumerate(self.modalities):
                # 取当前 token 输入
                if tgt_modality == 'audio':
                    x_t = x_audio[:, t, :]
                elif tgt_modality == 'visual':
                    x_t = x_visual[:, t, :]
                else:
                    x_t = x_lexical[:, t, :]

                # 累加跨模态耦合上下文
                coupling_context = 0.0
                for src_idx, src_modality in enumerate(self.modalities):
                    if src_modality == tgt_modality:
                        continue  # 自环已由 cell 内部 A_bar*h_prev 处理
                    w_ij = attn_weights[:, tgt_idx, src_idx].unsqueeze(-1)  # [B, 1]
                    proj_layer = self.coupling_projections[
                        f"{src_modality}_to_{tgt_modality}"
                    ]
                    h_src_projected = proj_layer(h_prev[src_modality])
                    coupling_context = coupling_context + (w_ij * h_src_projected)

                # 调用 Mamba-3 风格 cell, 注入耦合项
                y_t, h_new = self.mamba_cores[tgt_modality](
                    x_t,
                    h_prev[tgt_modality],
                    coupled_influence=coupling_context,
                )
                h_states[tgt_modality] = h_new
                outputs[tgt_modality].append(y_t)

        # 时间维堆叠 + 残差 + LN
        out_audio = torch.stack(outputs['audio'], dim=1)
        out_visual = torch.stack(outputs['visual'], dim=1)
        out_lexical = torch.stack(outputs['lexical'], dim=1)

        out_audio = self.layer_norms['audio'](out_audio + x_audio)
        out_visual = self.layer_norms['visual'](out_visual + x_visual)
        out_lexical = self.layer_norms['lexical'](out_lexical + x_lexical)

        return out_audio, out_visual, out_lexical

    # ------------------------------------------------------------------------
    # Parallel 实现: 通过宽 Mamba-3 内核做隐式跨模态混合
    # ------------------------------------------------------------------------
    def forward_parallel(self, x_audio, x_visual, x_lexical):
        """
        高性能近似实现: 沿特征维拼接三个模态为宽序列, 喂入一个 Mamba-3 块。
        Mamba-3 内部通过 in_proj + 数据依赖 A 的 SSD 扫描 + RoPE 完成跨模态混合
        (相比 Mamba2: 没有 conv1d, 改进的离散化等价模拟卷积)。
        """
        # [Mamba3 改造] 沿特征维拼接: [B, L, 3*d_model]
        x_concat = torch.cat([x_audio, x_visual, x_lexical], dim=-1)

        # [Mamba3 改造] 调用 Mamba-3。注意 Mamba3.forward 签名不再接受 seqlen,
        # 仅 (u, seq_idx=None, cu_seqlens=None, inference_params=None)。
        out_concat = self.parallel_mamba(x_concat)  # [B, L, 3*d_model]

        # 拆分回三个模态 (使用各自独立的 Linear, 增强解耦能力)
        out_audio = self.parallel_proj['audio'](out_concat)
        out_visual = self.parallel_proj['visual'](out_concat)
        out_lexical = self.parallel_proj['lexical'](out_concat)

        # 残差 + LN
        out_audio = self.layer_norms['audio'](out_audio + x_audio)
        out_visual = self.layer_norms['visual'](out_visual + x_visual)
        out_lexical = self.layer_norms['lexical'](out_lexical + x_lexical)

        return out_audio, out_visual, out_lexical


# ============================================================================
# Usage Example
# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Coupled Mamba3 Module Test")  # [Mamba3 改造] 文案更新
    print("=" * 60)

    # [Mamba3 改造] D_STATE 默认从 32 提升到 128, 与 Mamba3 官方对齐;
    # D_MODEL=64 时 d_inner_wide = 2*3*64 = 384, 384 % headdim(64) == 0 ✅
    BATCH_SIZE = 4
    SEQ_LEN = 32
    D_MODEL = 64
    D_STATE = 128

    # 实例化: use_parallel=True 需安装 mamba_ssm 且有 GPU+Triton
    model = CoupledMamba3(
        d_model=D_MODEL,
        d_state=D_STATE,
        use_parallel=False,
        headdim=64,
    )

    x_a = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL)
    x_v = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL)
    x_l = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL)

    print(f"Input Shapes: {x_a.shape}")
    print(
        f"Mode: {'Parallel (Mamba3)' if model.use_parallel else 'Loop (Explicit Coupling, Mamba3-style Cell)'}"
    )

    out_a, out_v, out_l = model(x_a, x_v, x_l)

    print("\nForward Pass Successful!")
    print(f"Output Audio:   {out_a.shape}")
    print(f"Output Visual:  {out_v.shape}")
    print(f"Output Lexical: {out_l.shape}")

    if torch.isnan(out_a).any() or torch.isnan(out_v).any() or torch.isnan(out_l).any():
        print("Warning: NaNs detected in output!")
    else:
        print("Output numerical check passed.")
    print("=" * 60)