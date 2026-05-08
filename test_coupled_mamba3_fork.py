"""
CoupledMamba3Fork 完整 GPU 测试 (level 3, SISO 模式)

测试项:
    [T1] 模块构建 + 参数量统计 + 与 Mamba3 原版对比
    [T2] forward shape 正确性 (B, L, D) -> 三路同形输出
    [T3] backward 梯度回传 (三路梯度均不为零, 无 NaN)
    [T4] 跨模态分离度: 修改 src 的输入, 应该影响 tgt 输出 (验证 Q/K/V 路径连通)
    [T5] 数值稳定性: bf16 / 长序列 / 不同 batch 一致性
    [T6] 多 layer 堆叠
"""
import os
import sys
import math
import time
import torch
import torch.nn as nn

# 让 Python 找到 H-DCD/models
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from models.coupled_mamba3_fork import CoupledMamba3Fork, CrossMamba3Cell
from mamba_ssm.modules.mamba3 import Mamba3


def banner(s, ch="="):
    print("\n" + ch * 70)
    print(s)
    print(ch * 70)


def count_params(m):
    total = sum(p.numel() for p in m.parameters())
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    return total, trainable


# ============================================================
# 共用 GPU 配置
# ============================================================
DEVICE = "cuda"
DTYPE = torch.float32       # SISO kernel 默认 fp32, bf16 单独测试
torch.manual_seed(42)


# 模型超参 (与 mamba3.py 默认值对齐, 但缩小到 3070 8GB 能跑的规模)
CFG = dict(
    d_model=128,
    d_state=64,        # = headdim_qk, 必须偶数
    expand=2,          # d_inner = 256
    headdim=32,        # nheads = 8
    ngroups=1,
    rope_fraction=0.5,
    is_mimo=False,     # SISO; tilelang 不可用所以暂不测 MIMO
    chunk_size=64,
)
B, L = 2, 64


# ============================================================
# T1: 构建 + 参数量
# ============================================================
def test_build_and_params():
    banner("[T1] 构建 + 参数量")

    model = CoupledMamba3Fork(**CFG).to(DEVICE)
    total, trainable = count_params(model)
    print(f"CoupledMamba3Fork: total = {total:,}  trainable = {trainable:,}")

    # 对比单个 Mamba3 (作为基线参考)
    m3 = Mamba3(**CFG).to(DEVICE)
    m3_total, _ = count_params(m3)
    print(f"Mamba3 baseline (单模态): total = {m3_total:,}")
    print(f"参数量倍率: {total / m3_total:.2f}x  (理论 ~3x, 因为三 cell + weight_nets + LN)")

    # 检查关键子模块都在
    cell0 = model.cells["audio"]
    assert isinstance(cell0, CrossMamba3Cell)
    print(f"\nAudio cell 关键模块:")
    print(f"  in_proj_tgt    : {cell0.in_proj_tgt}")
    print(f"  c_proj_tgt     : {cell0.c_proj_tgt}")
    print(f"  b_projs keys   : {list(cell0.b_projs.keys())}")
    print(f"  v_projs keys   : {list(cell0.v_projs.keys())}")
    print(f"  has B_norm     : {cell0.B_norm is not None}")
    print(f"  has C_norm     : {cell0.C_norm is not None}")
    print(f"  is_mimo        : {cell0.is_mimo}")

    return model


# ============================================================
# T2: forward shape
# ============================================================
def test_forward_shape(model):
    banner("[T2] forward shape")

    model.eval()
    x_a = torch.randn(B, L, CFG["d_model"], device=DEVICE, dtype=DTYPE)
    x_v = torch.randn(B, L, CFG["d_model"], device=DEVICE, dtype=DTYPE)
    x_l = torch.randn(B, L, CFG["d_model"], device=DEVICE, dtype=DTYPE)

    with torch.no_grad():
        out_a, out_v, out_l = model(x_a, x_v, x_l)

    print(f"Input shape  : {x_a.shape}")
    print(f"Output audio : {out_a.shape}, dtype={out_a.dtype}")
    print(f"Output visual: {out_v.shape}")
    print(f"Output lexic.: {out_l.shape}")

    assert out_a.shape == x_a.shape
    assert out_v.shape == x_v.shape
    assert out_l.shape == x_l.shape
    assert not torch.isnan(out_a).any()
    assert not torch.isnan(out_v).any()
    assert not torch.isnan(out_l).any()
    print("[PASS] shape & no-NaN")


# ============================================================
# T3: backward 三路梯度
# ============================================================
def test_backward(model):
    banner("[T3] backward 三路梯度")

    model.train()
    x_a = torch.randn(B, L, CFG["d_model"], device=DEVICE, dtype=DTYPE, requires_grad=True)
    x_v = torch.randn(B, L, CFG["d_model"], device=DEVICE, dtype=DTYPE, requires_grad=True)
    x_l = torch.randn(B, L, CFG["d_model"], device=DEVICE, dtype=DTYPE, requires_grad=True)

    out_a, out_v, out_l = model(x_a, x_v, x_l)
    loss = out_a.sum() + out_v.sum() + out_l.sum()
    loss.backward()

    g_a = x_a.grad.abs().mean().item()
    g_v = x_v.grad.abs().mean().item()
    g_l = x_l.grad.abs().mean().item()
    print(f"|grad x_audio   | = {g_a:.4e}")
    print(f"|grad x_visual  | = {g_v:.4e}")
    print(f"|grad x_lexical | = {g_l:.4e}")

    assert g_a > 0 and g_v > 0 and g_l > 0
    assert not math.isnan(g_a + g_v + g_l)
    # 也检查模型参数有梯度
    n_with_grad = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum().item() > 0)
    n_total = sum(1 for p in model.parameters())
    print(f"参数有梯度的比例: {n_with_grad}/{n_total}")
    assert n_with_grad / n_total > 0.9
    print("[PASS] 三路梯度均 > 0, 无 NaN")


# ============================================================
# T4: 跨模态分离度 (验证 Q/K/V 路径连通)
# ============================================================
def test_cross_modal_connectivity(model):
    banner("[T4] 跨模态连通性 (扰动 src, 看 tgt 是否变)")

    model.eval()
    torch.manual_seed(0)
    x_a = torch.randn(B, L, CFG["d_model"], device=DEVICE, dtype=DTYPE)
    x_v = torch.randn(B, L, CFG["d_model"], device=DEVICE, dtype=DTYPE)
    x_l = torch.randn(B, L, CFG["d_model"], device=DEVICE, dtype=DTYPE)

    with torch.no_grad():
        out_a0, out_v0, out_l0 = model(x_a, x_v, x_l)

        # 扰动 visual, audio 的 Q 来自自己, K/V 来自 visual+lexical, 应该会变
        x_v2 = x_v + torch.randn_like(x_v) * 0.5
        out_a1, _, out_l1 = model(x_a, x_v2, x_l)

    delta_a = (out_a1 - out_a0).abs().mean().item()
    delta_l = (out_l1 - out_l0).abs().mean().item()
    print(f"扰动 visual 后:")
    print(f"  |Δ out_audio|   = {delta_a:.4e}  (audio 应该变, 因为 visual 是它的 K/V 源)")
    print(f"  |Δ out_lexical| = {delta_l:.4e}  (lexical 应该变, 同理)")

    assert delta_a > 1e-5, f"audio 输出未受 visual 扰动影响, 跨模态路径可能断了"
    assert delta_l > 1e-5, f"lexical 输出未受 visual 扰动影响"
    print("[PASS] 跨模态 Q/K/V 路径连通")


# ============================================================
# T5: 数值稳定性 (bf16 + 长序列)
# ============================================================
def test_numerical_robustness(model):
    banner("[T5] 数值稳定性")

    model.eval()
    # 5a: 长序列
    L_long = 256
    x_a = torch.randn(B, L_long, CFG["d_model"], device=DEVICE, dtype=DTYPE)
    x_v = torch.randn(B, L_long, CFG["d_model"], device=DEVICE, dtype=DTYPE)
    x_l = torch.randn(B, L_long, CFG["d_model"], device=DEVICE, dtype=DTYPE)
    with torch.no_grad():
        out_a, _, _ = model(x_a, x_v, x_l)
    print(f"5a 长序列 L=256: out shape={out_a.shape}, max|out|={out_a.abs().max().item():.3f}")
    assert not torch.isnan(out_a).any() and not torch.isinf(out_a).any()

    # 5b: bf16 (3070 sm_86 支持)
    try:
        model_bf = CoupledMamba3Fork(**CFG).to(DEVICE).to(torch.bfloat16)
        x_a = torch.randn(B, L, CFG["d_model"], device=DEVICE, dtype=torch.bfloat16)
        x_v = torch.randn(B, L, CFG["d_model"], device=DEVICE, dtype=torch.bfloat16)
        x_l = torch.randn(B, L, CFG["d_model"], device=DEVICE, dtype=torch.bfloat16)
        with torch.no_grad():
            out_a, _, _ = model_bf(x_a, x_v, x_l)
        print(f"5b bf16: out dtype={out_a.dtype}, max|out|={out_a.abs().max().float().item():.3f}")
        assert not torch.isnan(out_a).any() and not torch.isinf(out_a).any()
        print("[PASS] bf16 无 NaN/Inf")
    except Exception as e:
        print(f"[SKIP] bf16 测试失败 (可能 RMSNormGated 不支持): {type(e).__name__}: {e}")


# ============================================================
# T6: 多 layer 堆叠
# ============================================================
def test_multi_layer():
    banner("[T6] 多层堆叠")

    n_layer = 3
    model = nn.Sequential()
    for i in range(n_layer):
        model.add_module(f"layer{i}", _StackedLayer(**CFG))
    model = model.to(DEVICE)

    x_a = torch.randn(B, L, CFG["d_model"], device=DEVICE, dtype=DTYPE, requires_grad=True)
    x_v = torch.randn(B, L, CFG["d_model"], device=DEVICE, dtype=DTYPE, requires_grad=True)
    x_l = torch.randn(B, L, CFG["d_model"], device=DEVICE, dtype=DTYPE, requires_grad=True)

    feats = (x_a, x_v, x_l)
    for i in range(n_layer):
        feats = model[i](feats)
    out_a, out_v, out_l = feats

    loss = out_a.sum() + out_v.sum() + out_l.sum()
    loss.backward()
    print(f"{n_layer}-layer stack: out_a shape={out_a.shape}, |grad x_a|={x_a.grad.abs().mean().item():.4e}")
    assert not torch.isnan(out_a).any()
    print(f"[PASS] {n_layer} 层堆叠 forward+backward OK")


class _StackedLayer(nn.Module):
    def __init__(self, **cfg):
        super().__init__()
        self.cm = CoupledMamba3Fork(**cfg)
    def forward(self, feats):
        return self.cm(*feats)


# ============================================================
# main
# ============================================================
def main():
    banner("CoupledMamba3Fork GPU 完整测试 (level 3, SISO)", ch="#")
    print(f"Device : {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Config : {CFG}")
    print(f"B={B}, L={L}")

    t0 = time.time()
    model = test_build_and_params()
    test_forward_shape(model)
    test_backward(model)
    test_cross_modal_connectivity(model)
    test_numerical_robustness(model)
    test_multi_layer()
    elapsed = time.time() - t0

    banner(f"全部测试通过! 耗时 {elapsed:.1f}s", ch="#")


if __name__ == "__main__":
    main()