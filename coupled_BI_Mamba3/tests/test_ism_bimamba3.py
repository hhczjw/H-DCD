"""
ISMEncoder × BiMamba3 接入测试
==============================
验证 ISMEncoder(mixer_type="bimamba3") 的 shape 与 backward 正确性,
并对比 mixer_type="bimamba" (Mamba-2 v2) 基线确保向后兼容。

运行:
    cd H-DCD/coupled_BI_Mamba3
    python tests/test_ism_bimamba3.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import _path_setup  # noqa: F401 — 注入 mamba/ 到 sys.path

import torch
from layers.ism import ISMEncoder, BIMAMBA3_AVAILABLE


def _banner(title: str):
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def _count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# T1: mixer_type="bimamba" 基线 (Mamba-2 v2)
# ---------------------------------------------------------------------------
def test_bimamba_baseline(device, dtype):
    _banner("T1  ISMEncoder(mixer_type='bimamba')  基线 — Mamba-2 v2")

    enc = ISMEncoder(
        d_model=256, seq_len=50, depth=2,
        d_state=16, d_conv=4, expand=2,
        mixer_type="bimamba",
    ).to(device=device, dtype=dtype)

    print(f"参数量: {_count_params(enc) / 1e6:.2f} M")

    x = torch.randn(2, 50, 256, device=device, dtype=dtype, requires_grad=True)
    y = enc(x)
    assert y.shape == (2, 50, 256), f"shape mismatch: {y.shape}"
    print(f"✔  forward OK: {tuple(y.shape)}")

    loss = y.sum()
    loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    print(f"✔  backward OK: loss={loss.item():.4f}, x.grad finite")


# ---------------------------------------------------------------------------
# T2: mixer_type="bimamba3" (Mamba-3 v2)
# ---------------------------------------------------------------------------
def test_bimamba3(device, dtype):
    _banner("T2  ISMEncoder(mixer_type='bimamba3')  — Mamba-3 v2")

    if not BIMAMBA3_AVAILABLE:
        print("⚠  BiMamba3 不可用, 跳过")
        return

    # Mamba-3 的 d_state 需要 >= headdim 且最好是 128; 这里 d_state=64, headdim=32
    enc = ISMEncoder(
        d_model=256, seq_len=50, depth=2,
        d_state=64,        # Mamba-3 的 state 维度
        d_conv=4,          # 忽略 (Mamba-3 无 conv1d)
        expand=2,
        mixer_type="bimamba3",
        bimamba3_headdim=32,
        bimamba3_ngroups=1,
        bimamba3_rope_fraction=0.5,
        bimamba3_chunk_size=64,
        bimamba3_is_mimo=False,
        bimamba3_fusion="add_divide2",
    ).to(device=device, dtype=dtype)

    print(f"参数量: {_count_params(enc) / 1e6:.2f} M")

    x = torch.randn(2, 50, 256, device=device, dtype=dtype, requires_grad=True)
    y = enc(x)
    assert y.shape == (2, 50, 256), f"shape mismatch: {y.shape}"
    print(f"✔  forward OK: {tuple(y.shape)}")

    loss = y.float().sum()
    loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    print(f"✔  backward OK: loss={loss.item():.4f}, x.grad finite")


# ---------------------------------------------------------------------------
# T3: 四种融合模式
# ---------------------------------------------------------------------------
def test_bimamba3_fusions(device, dtype):
    _banner("T3  ISMEncoder(bimamba3)  四种融合模式 forward+backward")

    if not BIMAMBA3_AVAILABLE:
        print("⚠  BiMamba3 不可用, 跳过")
        return

    for fusion in ["add", "add_divide2", "concat_proj", "gated"]:
        enc = ISMEncoder(
            d_model=256, seq_len=50, depth=1,
            d_state=64, expand=2,
            mixer_type="bimamba3",
            bimamba3_headdim=32,
            bimamba3_fusion=fusion,
        ).to(device=device, dtype=dtype)

        x = torch.randn(2, 50, 256, device=device, dtype=dtype, requires_grad=True)
        y = enc(x)
        loss = y.float().sum()
        loss.backward()
        assert y.shape == (2, 50, 256)
        assert torch.isfinite(x.grad).all()
        print(f"✔  fusion={fusion!r:>16s}  loss={loss.item():9.4f}   params={_count_params(enc)/1e6:.2f}M")


# ---------------------------------------------------------------------------
# T4: MIMO 模式
# ---------------------------------------------------------------------------
def test_bimamba3_mimo(device, dtype):
    _banner("T4  ISMEncoder(bimamba3, is_mimo=True)  MIMO rank-4")

    if not BIMAMBA3_AVAILABLE:
        print("⚠  BiMamba3 不可用, 跳过")
        return

    try:
        enc = ISMEncoder(
            d_model=256, seq_len=50, depth=1,
            d_state=64, expand=2,
            mixer_type="bimamba3",
            bimamba3_headdim=32,
            bimamba3_chunk_size=16,         # MIMO 要求 chunk_size * mimo_rank <= 64
            bimamba3_is_mimo=True,
            bimamba3_mimo_rank=4,
            bimamba3_fusion="add_divide2",
            bimamba3_share_mimo=True,
        ).to(device=device, dtype=dtype)
    except AssertionError as e:
        print(f"⚠  MIMO kernel 不可用, 跳过 ({e})")
        return

    x = torch.randn(2, 50, 256, device=device, dtype=dtype, requires_grad=True)
    y = enc(x)
    loss = y.float().sum()
    loss.backward()
    assert y.shape == (2, 50, 256)
    print(f"✔  MIMO forward+backward OK: loss={loss.item():.4f}, params={_count_params(enc)/1e6:.2f}M")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("✘  需要 CUDA 设备"); sys.exit(1)
    device = torch.device("cuda")
    dtype = torch.float16   # ISMEncoder 的 fused_add_norm RMSNorm 要求 fp16/bf16

    print(f"✔  CUDA 设备: {torch.cuda.get_device_name()}")
    print(f"✔  BIMAMBA3_AVAILABLE = {BIMAMBA3_AVAILABLE}")

    test_bimamba_baseline(device, dtype)
    test_bimamba3(device, dtype)
    test_bimamba3_fusions(device, dtype)
    test_bimamba3_mimo(device, dtype)

    _banner("汇总")
    print("✔  全部测试通过")