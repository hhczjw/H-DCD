"""
VIM 冒烟测试 — 验证内置 mamba_ssm 的 VIM 改造是否全链路可用
============================================================

覆盖三个层级:
    T1. 底层算子  : Mamba(d_model, bimamba_type="v2") 前向 + 反向
    T2. 顶层模型  : vim_tiny(num_classes=10) 前向 (224x224 RGB)
    T3. 下游集成  : ISMEncoder (原工程的 ISM 块)

运行:
    cd H-DCD/coupled_BI_Mamba3
    python -m tests.test_vim_smoke
或:
    python tests/test_vim_smoke.py

注意: Mamba 的 CUDA kernel 依赖 GPU, 本脚本自动检测并 skip CPU 场景。
"""
from __future__ import annotations

import os
import sys
import traceback

import torch

# 确保能 import 到内置的 mamba_ssm 和 layers
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "mamba"))   # 内置 mamba_ssm
sys.path.insert(0, ROOT)                           # layers.ism


def _print_banner(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def _check_cuda():
    if not torch.cuda.is_available():
        print("⚠  未检测到 CUDA, Mamba selective_scan 需要 GPU。跳过所有测试。")
        return False
    print(f"✔  CUDA 设备: {torch.cuda.get_device_name(0)}")
    return True


# ---------------------------------------------------------------------------
# T1: 原生 Mamba(bimamba_type="v2") 前向 + 反向
# ---------------------------------------------------------------------------
def test_mamba_bimamba_v2():
    _print_banner("T1  Mamba(bimamba_type='v2') 前向 + 反向")
    from mamba_ssm.modules.mamba_simple import Mamba

    B, L, D = 2, 197, 192
    device = torch.device("cuda")
    dtype = torch.float16  # Mamba fast_path 要求 fp16/bf16

    model = Mamba(d_model=D, bimamba_type="v2", if_divide_out=True).to(device=device, dtype=dtype)
    x = torch.randn(B, L, D, device=device, dtype=dtype, requires_grad=True)

    y = model(x)
    assert y.shape == (B, L, D), f"Shape mismatch: {y.shape}"
    print(f"✔  forward  : input {tuple(x.shape)} -> output {tuple(y.shape)}")

    # 反向传播
    loss = y.float().pow(2).mean()
    loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all(), "反向梯度异常"
    print(f"✔  backward : loss={loss.item():.4f}, grad finite")

    # 验证 v2 专属参数存在
    assert hasattr(model, "A_b_log"), "v2 应有 A_b_log"
    assert hasattr(model, "conv1d_b"), "v2 应有 conv1d_b"
    assert hasattr(model, "x_proj_b"), "v2 应有 x_proj_b"
    assert hasattr(model, "dt_proj_b"), "v2 应有 dt_proj_b"
    assert hasattr(model, "D_b"),      "v2 应有 D_b"
    print("✔  v2 反向分支参数齐全: A_b_log / conv1d_b / x_proj_b / dt_proj_b / D_b")


# ---------------------------------------------------------------------------
# T2: vim_tiny 顶层模型前向
# ---------------------------------------------------------------------------
def test_vim_tiny():
    _print_banner("T2  vim_tiny 顶层模型前向")
    from mamba_ssm.models.models_vim import vim_tiny

    device = torch.device("cuda")
    model = vim_tiny(num_classes=10).to(device=device, dtype=torch.float16)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"✔  模型参数量: {n_params:.2f} M")

    x = torch.randn(2, 3, 224, 224, device=device, dtype=torch.float16)
    logits = model(x)
    assert logits.shape == (2, 10), f"Shape mismatch: {logits.shape}"
    print(f"✔  forward  : input {tuple(x.shape)} -> logits {tuple(logits.shape)}")

    # 简单反向 (验证整条链路可训练)
    logits.float().sum().backward()
    print(f"✔  backward : 全链路梯度流通")


# ---------------------------------------------------------------------------
# T3: ISMEncoder 集成
# ---------------------------------------------------------------------------
def test_ism_encoder():
    _print_banner("T3  ISMEncoder (layers/ism.py) 前向 + 反向")
    from layers.ism import ISMEncoder

    B, L, D = 4, 50, 128
    device = torch.device("cuda")
    dtype = torch.float16

    model = ISMEncoder(d_model=D, seq_len=L, depth=2).to(device=device, dtype=dtype)
    x = torch.randn(B, L, D, device=device, dtype=dtype, requires_grad=True)

    y = model(x)
    assert y.shape == (B, L, D), f"Shape mismatch: {y.shape} (expect {(B, L, D)})"
    print(f"✔  forward  : input {tuple(x.shape)} -> output {tuple(y.shape)}")

    loss = y.float().pow(2).mean()
    loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    print(f"✔  backward : loss={loss.item():.4f}, grad finite")


# ---------------------------------------------------------------------------
def main():
    if not _check_cuda():
        sys.exit(0)

    tests = [
        ("T1 Mamba(bimamba_type='v2')", test_mamba_bimamba_v2),
        ("T2 vim_tiny",                 test_vim_tiny),
        ("T3 ISMEncoder",               test_ism_encoder),
    ]

    failed = []
    for name, fn in tests:
        try:
            fn()
        except Exception as e:
            print(f"\n✘  {name} FAILED: {type(e).__name__}: {e}")
            traceback.print_exc()
            failed.append(name)

    _print_banner("测试汇总")
    if failed:
        print(f"✘  {len(failed)}/{len(tests)} 项失败:")
        for n in failed:
            print(f"   - {n}")
        sys.exit(1)
    else:
        print(f"✔  全部 {len(tests)} 项通过")


if __name__ == "__main__":
    main()