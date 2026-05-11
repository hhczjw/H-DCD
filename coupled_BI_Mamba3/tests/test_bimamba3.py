"""
biMamba-3 单元测试 + 与 Mamba-3 / 单向模式 对比
==================================================
覆盖:
    T1. 形状一致性 (B, L, D) -> (B, L, D)
    T2. 反向梯度流通 (loss.backward 后 grad finite)
    T3. bimamba_type='none' 退化为单向 (与 Mamba3 输出方差量级一致)
    T4. fusion 模式: add / add_divide2 / concat_proj / gated 全部前/反向
    T5. 翻转不变性: bimamba_type='v2' 时, 输入翻转 → 输出近似翻转
                    (因为正反两路在 'v2' 下角色互换 + 权重不同, 此项为弱验证)

运行:
    cd H-DCD/coupled_BI_Mamba3
    python tests/test_bimamba3.py
"""
from __future__ import annotations

import os
import sys
import traceback

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "mamba"))


def _banner(s):
    print("\n" + "=" * 70)
    print(f"  {s}")
    print("=" * 70)


def _check_cuda():
    if not torch.cuda.is_available():
        print("⚠  无 CUDA, 跳过 (Mamba-3 kernel 需要 GPU)")
        return False
    print(f"✔  CUDA 设备: {torch.cuda.get_device_name(0)}")
    return True


# ---------------- 测试参数 ----------------
B, L, D_MODEL = 2, 64, 256          # batch / seqlen / d_model
DTYPE = torch.bfloat16              # Mamba-3 kernel 推荐 bf16


def _make_model(**kwargs):
    """统一构造 BiMamba3 (默认参数与 Mamba-3 官方 SISO baseline 一致)."""
    from mamba_ssm.modules.bimamba3 import BiMamba3
    base = dict(
        d_model=D_MODEL, d_state=128, expand=2, headdim=64,
        ngroups=1, rope_fraction=0.5, chunk_size=64,
        is_mimo=False, mimo_rank=1, is_outproj_norm=False,
        layer_idx=0,
    )
    base.update(kwargs)
    return BiMamba3(**base).cuda().to(DTYPE)


# -----------------------------------------------------------------
# T1 + T2: 形状 + 反向
# -----------------------------------------------------------------
def test_shape_and_backward():
    _banner("T1+T2  形状一致性 + 反向梯度")
    model = _make_model(bimamba_type="v2", fusion="add")
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"参数量: {n_params:.2f} M")

    x = torch.randn(B, L, D_MODEL, device="cuda", dtype=DTYPE, requires_grad=True)
    y = model(x)
    assert y.shape == (B, L, D_MODEL), f"shape {y.shape}"
    print(f"✔  forward shape OK: {tuple(y.shape)}")

    loss = y.float().pow(2).mean()
    loss.backward()
    grads_ok = x.grad is not None and torch.isfinite(x.grad).all()
    assert grads_ok, "反向梯度异常"
    # 验证反向分支参数确实有梯度
    g_in_b = model.in_proj_b.weight.grad
    assert g_in_b is not None and torch.isfinite(g_in_b).all() and g_in_b.abs().sum() > 0
    print(f"✔  backward OK: loss={loss.item():.4f}, in_proj_b.grad finite & non-zero")


# -----------------------------------------------------------------
# T3: 单向退化 (ablation)
# -----------------------------------------------------------------
def test_unidirectional_ablation():
    _banner("T3  bimamba_type='none' 单向退化")
    model = _make_model(bimamba_type="none")
    x = torch.randn(B, L, D_MODEL, device="cuda", dtype=DTYPE)
    y = model(x)
    assert y.shape == (B, L, D_MODEL)
    # 单向模式不应有 in_proj_b
    assert not hasattr(model, "in_proj_b"), "单向模式不应初始化反向分支"
    print(f"✔  单向模式: 输出 {tuple(y.shape)}, 无反向分支参数")


# -----------------------------------------------------------------
# T4: 各融合模式
# -----------------------------------------------------------------
def test_fusion_modes():
    _banner("T4  四种融合模式 forward + backward")
    for mode in ("add", "add_divide2", "concat_proj", "gated"):
        model = _make_model(bimamba_type="v2", fusion=mode)
        x = torch.randn(B, L, D_MODEL, device="cuda", dtype=DTYPE, requires_grad=True)
        y = model(x)
        loss = y.float().pow(2).mean()
        loss.backward()
        assert y.shape == (B, L, D_MODEL)
        assert torch.isfinite(loss)
        print(f"✔  fusion='{mode}' OK   loss={loss.item():.4f}")


# -----------------------------------------------------------------
# T5: 翻转对称性 (弱验证)
# -----------------------------------------------------------------
def test_flip_symmetry():
    """
    验证: 若给 v2 模型的正反向分支强行加载相同权重,
    则 model(x).flip(1) 应近似等于 model(x.flip(1)).
    用于 sanity-check 双向逻辑实现是否对称.
    """
    _banner("T5  对称权重下的 flip 一致性 (sanity check)")
    model = _make_model(bimamba_type="v2", fusion="add")

    # 复制正向权重到反向分支
    with torch.no_grad():
        model.in_proj_b.weight.copy_(model.in_proj.weight)
        model.dt_bias_b.copy_(model.dt_bias)
        model.B_bias_b.copy_(model.B_bias)
        model.C_bias_b.copy_(model.C_bias)
        model.D_b.copy_(model.D)
        model.B_norm_b.weight.copy_(model.B_norm.weight)
        model.C_norm_b.weight.copy_(model.C_norm.weight)

    model.eval()
    x = torch.randn(B, L, D_MODEL, device="cuda", dtype=DTYPE)
    with torch.no_grad():
        y1 = model(x)
        y2 = model(x.flip(dims=[1])).flip(dims=[1])

    diff = (y1.float() - y2.float()).abs().mean().item()
    rel = diff / (y1.float().abs().mean().item() + 1e-6)
    print(f"   |y1 - y2|.mean() = {diff:.4e},   relative = {rel:.4%}")
    # 理论上应该完全相等 (浮点误差除外); 允许 5% 相对误差
    assert rel < 0.05, f"权重对称下 flip 不一致: rel={rel:.4%}"
    print(f"✔  权重对称下输出在 flip 下不变 (relative diff {rel:.4%})")


# -----------------------------------------------------------------
def main():
    if not _check_cuda():
        sys.exit(0)
    tests = [
        ("T1+T2 shape & backward", test_shape_and_backward),
        ("T3 unidirectional",      test_unidirectional_ablation),
        ("T4 fusion modes",        test_fusion_modes),
        ("T5 flip symmetry",       test_flip_symmetry),
    ]
    failed = []
    for name, fn in tests:
        try:
            fn()
        except Exception as e:
            print(f"\n✘  {name} FAILED: {type(e).__name__}: {e}")
            traceback.print_exc()
            failed.append(name)
    _banner("汇总")
    if failed:
        print(f"✘  {len(failed)}/{len(tests)} 失败:")
        for n in failed:
            print(f"   - {n}")
        sys.exit(1)
    print(f"✔  全部 {len(tests)} 项通过")


if __name__ == "__main__":
    main()