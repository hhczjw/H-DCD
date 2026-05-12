"""
P0 改造冒烟测试: GLCE + ISM-CLS sub_loss
=========================================
目标: 在无真实数据 / 无 BERT 下验证
    1. ISMEncoder(return_cls=True) 返回正确形状
    2. MSAClassifier(sub_loss_lambda>0) forward 返回 dict{logits, aux_logits, sub_T/A/V}
    3. RegressionWithDiscreteCE 能吃 sub_outputs, 产出有限 loss
    4. loss.backward() 能跑通, 梯度流经 sub_fc_T/A/V

运行:
    cd H-DCD/coupled_BI_Mamba3
    python tests/test_sub_loss_smoke.py
"""
from __future__ import annotations

import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "mamba"))
sys.path.insert(0, ROOT)


def main():
    if not torch.cuda.is_available():
        print("⚠  无 CUDA, 跳过.")
        return

    from layers.ism import ISMEncoder
    from models import MSAClassifier
    from losses import RegressionWithDiscreteCE

    device = torch.device("cuda")
    torch.manual_seed(0)

    # ---- 1. ISMEncoder return_cls ----
    ism = ISMEncoder(d_model=128, seq_len=50, depth=2, d_state=16,
                     mixer_type="bimamba").to(device)
    x = torch.randn(4, 50, 128, device=device)
    seq_only = ism(x)
    assert seq_only.shape == (4, 50, 128), f"seq_only shape wrong: {seq_only.shape}"
    seq, cls = ism(x, return_cls=True)
    assert seq.shape == (4, 50, 128), f"seq shape wrong: {seq.shape}"
    assert cls.shape == (4, 128), f"cls shape wrong: {cls.shape}"
    print(f"[1/4] ✔  ISMEncoder return_cls OK | seq={tuple(seq.shape)} cls={tuple(cls.shape)}")

    # ---- 2. MSAClassifier with sub_loss_lambda>0 ----
    B, L = 4, 50
    model = MSAClassifier(
        text_input_dim=768, audio_input_dim=5, video_input_dim=20,
        d_model=128, num_layers=2, num_classes=1, task_type="regression",
        pool_type="attention", dropout=0.1,
        use_bert=False,  # 不依赖网络
        ism_depth=2, ism_seq_len=L, ism_d_state=16, ism_mixer_type="bimamba",
        d_state=16, expand=2, headdim=16, ngroups=1, rope_fraction=0.5,
        is_mimo=False, mimo_rank=4, chunk_size=64, is_outproj_norm=False,
        aux_num_classes=7,          # 开启 aux CE
        sub_loss_lambda=0.3,        # 开启 sub_loss
    ).to(device)
    assert model.use_sub_loss, "use_sub_loss 应为 True"
    assert model.aux_head is not None, "aux_head 应已创建"

    text = torch.randn(B, L, 768, device=device)
    audio = torch.randn(B, L, 5, device=device)
    video = torch.randn(B, L, 20, device=device)
    out = model(text, audio, video)
    assert isinstance(out, dict), f"应返回 dict, got {type(out)}"
    assert "logits" in out and out["logits"].shape == (B, 1)
    assert "aux_logits" in out and out["aux_logits"].shape == (B, 7)
    for k in ("sub_T", "sub_A", "sub_V"):
        assert k in out and out[k].shape == (B, 1), f"{k} 形状错"
    print(f"[2/4] ✔  MSAClassifier 多头输出 OK | keys={list(out.keys())}")

    # ---- 3. RegressionWithDiscreteCE ----
    crit = RegressionWithDiscreteCE(alpha=0.3, num_aux_classes=7,
                                    sub_loss_lambda=0.3).to(device)
    label = torch.randn(B, device=device).clamp(-3, 3)
    loss = crit(out["logits"], out["aux_logits"], label,
                sub_outputs=(out["sub_T"], out["sub_A"], out["sub_V"]))
    assert torch.isfinite(loss), f"loss 非有限: {loss.item()}"
    print(f"[3/4] ✔  复合 loss OK | loss={loss.item():.4f}")

    # ---- 4. Backward 梯度流验证 ----
    loss.backward()
    for name in ("sub_fc_T", "sub_fc_A", "sub_fc_V"):
        mod = getattr(model, name)
        assert mod.weight.grad is not None, f"{name} 无梯度"
        assert mod.weight.grad.abs().sum() > 0, f"{name} 梯度全 0"
    # 主 head 与 aux_head 也应有梯度
    assert model.head[-1].weight.grad is not None and model.head[-1].weight.grad.abs().sum() > 0
    assert model.aux_head[-1].weight.grad is not None and model.aux_head[-1].weight.grad.abs().sum() > 0
    print(f"[4/4] ✔  Backward 梯度流验证 OK (sub_fc_T/A/V + head + aux_head 均有梯度)")

    print("\n✅  所有冒烟测试通过, 可以开始 Run E 实验")


if __name__ == "__main__":
    main()