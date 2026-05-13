"""
Smoke test: 验证 Run L 端到端 padding-aware 前向不报错
================================================================
覆盖:
    1. MSAClassifier 完整前向 (假数据 + 真实 lengths 分布)
    2. ISMEncoder 单独前向, 带 mask
    3. 验证 pad 区在各阶段输出仍为 0 (zero-out 实际生效)
"""
from __future__ import annotations

import os
import sys

# 注入 mamba_ssm
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT)
import _path_setup  # noqa: F401

import torch
from configs import load_config
from models import MSAClassifier


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[smoke] device={device}")

    args = load_config("MOSI")
    text_dim, audio_dim, video_dim = args.feature_dims
    ism_seq_len = int(getattr(args, "ism_seq_len", args.seq_lens[0]))

    print(f"[smoke] dims: text={text_dim}, audio={audio_dim}, video={video_dim}")
    print(f"[smoke] seq_lens={args.seq_lens}  ism_seq_len={ism_seq_len}")

    # ---- 构建模型 (Run L 配置) ----
    model = MSAClassifier(
        text_input_dim=text_dim,
        audio_input_dim=audio_dim,
        video_input_dim=video_dim,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_classes=args.num_classes,
        task_type=args.task_type,
        pool_type=args.pool_type,
        dropout=args.dropout,
        use_bert=args.use_bert,
        bert_pretrained=getattr(args, "bert_pretrained", "bert-base-uncased"),
        bert_finetune=False,
        ism_depth=3,
        ism_seq_len=ism_seq_len,
        ism_d_state=64,
        ism_mixer_type="bimamba3",
        ism_bimamba3_fusion=getattr(args, "ism_bimamba3_fusion", "add_divide2"),
        d_state=args.d_state,
        expand=args.expand,
        headdim=args.headdim,
        ngroups=args.ngroups,
        rope_fraction=args.rope_fraction,
        is_mimo=args.is_mimo,
        mimo_rank=args.mimo_rank,
        chunk_size=args.chunk_size,
        is_outproj_norm=args.is_outproj_norm,
        v_self_ratio=0.0,
        aux_num_classes=7,
        sub_loss_lambda=0.2,
    ).to(device)
    model.eval()

    # ---- 假数据 ----
    B = 4
    L = 50
    # text_bert: (B, 3, L) - input_ids, attention_mask, token_type_ids
    text_bert = torch.zeros(B, 3, L, dtype=torch.long, device=device)
    text_bert[:, 0, :10] = 100             # 假 token id
    text_bert[:, 1, :10] = 1                # attention mask: 前 10 步有效
    audio = torch.randn(B, L, audio_dim, device=device)
    vision = torch.randn(B, L, video_dim, device=device)

    # 模拟 unaligned_50 真实 lengths 分布
    audio_lengths = torch.tensor([5, 30, 50, 200], dtype=torch.long, device=device)  # 含 >50
    vision_lengths = torch.tensor([8, 25, 50, 350], dtype=torch.long, device=device)

    # 严格把 pad 区置 0 (模拟 data_loader)
    for b in range(B):
        a_eff = min(int(audio_lengths[b].item()), L)
        v_eff = min(int(vision_lengths[b].item()), L)
        audio[b, a_eff:, :] = 0.0
        vision[b, v_eff:, :] = 0.0

    print(f"[smoke] audio_lengths={audio_lengths.tolist()} (clip max=L=50)")
    print(f"[smoke] vision_lengths={vision_lengths.tolist()} (clip max=L=50)")

    # ---- 前向 ----
    with torch.no_grad():
        out = model(
            text=text_bert,
            audio=audio,
            video=vision,
            audio_lengths=audio_lengths,
            vision_lengths=vision_lengths,
        )

    if isinstance(out, dict):
        for k, v in out.items():
            if isinstance(v, torch.Tensor):
                print(f"[smoke] out[{k}].shape={tuple(v.shape)}  has_nan={torch.isnan(v).any().item()}  has_inf={torch.isinf(v).any().item()}")
            else:
                print(f"[smoke] out[{k}]={type(v).__name__}")
    else:
        print(f"[smoke] out.shape={tuple(out.shape)}  has_nan={torch.isnan(out).any().item()}")

    # ---- 单独验证 ISMEncoder mask zero-out 生效 ----
    print("\n[smoke] === Single ISMEncoder mask check ===")
    from layers.ism import ISMEncoder
    ism = ISMEncoder(
        d_model=128, seq_len=L, depth=2, d_state=64,
        mixer_type="bimamba3", bimamba3_headdim=64, bimamba3_fusion="add_divide2",
    ).to(device).eval()

    x = torch.randn(2, L, 128, device=device)
    mask = torch.zeros(2, L, dtype=torch.bool, device=device)
    mask[0, :5] = True       # 样本0 仅前 5 步有效
    mask[1, :45] = True      # 样本1 前 45 步有效

    # pad 区先 zero-out 再喂
    x = x * mask.unsqueeze(-1).to(x.dtype)
    with torch.no_grad():
        y = ism(x, mask=mask, return_cls=False)
    print(f"[smoke] ISM out shape={tuple(y.shape)}")
    # 检查 pad 位是否为 0
    for b in range(2):
        eff = mask[b].sum().item()
        pad_norm = y[b, eff:, :].abs().sum().item()
        head_norm = y[b, :eff, :].abs().sum().item()
        print(f"[smoke] sample{b}: eff={eff}, ||head||={head_norm:.4f}, ||pad_tail||={pad_norm:.6e}  "
              f"{'OK (≈0)' if pad_norm < 1e-3 else 'FAIL'}")

    print("\n[smoke] ✓ all forward passes succeeded")


if __name__ == "__main__":
    main()