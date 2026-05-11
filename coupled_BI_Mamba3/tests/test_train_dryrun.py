"""
训练前端到端 Dry-Run + 显存评估
==================================
用 mock 数据 (不需要真实数据集 .pkl) 完整跑一遍:
    MSAClassifier 前向 → loss → 反向 → optimizer.step()

同时输出:
    - 模型参数量
    - peak GPU 显存
    - 单 batch 前向 / 反向耗时

运行:
    cd H-DCD/coupled_BI_Mamba3
    python tests/test_train_dryrun.py [--dataset MOSI] [--batch_size 16]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "mamba"))
sys.path.insert(0, ROOT)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="MOSI",
                   choices=["MOSI", "MOSEI", "SIMS", "IEMOCAP", "MELD"])
    p.add_argument("--batch_size", type=int, default=None,
                   help="覆盖 config.json 中的 batch_size")
    p.add_argument("--no_bert", action="store_true",
                   help="跳过 BERT (使用 nn.Embedding fallback), 便于无网环境测试")
    return p.parse_args()


def build_mock_batch(args, batch_size: int, device, dtype):
    """根据 config 中的 seq_lens / feature_dims 生成 mock batch."""
    text_dim, audio_dim, video_dim = args.feature_dims
    L_t, L_a, L_v = args.seq_lens

    # text: 若 use_bert, 期望 (B, 3, L_t) long; 否则 (B, L_t, text_dim) float
    if getattr(args, "use_bert", False):
        # input_ids 在 BERT 词表内 (< 30522)
        text = torch.randint(0, 30000, (batch_size, 3, L_t), device=device, dtype=torch.long)
        # attention_mask 全 1, token_type_ids 全 0
        text[:, 1] = 1
        text[:, 2] = 0
    else:
        text = torch.randn(batch_size, L_t, text_dim, device=device, dtype=dtype)

    audio = torch.randn(batch_size, L_a, audio_dim, device=device, dtype=dtype)
    vision = torch.randn(batch_size, L_v, video_dim, device=device, dtype=dtype)

    if args.task_type == "regression":
        labels = torch.randn(batch_size, device=device, dtype=torch.float32)
    else:
        labels = torch.randint(0, args.num_classes, (batch_size,), device=device, dtype=torch.long)

    batch = {
        "text": text, "audio": audio, "vision": vision,
        "audio_lengths":  torch.full((batch_size,), L_a, device=device, dtype=torch.long),
        "vision_lengths": torch.full((batch_size,), L_v, device=device, dtype=torch.long),
        "labels": {"M": labels},
        "ids": [f"mock_{i}" for i in range(batch_size)],
        "index": torch.arange(batch_size, device=device, dtype=torch.long),
    }
    return batch


def main():
    cli = parse_args()
    if not torch.cuda.is_available():
        print("⚠  无 CUDA, 跳过 dry-run。")
        sys.exit(0)

    from configs import load_config
    from models import MSAClassifier

    args = load_config(cli.dataset)
    if cli.batch_size:
        args.batch_size = cli.batch_size
    if cli.no_bert:
        args.use_bert = False

    print(f"=== Dataset: {cli.dataset}  batch_size={args.batch_size}  use_bert={args.use_bert} ===")
    print(f"feature_dims={args.feature_dims}  seq_lens={args.seq_lens}  task={args.task_type}")

    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)

    # ------------------------------------------------------------------
    # 1) 构建模型 (与 train.py 完全一致的参数)
    # ------------------------------------------------------------------
    text_dim, audio_dim, video_dim = args.feature_dims
    ism_seq_len = int(getattr(args, "ism_seq_len", args.seq_lens[0]))
    model = MSAClassifier(
        text_input_dim=text_dim, audio_input_dim=audio_dim, video_input_dim=video_dim,
        d_model=args.d_model, num_layers=args.num_layers,
        num_classes=args.num_classes, task_type=args.task_type,
        pool_type=args.pool_type, dropout=args.dropout,
        use_bert=args.use_bert,
        bert_pretrained=getattr(args, "bert_pretrained", "bert-base-uncased"),
        bert_finetune=getattr(args, "bert_finetune", True),
        ism_depth=int(getattr(args, "ism_depth", 1)),
        ism_seq_len=ism_seq_len,
        ism_d_state=int(getattr(args, "ism_d_state", 16)),
        d_state=args.d_state, expand=args.expand, headdim=args.headdim,
        ngroups=args.ngroups, rope_fraction=args.rope_fraction,
        is_mimo=args.is_mimo, mimo_rank=args.mimo_rank,
        chunk_size=args.chunk_size, is_outproj_norm=args.is_outproj_norm,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"\n[Model] 总参数 {n_params:.2f}M | 可训练 {n_trainable:.2f}M")

    # ------------------------------------------------------------------
    # 2) 构造 mock batch
    # ------------------------------------------------------------------
    batch = build_mock_batch(args, args.batch_size, device, dtype=torch.float32)

    # ------------------------------------------------------------------
    # 3) Forward
    # ------------------------------------------------------------------
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    torch.cuda.synchronize()
    t0 = time.time()
    output = model(batch["text"], batch["audio"], batch["vision"])
    torch.cuda.synchronize()
    fwd_ms = (time.time() - t0) * 1000

    # 兼容 dict / tensor 输出
    if isinstance(output, dict):
        pred = output.get("M", output.get("logits", next(iter(output.values()))))
    else:
        pred = output
    print(f"\n[Forward]  output shape: {tuple(pred.shape)}  ({fwd_ms:.1f} ms)")

    # ------------------------------------------------------------------
    # 4) Loss + Backward + Step
    # ------------------------------------------------------------------
    label = batch["labels"]["M"]
    if args.task_type == "regression":
        loss = torch.nn.functional.l1_loss(pred.squeeze(-1).float(), label.float())
    else:
        loss = torch.nn.functional.cross_entropy(pred.float(), label)

    torch.cuda.synchronize()
    t0 = time.time()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    torch.cuda.synchronize()
    bwd_ms = (time.time() - t0) * 1000

    assert torch.isfinite(loss), f"Loss 非有限: {loss.item()}"
    print(f"[Backward] loss={loss.item():.4f}  ({bwd_ms:.1f} ms)")

    # ------------------------------------------------------------------
    # 5) 显存
    # ------------------------------------------------------------------
    peak_mb = torch.cuda.max_memory_allocated(device) / 1024 / 1024
    reserved_mb = torch.cuda.max_memory_reserved(device) / 1024 / 1024
    print(f"\n[GPU]      peak allocated: {peak_mb:.0f} MB  |  peak reserved: {reserved_mb:.0f} MB")
    print(f"           单 step 总耗时 ≈ {fwd_ms + bwd_ms:.0f} ms")

    print("\n✔  Dry-run 完成, 可以开始训练。")


if __name__ == "__main__":
    main()