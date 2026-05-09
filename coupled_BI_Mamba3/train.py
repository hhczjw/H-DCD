"""
单数据集训练入口:
    python train.py --dataset MOSI [--seed 42]
"""
from __future__ import annotations

import argparse
import json
import os

from configs import load_config
from dataset import MMDataLoader
from models import MSAClassifier
from trainer import Trainer
from utils import set_seed, setup_logger


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="MOSI",
                   choices=["MOSI", "MOSEI", "SIMS", "IEMOCAP", "MELD"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def main():
    cli = parse_args()
    args = load_config(cli.dataset)
    if cli.epochs:     args.epochs = cli.epochs
    if cli.batch_size: args.batch_size = cli.batch_size
    if cli.lr:         args.learning_rate = cli.lr

    set_seed(cli.seed)
    logger = setup_logger(args.logs_dir, name=f"MSA_{cli.dataset}")
    logger.info(f"Args: {json.dumps(vars(args), indent=2, default=str)}")

    # 1) 数据
    loaders = MMDataLoader(args, num_workers=int(args.num_workers))

    # 2) 模型
    text_dim, audio_dim, video_dim = args.feature_dims
    if args.use_bert:
        text_dim = 1   # 占位: trainer._forward_pred 把 BERT (B,3,L) 转成 (B,L,1)
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
        d_state=args.d_state,
        expand=args.expand,
        headdim=args.headdim,
        ngroups=args.ngroups,
        rope_fraction=args.rope_fraction,
        is_mimo=args.is_mimo,
        mimo_rank=args.mimo_rank,
        chunk_size=args.chunk_size,
        is_outproj_norm=args.is_outproj_norm,
    )

    # 3) Trainer
    trainer = Trainer(args, model, logger)

    # 4) 训练循环 + early stop
    best_score, patience, best_path = -1e9, 0, None
    higher_better = args.KeyEval != "Loss"
    sign = 1.0 if higher_better else -1.0

    for epoch in range(1, int(args.epochs) + 1):
        trainer.train_one_epoch(loaders["train"], epoch)
        val = trainer.evaluate(loaders["valid"], split="valid")
        key = "F1" if higher_better else "MAE"
        score = sign * val.get(key, val.get("Loss", 0.0))
        if score > best_score:
            best_score = score
            patience = 0
            best_path = os.path.join(args.checkpoints_dir, f"{cli.dataset}_best.pt")
            trainer.save(best_path)
        else:
            patience += 1
            if patience >= int(args.early_stop):
                logger.info(f"Early stop at epoch {epoch}")
                break

    # 5) 测试
    if best_path:
        trainer.load(best_path)
    test_metrics = trainer.evaluate(loaders["test"], split="test")
    os.makedirs(args.results_dir, exist_ok=True)
    with open(os.path.join(args.results_dir, f"{cli.dataset}_test.json"), "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)


if __name__ == "__main__":
    main()