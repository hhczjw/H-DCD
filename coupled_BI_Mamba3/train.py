"""
单数据集训练入口:
    python train.py --dataset MOSI [--seed 42]

改进:
    - 修复 early stop 监控指标: 直接使用 KeyEval 指定的指标
    - 支持多种子运行并汇总
"""
from __future__ import annotations

import _path_setup  # noqa: F401  必须在其他项目模块 import 之前, 注入内置 mamba_ssm 路径

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
    # ---- 训练相关 (覆盖 config.json) ----
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None, help="main learning rate")
    p.add_argument("--bert_lr", type=float, default=None)
    p.add_argument("--weight_decay", type=float, default=None)
    p.add_argument("--dropout", type=float, default=None)
    p.add_argument("--warmup_ratio", type=float, default=None)
    p.add_argument("--grad_accum_steps", type=int, default=None)
    p.add_argument("--early_stop", type=int, default=None)
    p.add_argument("--contrastive_weight", type=float, default=None)
    p.add_argument("--augment_prob", type=float, default=None)
    # ---- 模型结构 (覆盖 config.json) ----
    p.add_argument("--d_model", type=int, default=None)
    p.add_argument("--num_layers", type=int, default=None)
    p.add_argument("--d_state", type=int, default=None)
    p.add_argument("--ism_depth", type=int, default=None)
    p.add_argument("--ism_d_state", type=int, default=None)
    p.add_argument("--ism_mixer_type", type=str, default=None,
                   choices=["bimamba", "bimamba3"],
                   help="ISM mixer: bimamba (Mamba-2 兼容) or bimamba3 (Mamba-3)")
    p.add_argument("--ism_bimamba3_fusion", type=str, default=None,
                   choices=["add_divide2", "concat_proj", "gated", "add"])
    p.add_argument("--ism_bimamba3_is_mimo", type=lambda x: x.lower() == "true", default=None)
    p.add_argument("--device", type=str, default="cuda")
    # ---- 输出路径后缀, 多次实验互不覆盖 ----
    p.add_argument("--exp_tag", type=str, default="", help="可选 tag, 加在 checkpoint/log 文件名")
    return p.parse_args()


def _override(args, cli, key, src_attr=None):
    """如果 cli 中该字段非 None, 用它覆盖 args."""
    src = src_attr or key
    v = getattr(cli, src, None)
    if v is not None:
        setattr(args, key, v)


def main():
    cli = parse_args()
    args = load_config(cli.dataset)
    # 训练超参
    _override(args, cli, "epochs")
    _override(args, cli, "batch_size")
    _override(args, cli, "learning_rate", "lr")
    _override(args, cli, "bert_learning_rate", "bert_lr")
    _override(args, cli, "weight_decay")
    _override(args, cli, "dropout")
    _override(args, cli, "warmup_ratio")
    _override(args, cli, "grad_accum_steps")
    _override(args, cli, "early_stop")
    _override(args, cli, "contrastive_weight")
    _override(args, cli, "augment_prob")
    # 模型结构
    _override(args, cli, "d_model")
    _override(args, cli, "num_layers")
    _override(args, cli, "d_state")
    _override(args, cli, "ism_depth")
    _override(args, cli, "ism_d_state")
    _override(args, cli, "ism_mixer_type")
    _override(args, cli, "ism_bimamba3_fusion")
    _override(args, cli, "ism_bimamba3_is_mimo")

    set_seed(cli.seed)
    logger = setup_logger(args.logs_dir, name=f"MSA_{cli.dataset}")
    logger.info(f"Args: {json.dumps(vars(args), indent=2, default=str)}")

    # 1) 数据
    loaders = MMDataLoader(args, num_workers=int(args.num_workers))

    # 2) 模型
    text_dim, audio_dim, video_dim = args.feature_dims
    ism_seq_len = int(getattr(args, "ism_seq_len", args.seq_lens[0]))
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
        bert_finetune=getattr(args, "bert_finetune", True),
        ism_depth=int(getattr(args, "ism_depth", 1)),
        ism_seq_len=ism_seq_len,
        ism_d_state=int(getattr(args, "ism_d_state", 16)),
        # --- BiMamba3 (Mamba-3 双向) 接入参数, 默认 "bimamba" 走 Mamba-2 兼容旧路径 ---
        ism_mixer_type=getattr(args, "ism_mixer_type", "bimamba"),
        ism_bimamba3_headdim=int(getattr(args, "ism_bimamba3_headdim", 64)),
        ism_bimamba3_ngroups=int(getattr(args, "ism_bimamba3_ngroups", 1)),
        ism_bimamba3_rope_fraction=float(getattr(args, "ism_bimamba3_rope_fraction", 0.5)),
        ism_bimamba3_chunk_size=int(getattr(args, "ism_bimamba3_chunk_size", 64)),
        ism_bimamba3_is_mimo=bool(getattr(args, "ism_bimamba3_is_mimo", False)),
        ism_bimamba3_mimo_rank=int(getattr(args, "ism_bimamba3_mimo_rank", 4)),
        ism_bimamba3_is_outproj_norm=bool(getattr(args, "ism_bimamba3_is_outproj_norm", False)),
        ism_bimamba3_fusion=getattr(args, "ism_bimamba3_fusion", "add_divide2"),
        ism_bimamba3_share_mimo=bool(getattr(args, "ism_bimamba3_share_mimo", True)),
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

    # 4) 训练循环 + early stop (修复: 直接用 KeyEval 指定的指标)
    key_eval = args.KeyEval           # "Acc2", "F1", "Loss" 等
    higher_better = key_eval != "Loss" and key_eval != "MAE"
    best_score = -1e9 if higher_better else 1e9
    patience, best_path = 0, None

    logger.info(f"Early stop monitor: {key_eval} | higher_better={higher_better}")

    for epoch in range(1, int(args.epochs) + 1):
        trainer.train_one_epoch(loaders["train"], epoch)
        val = trainer.evaluate(loaders["valid"], split="valid")

        # 直接用 KeyEval 指定的指标做 early stop
        current_score = val.get(key_eval, val.get("MAE", val.get("Loss", 0.0)))

        improved = (current_score > best_score) if higher_better else (current_score < best_score)
        if improved:
            best_score = current_score
            patience = 0
            tag = f"_{cli.exp_tag}" if cli.exp_tag else ""
            best_path = os.path.join(args.checkpoints_dir, f"{cli.dataset}{tag}_seed{cli.seed}_best.pt")
            trainer.save(best_path)
            logger.info(f"  >> New best {key_eval}={current_score:.4f}")
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
    tag = f"_{cli.exp_tag}" if cli.exp_tag else ""
    with open(os.path.join(args.results_dir, f"{cli.dataset}{tag}_seed{cli.seed}_test.json"), "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)


if __name__ == "__main__":
    main()