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
    p.add_argument("--bert_pretrained", type=str, default=None,
                   help="文本预训练模型名 (覆盖 config.json); e.g. bert-base-uncased / roberta-base")
    p.add_argument("--bert_finetune", type=lambda x: x.lower() == "true", default=None)
    p.add_argument("--weight_decay", type=float, default=None)
    p.add_argument("--dropout", type=float, default=None)
    p.add_argument("--warmup_ratio", type=float, default=None)
    p.add_argument("--grad_accum_steps", type=int, default=None)
    p.add_argument("--grad_clip", type=float, default=None,
                   help="梯度裁剪 max_norm; 默认 0.5. Mamba3 长训易出梯度尖峰, 不建议 > 1.0")
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
    # ---- 跨模态 V_self 注入 (问题 ③ 修复) ----
    p.add_argument("--v_self_ratio", type=float, default=None,
                   help="cross-modal V 通道中 tgt 自身的比例; 0=关闭(默认), 推荐 0.2~0.4")
    # ---- 复合损失 (回归 + 离散 CE 辅助头, 直接攻 Acc7) ----
    p.add_argument("--aux_cls_weight", type=float, default=None,
                   help="辅助分类 loss 权重 alpha; 0 = 关闭. 推荐 0.1~0.5")
    p.add_argument("--aux_num_classes", type=int, default=None,
                   help="辅助分类类别数; MOSI 应为 7 (=2*clip_range+1)")
    p.add_argument("--sub_loss_lambda", type=float, default=None,
                   help="模态级 sub_loss 权重 (对齐 MSAmba sub_fc_T/A/V); 0 = 关闭. 推荐 0.3~0.5")
    # ---- early stop 监控指标 (覆盖 config.json KeyEval) ----
    p.add_argument("--key_eval", type=str, default=None,
                   choices=["Acc2", "MAE", "F1", "Loss", "Acc7"],
                   help="early stop 监控指标; 默认沿用 config.json")
    # ---- 辅助 ckpt 指标 (dual ckpt) ----
    p.add_argument("--secondary_metric", type=str, default=None,
                   choices=["MAE", "Acc7", "Acc5", "Acc2", "F1", "none"],
                   help="辅助 ckpt 监控指标; 默认: regression -> MAE; 设为 'none' 关闭. "
                        "想冲 Acc7 时建议设为 Acc7. 不能与 --key_eval 相同, 否则自动关闭辅助 ckpt.")
    # ---- EMA 影子权重 ----
    p.add_argument("--ema_decay", type=float, default=None,
                   help="EMA 衰减系数; >0 启用, 推荐 0.999. 0/None=关闭")
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
    _override(args, cli, "bert_pretrained")
    _override(args, cli, "bert_finetune")
    _override(args, cli, "weight_decay")
    _override(args, cli, "dropout")
    _override(args, cli, "warmup_ratio")
    _override(args, cli, "grad_accum_steps")
    _override(args, cli, "grad_clip")
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
    _override(args, cli, "v_self_ratio")
    # 复合损失
    _override(args, cli, "aux_cls_weight")
    _override(args, cli, "aux_num_classes")
    _override(args, cli, "sub_loss_lambda")
    # 监控/EMA
    _override(args, cli, "KeyEval", "key_eval")
    _override(args, cli, "ema_decay")

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
        v_self_ratio=float(getattr(args, "v_self_ratio", 0.0) or 0.0),
        multi_task=bool(getattr(args, "multi_task", False)),
    )

    # 3) Trainer
    trainer = Trainer(args, model, logger)

    # 4) 训练循环 + early stop + 双 ckpt
    # 主监控: KeyEval (e.g. Acc2) 决定 early stop 与 "primary" ckpt
    # 辅助监控: 回归任务额外保存 MAE 最优 ckpt (用于 Acc7 评估)
    key_eval = args.KeyEval                     # "Acc2", "F1", "Loss" 等
    higher_better = key_eval not in ("Loss", "MAE")
    best_primary = -1e9 if higher_better else 1e9
    patience = 0

    tag = f"_{cli.exp_tag}" if cli.exp_tag else ""
    primary_path = os.path.join(args.checkpoints_dir, f"{cli.dataset}{tag}_seed{cli.seed}_best_{key_eval}.pt")

    # 仅回归任务且 KeyEval != MAE 时启用辅助 ckpt
    # ---- 辅助 ckpt 指标解析 (CLI 优先, 默认 MAE) ----
    # 规则:
    #   - cli.secondary_metric == "none"            -> 关闭辅助 ckpt
    #   - cli.secondary_metric is not None          -> 使用 CLI 指定值
    #   - cli.secondary_metric is None & 回归任务   -> 默认 MAE (向后兼容)
    #   - 与 key_eval 相同                          -> 自动关闭, 避免重复
    if cli.secondary_metric is None:
        secondary_metric = "MAE" if args.task_type == "regression" else None
    elif cli.secondary_metric == "none":
        secondary_metric = None
    else:
        secondary_metric = cli.secondary_metric

    use_dual_ckpt = (
        args.task_type == "regression"
        and secondary_metric is not None
        and secondary_metric != key_eval
    )
    # 辅助指标方向 (higher_better)
    sec_higher_better = (secondary_metric not in ("Loss", "MAE")) if secondary_metric else False
    best_secondary = (-1e9 if sec_higher_better else 1e9) if use_dual_ckpt else None
    secondary_path = (
        os.path.join(args.checkpoints_dir, f"{cli.dataset}{tag}_seed{cli.seed}_best_{secondary_metric}.pt")
        if use_dual_ckpt else None
    )

    logger.info(
        f"Early stop monitor: {key_eval} | higher_better={higher_better} | "
        f"dual_ckpt={'on(' + str(secondary_metric) + ',higher_better=' + str(sec_higher_better) + ')' if use_dual_ckpt else 'off'}"
    )

    for epoch in range(1, int(args.epochs) + 1):
        trainer.train_one_epoch(loaders["train"], epoch)
        val = trainer.evaluate(loaders["valid"], split="valid")

        # 主 ckpt: KeyEval 监控 + early stop 判定
        primary_score = val.get(key_eval, val.get("MAE", val.get("Loss", 0.0)))
        improved = (primary_score > best_primary) if higher_better else (primary_score < best_primary)
        if improved:
            best_primary = primary_score
            patience = 0
            trainer.save(primary_path)
            logger.info(f"  >> New best {key_eval}={primary_score:.4f}  [primary ckpt]")
        else:
            patience += 1

        # 辅助 ckpt: 用 secondary_metric 单独保存
        if use_dual_ckpt:
            sec_default = -1e9 if sec_higher_better else 1e9
            sec_score = val.get(secondary_metric, sec_default)
            sec_improved = (sec_score > best_secondary) if sec_higher_better else (sec_score < best_secondary)
            if sec_improved:
                best_secondary = sec_score
                trainer.save(secondary_path)
                logger.info(f"  >> New best {secondary_metric}={sec_score:.4f}  [secondary ckpt]")

        if patience >= int(args.early_stop):
            logger.info(f"Early stop at epoch {epoch}")
            break

    # 5) 测试: 分别 load 两个 ckpt 各报一次
    os.makedirs(args.results_dir, exist_ok=True)
    test_results = {}

    logger.info(f"=== Testing with PRIMARY ckpt (best {key_eval}) ===")
    trainer.load(primary_path)
    test_results[f"primary_{key_eval}"] = trainer.evaluate(loaders["test"], split="test_primary")

    if use_dual_ckpt and os.path.isfile(secondary_path):
        logger.info(f"=== Testing with SECONDARY ckpt (best {secondary_metric}) ===")
        trainer.load(secondary_path)
        test_results[f"secondary_{secondary_metric}"] = trainer.evaluate(loaders["test"], split="test_secondary")

    out_json = os.path.join(args.results_dir, f"{cli.dataset}{tag}_seed{cli.seed}_test.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(test_results, f, indent=2)
    logger.info(f"Test results saved: {out_json}")


if __name__ == "__main__":
    main()