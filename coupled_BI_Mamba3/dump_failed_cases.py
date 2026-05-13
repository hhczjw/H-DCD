import argparse
import torch
import numpy as np
import pandas as pd
import json
import os
from torch.utils.data import DataLoader

import _path_setup
from dataset.data_loader import MMDataset, _collate_fn
from models.classifier import MSAClassifier

class Config:
    def __init__(self, entries):
        for k, v in entries.items():
            setattr(self, k, v)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--out", type=str, default="failed_cases.csv")
    cli_args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading ckpt: {cli_args.ckpt}")
    ckpt = torch.load(cli_args.ckpt, map_location="cpu")
    args_dict = dict(ckpt["args"])
    args = Config(args_dict)

    # force test mode
    args.batch_size = 32

    print("Loading test dataset...")
    test_dataset = MMDataset(args, "test")
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=_collate_fn,
        pin_memory=True
    )

    print("Building model...")
    model = MSAClassifier(
        task_type=args.task_type,
        use_bert=args.use_bert,
        ism_num_layers=int(getattr(args, "ism_num_layers", 2)),
        ism_d_model=int(getattr(args, "ism_d_model", 128)),
        ism_d_state=int(getattr(args, "ism_d_state", 16)),
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
        is_mimo=getattr(args, "is_mimo", False),
        mimo_rank=getattr(args, "mimo_rank", 1),
        chunk_size=getattr(args, "chunk_size", 256),
        is_outproj_norm=getattr(args, "is_outproj_norm", False),
        v_self_ratio=float(getattr(args, "v_self_ratio", 0.0) or 0.0),
        multi_task=bool(getattr(args, "multi_task", False)),
        aux_num_classes=int(getattr(args, "aux_num_classes", 0)),
        sub_loss_lambda=float(getattr(args, "sub_loss_lambda", 0.0) or 0.0),
    )
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    all_ids = []
    all_preds = []
    all_truths = []

    use_cl = hasattr(model, "forward_with_contrastive")

    print("Evaluating...")
    with torch.no_grad():
        for batch in test_loader:
            for k in ["text", "audio", "vision", "index", "audio_lengths", "vision_lengths"]:
                if k in batch and isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(device)
            labels = batch["labels"]["M"].to(device)
            ids = batch["ids"]

            out = model.forward_pred(batch)
            if isinstance(out, tuple):
                logits = out[0]
            else:
                logits = out
            
            if args.task_type == "regression":
                p = logits.squeeze(-1).cpu().numpy()
            else:
                p = logits.cpu().numpy()
            t = batch["labels"]["M"].numpy()
            
            all_ids.extend(ids)
            all_preds.append(p)
            all_truths.append(t)

    preds = np.concatenate(all_preds, axis=0).flatten()
    truths = np.concatenate(all_truths, axis=0).flatten()

    # Calculate ACC7 predictions
    preds_a7 = np.clip(preds, -3.0, 3.0)
    truths_a7 = np.clip(truths, -3.0, 3.0)
    preds_a7_class = np.round(preds_a7)
    truths_a7_class = np.round(truths_a7)

    # Calculate ACC2 predictions (Has0)
    preds_a2 = (preds >= 0).astype(int)
    truths_a2 = (truths >= 0).astype(int)

    # Calculate ACC2 predictions (Non0)
    non_zeros = np.array([i for i, e in enumerate(truths) if e != 0])
    preds_a2_non0 = np.full_like(preds_a2, -1)
    truths_a2_non0 = np.full_like(truths_a2, -1)
    
    if len(non_zeros) > 0:
        preds_a2_non0[non_zeros] = (preds[non_zeros] > 0).astype(int)
        truths_a2_non0[non_zeros] = (truths[non_zeros] > 0).astype(int)

    df = pd.DataFrame({
        "id": all_ids,
        "truth_val": truths,
        "pred_val": preds,
        "truth_acc7": truths_a7_class,
        "pred_acc7": preds_a7_class,
        "truth_acc2": truths_a2,
        "pred_acc2": preds_a2,
        "truth_acc2_non0": truths_a2_non0,
        "pred_acc2_non0": preds_a2_non0,
        "acc7_correct": (truths_a7_class == preds_a7_class).astype(int),
        "acc2_correct": (truths_a2 == preds_a2).astype(int),
        "acc2_non0_correct": (truths_a2_non0 == preds_a2_non0).astype(int)
    })
    
    df["acc2_correct_but_acc7_wrong"] = ((df["acc2_correct"] == 1) & (df["acc7_correct"] == 0)).astype(int)
    df["err_abs"] = np.abs(df["truth_val"] - df["pred_val"])

    # Error analysis stats
    total_acc7 = df["acc7_correct"].mean()
    total_acc2 = df["acc2_correct"].mean()
    total_samples = len(df)
    
    print("\n" + "="*50)
    print(f"Total test samples: {total_samples}")
    print(f"Overall ACC7: {total_acc7:.4f}")
    print(f"Overall ACC2 (has0): {total_acc2:.4f}")
    
    acc2_yes_acc7_no_count = df["acc2_correct_but_acc7_wrong"].sum()
    print(f"Samples where ACC2 correct but ACC7 wrong: {acc2_yes_acc7_no_count} ({acc2_yes_acc7_no_count/total_samples*100:.1f}%)")
    
    # Save the dataframe
    df.to_csv(cli_args.out, index=False)
    print(f"Analysis saved to: {cli_args.out}")

    print("\n--- Detailed Error Patterns (Top 10 Worst Absolute Errors) ---")
    worst = df.sort_values(by="err_abs", ascending=False).head(10)[["id", "truth_val", "pred_val", "truth_acc7", "pred_acc7", "err_abs"]]
    print(worst.to_string(index=False))
    
    print("\n--- Error Breakdown by True Acc7 Class ---")
    grouped = df.groupby("truth_acc7").agg({
        "pred_acc7": "mean",
        "acc7_correct": ["sum", "count", "mean"],
        "acc2_correct": "mean"
    })
    print(grouped)

if __name__ == "__main__":
    main()
