import _path_setup # Fix sys.path
import argparse
import torch
import pandas as pd
import numpy as np
from dataset.data_loader import MMDataset, _collate_fn
from models.classifier import MSAClassifier

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--out", type=str, default="failed_cases.csv")
    args_cli = parser.parse_args()
    
    ckpt = torch.load(args_cli.ckpt, map_location="cpu")
    
    import argparse as inner_argparse
    args = inner_argparse.Namespace(**dict(ckpt["args"]))
    
    dataset_name = getattr(args, "dataset", "MOSI").lower()
    text_dim, audio_dim, video_dim = 768, 5, 20
    # force fallback to not error if no train set is built
    test_dataset = MMDataset(args, "test")
    from torch.utils.data import DataLoader
    test_loader = DataLoader(test_dataset, batch_size=32, collate_fn=_collate_fn)

    try:
        ism_num_layers = int(getattr(args, "ism_num_layers", getattr(args, "ism_depth", 1)))
    except:
        ism_num_layers = 1

    model = MSAClassifier(
        text_input_dim=text_dim,
        audio_input_dim=audio_dim,
        video_input_dim=video_dim,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_classes=args.num_classes,
        task_type=args.task_type,
        pool_type=getattr(args, "pool_type", "attention"),
        dropout=getattr(args, "dropout", 0.15),
        use_bert=getattr(args, "use_bert", True),
        ism_depth=ism_num_layers,
        ism_d_state=int(getattr(args, "ism_d_state", 16)),
        d_state=getattr(args, "d_state", 16),
        expand=getattr(args, "expand", 2),
        headdim=getattr(args, "headdim", 64),
        ngroups=getattr(args, "ngroups", 1),
        v_self_ratio=float(getattr(args, "v_self_ratio", 0.0) or 0.0),
        multi_task=bool(getattr(args, "multi_task", False)),
        aux_num_classes=int(getattr(args, "aux_num_classes", 0)),
        sub_loss_lambda=float(getattr(args, "sub_loss_lambda", 0.0) or 0.0)
    )
    model.load_state_dict(ckpt["model"].copy() if isinstance(ckpt["model"], dict) else ckpt["model"], strict=False)
    model.eval()

    all_ids, all_preds, all_truths = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            text = batch["text"]
            audio = batch["audio"]
            video = batch["vision"]
            audio_lengths = batch.get("audio_lengths", None)
            vision_lengths = batch.get("vision_lengths", None)
            
            out = model(text=text, audio=audio, video=video, 
                        audio_lengths=audio_lengths, vision_lengths=vision_lengths)
            
            logits = out[0] if isinstance(out, tuple) else out
            p = logits.squeeze(-1).numpy() if args.task_type == "regression" else logits.numpy()
            t = batch["labels"]["M"].numpy()
            all_ids.extend(batch["ids"])
            all_preds.extend(p.tolist())
            all_truths.extend(t.tolist())

    preds = np.array(all_preds)
    truths = np.array(all_truths)
    
    preds_a7 = np.clip(np.round(preds), -3, 3)
    truths_a7 = np.clip(np.round(truths), -3, 3)
    preds_a2 = (preds >= 0).astype(int)
    truths_a2 = (truths >= 0).astype(int)

    df = pd.DataFrame({
        "id": all_ids, 
        "pred": preds, 
        "truth": truths,
        "pred_a7": preds_a7,
        "truth_a7": truths_a7,
        "pred_a2": preds_a2,
        "truth_a2": truths_a2
    })
    df["err"] = np.abs(df["pred"] - df["truth"])
    df.to_csv(args_cli.out, index=False)
    
    print("\n======== ACC7 ERROR ANALYSIS ========")
    err_count = sum(df["pred_a7"] != df["truth_a7"])
    print(f"Total ACC7 Mistakes: {err_count} / {len(df)}")
    print("\nClass Distribution of Mistakes (Truth Class -> Mistake Count):")
    print(df[df["pred_a7"] != df["truth_a7"]]["truth_a7"].value_counts().sort_index())
    
    print("\nMean Prediction Value per True Class (Should go monotonically from -3 to +3):")
    print(df.groupby("truth_a7")["pred"].mean().round(2))
    
    print("\nACC2 Mistakes:")
    print(sum(df["pred_a2"] != df["truth_a2"]), "/", len(df))
    
    df["acc2_right_acc7_wrong"] = ((df["pred_a2"] == df["truth_a2"]) & (df["pred_a7"] != df["truth_a7"]))
    print(f"\nACC2 Correct BUT ACC7 Wrong: {sum(df['acc2_right_acc7_wrong'])}")
    
    print("\nTop 15 worst predictions (Acc2 right but Acc7 very wrong):")
    print(df[df["acc2_right_acc7_wrong"]].sort_values("err", ascending=False).head(15)[["id", "truth", "pred", "truth_a7", "pred_a7", "err"]].to_string(index=False))

main()
