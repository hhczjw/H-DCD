"""
检查 MOSI unaligned_50.pkl 的真实 seq lengths 分布
用于决策 padding mask 的必要性与改造强度
"""
import pickle
import numpy as np
import sys
import os

# --- 定位 pkl ---
CONFIG_ROOT = "/media/zjw/951FB31A9E1EB7E0/dateSet/MSA-DataSets"
PKL_PATH = os.path.join(CONFIG_ROOT, "CMU-MOSI/Processed/unaligned_50.pkl")

if not os.path.isfile(PKL_PATH):
    print(f"[ERR] Pkl not found: {PKL_PATH}")
    sys.exit(1)

print(f"[INFO] Loading {PKL_PATH} ...")
with open(PKL_PATH, "rb") as f:
    data = pickle.load(f)

print(f"[INFO] Top keys: {list(data.keys())}")

for split in ("train", "valid", "test"):
    if split not in data:
        continue
    d = data[split]
    print(f"\n==================== [{split}] N={len(d.get('id', d.get('regression_labels', [])))} ====================")
    print(f"Available keys: {list(d.keys())}")

    # 形状
    for mod in ("text", "text_bert", "audio", "vision"):
        if mod in d:
            x = np.asarray(d[mod])
            print(f"  {mod:12s} shape={x.shape}  dtype={x.dtype}")

    # seq_lengths 分布
    for key in ("audio_lengths", "vision_lengths", "text_lengths"):
        if key in d:
            lens = np.asarray(d[key])
            print(f"\n  >>> {key}:")
            print(f"      min={lens.min()}, max={lens.max()}, mean={lens.mean():.2f}, median={np.median(lens):.1f}")
            # 分位数
            pcts = [10, 25, 50, 75, 90, 95, 99]
            qs = np.percentile(lens, pcts)
            pct_str = "  ".join(f"P{p}={q:.0f}" for p, q in zip(pcts, qs))
            print(f"      {pct_str}")
            # pad 占比
            L = 50  # config 设置
            pad_ratio = 1.0 - lens.mean() / L
            print(f"      pad_ratio (avg over samples) = {pad_ratio:.1%}")
            # 稀疏样本比例
            for thresh in (10, 20, 30):
                ratio = (lens < thresh).mean()
                print(f"      P(len < {thresh}) = {ratio:.1%}")

    # BERT text 的实际有效长度 (attention_mask)
    if "text_bert" in d:
        text_bert = np.asarray(d["text_bert"])  # (N, 3, L)
        if text_bert.shape[1] == 3:
            att_mask = text_bert[:, 1, :]  # (N, L)
            real_lens = att_mask.sum(axis=1)
            print(f"\n  >>> text_bert attention_mask 统计:")
            print(f"      mean={real_lens.mean():.2f}, median={np.median(real_lens):.1f}")
            print(f"      P10={np.percentile(real_lens, 10):.0f}, P90={np.percentile(real_lens, 90):.0f}")
            print(f"      pad_ratio = {1 - real_lens.mean() / 50:.1%}")

    # 检查 pad 位置是否真的是 0
    if "audio" in d and "audio_lengths" in d:
        audio = np.asarray(d["audio"])
        lens = np.asarray(d["audio_lengths"])
        # 取前 5 样本看 pad 区域是否为 0
        for i in range(min(3, len(lens))):
            L_real = int(lens[i])
            if L_real < audio.shape[1]:
                pad_region = audio[i, L_real:, :]
                is_zero = np.allclose(pad_region, 0)
                print(f"      sample[{i}]: len={L_real}, pad[{L_real}:]."
                      f"abs_max={np.abs(pad_region).max():.6f}  is_zero={is_zero}")

print("\n[DONE]")