"""
诊断 MOSI unaligned_50 数据真实形状 + audio_lengths 含义
确认:
  (1) pkl 原始 audio/vision shape 是多少
  (2) audio_lengths 是相对哪个维度计数
  (3) 经过 dataset _truncate(seq_lens=[50,50,50]) 后, lengths 的有效语义
"""
import pickle
import numpy as np
import os

PKL_PATH = "/media/zjw/951FB31A9E1EB7E0/dateSet/MSA-DataSets/CMU-MOSI/Processed/unaligned_50.pkl"
print(f"[INFO] Loading {PKL_PATH}")
with open(PKL_PATH, "rb") as f:
    data = pickle.load(f)

split = data["train"]
audio = np.asarray(split["audio"])
vision = np.asarray(split["vision"])
audio_lengths = np.asarray(split["audio_lengths"])
vision_lengths = np.asarray(split["vision_lengths"])

print("\n=== 原始 pkl shape ===")
print(f"  audio:  {audio.shape}")
print(f"  vision: {vision.shape}")
print(f"  audio_lengths range:  [{audio_lengths.min()}, {audio_lengths.max()}]  mean={audio_lengths.mean():.2f}")
print(f"  vision_lengths range: [{vision_lengths.min()}, {vision_lengths.max()}]  mean={vision_lengths.mean():.2f}")

# 检查 audio_lengths 是否对应"前 N 步有效, 后 pad"
print("\n=== 验证 audio_lengths 含义 (随机抽 3 个样本) ===")
for idx in [0, 100, 500]:
    if idx >= audio.shape[0]:
        continue
    L_a = audio.shape[1]                           # 实际长度维 (可能是 50 或 375)
    aL = int(audio_lengths[idx])
    a = audio[idx]                                  # (L_a, D_a)
    # 前 aL 步范数 vs 后 L_a-aL 步范数
    head = np.linalg.norm(a[:aL])     if aL > 0 else 0.0
    tail = np.linalg.norm(a[aL:])     if aL < L_a else 0.0
    print(f"  sample[{idx}]: audio_lengths={aL}, L_a={L_a}, "
          f"||head[:{aL}]||={head:.3f}  ||tail[{aL}:]||={tail:.6f}")

# 同样验证 vision
print("\n=== 验证 vision_lengths 含义 ===")
for idx in [0, 100, 500]:
    if idx >= vision.shape[0]:
        continue
    L_v = vision.shape[1]
    vL = int(vision_lengths[idx])
    v = vision[idx]
    head = np.linalg.norm(v[:vL])     if vL > 0 else 0.0
    tail = np.linalg.norm(v[vL:])     if vL < L_v else 0.0
    print(f"  sample[{idx}]: vision_lengths={vL}, L_v={L_v}, "
          f"||head[:{vL}]||={head:.3f}  ||tail[{vL}:]||={tail:.6f}")

# 截断到 50 后的 pad_ratio
print("\n=== 假设 seq_lens=[50,50,50] 截断后, pad_ratio (基于 Lt=50) ===")
Lt = 50
clipped_a = np.minimum(audio_lengths, Lt)
clipped_v = np.minimum(vision_lengths, Lt)
print(f"  audio:  effective_mean={clipped_a.mean():.2f}  pad_ratio={(Lt-clipped_a.mean())/Lt*100:.1f}%")
print(f"  vision: effective_mean={clipped_v.mean():.2f}  pad_ratio={(Lt-clipped_v.mean())/Lt*100:.1f}%")
print(f"  audio  样本中 lengths>=50 比例: {(audio_lengths>=Lt).mean()*100:.1f}%")
print(f"  vision 样本中 lengths>=50 比例: {(vision_lengths>=Lt).mean()*100:.1f}%")