#!/usr/bin/env python3
"""
离线提取 Data2Vec 音频特征 (完全对齐 CAGMamba 管线)
==================================================

读取 CAGMamba 格式的 label.csv + WAV 文件,
用 Data2Vec 模型提取特征, 保存为 .pkl 供 Coupled-BI-Mamba3 加载.

用法:
    python scripts/extract_audio_data2vec.py \
        --csv_path /path/to/MOSI/label.csv \
        --audio_dir /path/to/MOSI/wav \
        --output ./features/mosi_audio_data2vec.pkl \
        --model_name facebook/data2vec-audio-base-960h \
        --target_frames 50

管线 (对齐 CAGMamba):
    label.csv → video_id/clip_id → {audio_dir}/{video_id}/{clip_id}.wav
    → torchaudio.load() → mono → Wav2Vec2FeatureExtractor
    → Data2Vec Audio Model → last_hidden_state
    → attention-based 有效帧检测 → adaptive_avg_pool1d → (N, target_frames, hidden_dim)
    → 保存为 .pkl

依赖:
    pip install torchaudio transformers soundfile pandas tqdm
"""

import argparse
import os
import pickle
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm


def load_audio_wav(audio_path: str, target_sr: int = 16000) -> np.ndarray:
    """
    加载 WAV 文件, 返回 (T,) float32 单声道波形.
    使用 librosa (纯 Python, 无 PyTorch ABI 依赖).
    加载失败时返回零向量 (对应 CAGMamba 的 invalid_files).
    """
    try:
        import librosa
        audio, sr = librosa.load(audio_path, sr=target_sr, mono=True)
        return audio.astype(np.float32)
    except Exception:
        pass

    try:
        import soundfile as sf
        audio, sr = sf.read(audio_path, dtype='float32')
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != target_sr:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        return audio.astype(np.float32)
    except Exception:
        pass

    # 回退: torchaudio
    try:
        import torchaudio
        sound, sr = torchaudio.load(audio_path)
        sound_mono = torch.mean(sound, dim=0, keepdim=False).numpy()
        if sr != target_sr:
            import librosa
            sound_mono = librosa.resample(sound_mono, orig_sr=sr, target_sr=target_sr)
        return sound_mono.astype(np.float32)
    except Exception:
        pass

    # 所有方法失败 → 返回零向量 (已知损坏文件, 不影响训练)
    raise RuntimeError(f"无法加载 {audio_path}")


def load_and_process_audio(audio_path: str, feature_extractor, max_length=96000):
    """
    CAGMamba 风格音频加载:
    1. soundfile/torchaudio 加载 → (T,) float32 单声道
    2. Wav2Vec2FeatureExtractor 处理
    """
    sound_mono = load_audio_wav(audio_path)  # np.ndarray (T,) float32

    features = feature_extractor(
        sound_mono,
        sampling_rate=16000,
        max_length=max_length,
        return_attention_mask=True,
        truncation=True,
        padding="max_length",
    )
    input_values = torch.tensor(
        np.array(features['input_values']), dtype=torch.float32
    ).squeeze()
    attention_mask = torch.tensor(
        np.array(features['attention_mask']), dtype=torch.long
    ).squeeze()
    return input_values, attention_mask


def extract_data2vec_features(
    model, input_values, attention_mask, target_frames, device
):
    """
    简化特征提取 (不依赖 attention maps):
    1. Data2Vec forward → last_hidden_state (内部已通过 mask 处理 padding)
    2. Adaptive pool → target_frames
    """
    with torch.no_grad():
        audio_out = model(
            input_values.unsqueeze(0).to(device),
            attention_mask=attention_mask.unsqueeze(0).to(device),
        )
    hs = audio_out.last_hidden_state.squeeze(0)  # (T_raw, D)

    if target_frames > 0 and hs.size(0) != target_frames:
        hs = F.adaptive_avg_pool1d(
            hs.T.unsqueeze(0), target_frames
        ).squeeze(0).T  # (target_frames, D)

    return hs.cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv_path", type=str, required=True,
        help="Path to label.csv (CAGMamba format)"
    )
    parser.add_argument(
        "--audio_dir", type=str, required=True,
        help="Directory containing {video_id}/{clip_id}.wav files"
    )
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument(
        "--model_name", type=str,
        default="facebook/data2vec-audio-base-960h"
    )
    parser.add_argument("--target_frames", type=int, default=50,
                        help="目标帧数; 0 或负数 = 不池化, 保留原始帧率")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ── 加载 Data2Vec 模型 ──
    print(f"Loading {args.model_name}...")
    from transformers import Data2VecAudioModel, Wav2Vec2FeatureExtractor

    model = Data2VecAudioModel.from_pretrained(args.model_name, local_files_only=True).to(device)
    model.eval()

    # 冻结 CNN feature extractor (对齐 CAGMamba)
    if hasattr(model, 'feature_extractor'):
        for p in model.feature_extractor.parameters():
            p.requires_grad = False

    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=16000,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=True,
    )

    # ── 读取 label.csv ──
    print(f"Reading {args.csv_path}...")
    df = pd.read_csv(args.csv_path)

    # 排除损坏的 WAV 文件 (对齐 CAGMamba)
    invalid_files = [
        '3aIQUQgawaI/12.wav', '94ULum9MYX0/2.wav',
        'mRnEJOLkhp8/24.wav', 'aE-X_QdDaqQ/3.wav',
        '94ULum9MYX0/11.wav', 'mRnEJOLkhp8/26.wav',
    ]
    for f in invalid_files:
        vid = f.split('/')[0]
        cid = f.split('/')[1].split('.')[0]
        df = df[~((df['video_id'] == vid) & (df['clip_id'] == int(cid)))]

    # 按 video_id, clip_id 排序 (对齐 .pkl 中的顺序)
    df = df.sort_values(by=['video_id', 'clip_id']).reset_index(drop=True)
    print(f"Total valid samples: {len(df)}")

    hidden_dim = model.config.hidden_size

    # ── 逐样本提取 ──
    all_features = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting"):
        video_id = str(row['video_id'])
        clip_id = str(row['clip_id'])
        # if len(clip_id) < 4:
        #     clip_id = clip_id.zfill(4)
        wav_path = os.path.join(args.audio_dir, video_id, f"{clip_id}.wav")

        try:
            input_values, attention_mask = load_and_process_audio(
                wav_path, feature_extractor
            )
            feats = extract_data2vec_features(
                model, input_values, attention_mask,
                args.target_frames, device,
            )
            all_features.append(feats)
        except Exception as e:
            # 损坏的 WAV 文件 → 零向量
            print(f"Error processing {wav_path}: {e}")
            n_frames = args.target_frames if args.target_frames > 0 else 1
            all_features.append(
                np.zeros((n_frames, hidden_dim), dtype=np.float32)
            )

    features_array = np.stack(all_features, axis=0)  # (N, T, D)
    print(f"Extracted: {features_array.shape}")

    # 保存为 .pkl (格式: {'train': {'audio': ...}, 'valid': ..., 'test': ...})
    # 按 mode 列拆分
    output_data = {}
    for mode in ['train', 'valid', 'test']:
        mask = df['mode'] == mode
        if mask.any():
            mode_ids = df[mask].apply(
                lambda r: f"{r['video_id']}_{r['clip_id']}", axis=1
            ).tolist()
            output_data[mode] = {
                'audio': features_array[mask.values].astype(np.float32),
                'audio_ids': mode_ids,
            }
            print(f"  {mode}: {output_data[mode]['audio'].shape}")

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'wb') as f:
        pickle.dump(output_data, f)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
