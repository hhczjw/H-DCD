#!/usr/bin/env python3
"""
从 MP4 视频文件中提取音频为 16kHz WAV (复制自 CAGMamba)
=========================================================

用法:
    # MOSI
    python scripts/extract_audio_wav.py \
        --raw_dir /media/zjw/.../CMU-MOSI/Raw \
        --wav_dir /media/zjw/.../CMU-MOSI/wav

    # MOSEI
    python scripts/extract_audio_wav.py \
        --raw_dir /media/zjw/.../CMU-MOSEI/Raw \
        --wav_dir /media/zjw/.../CMU-MOSEI/wav \
        --dataset mosei    # ← MOSEI 需要先修复损坏帧

依赖: pip install moviepy opencv-python tqdm
"""

import os
import argparse
import time
import cv2
from tqdm import tqdm
from moviepy.video.io.VideoFileClip import VideoFileClip


def extract(raw_dir, wav_dir):
    """
    遍历 raw_dir 下的所有 video_id 文件夹,
    对每个 .mp4 提取音轨, 保存为 wav_dir/{video_id}/{clip_id}.wav.
    """
    os.makedirs(wav_dir, exist_ok=True)

    for folder in tqdm(sorted(os.listdir(raw_dir)), desc="Extracting audio"):
        input_folder = os.path.join(raw_dir, folder)
        if not os.path.isdir(input_folder):
            continue

        output_folder = os.path.join(wav_dir, folder)
        os.makedirs(output_folder, exist_ok=True)

        for file_name in sorted(os.listdir(input_folder)):
            # 只处理 .mp4 文件, 跳过 -edited 版本
            if not file_name.endswith(".mp4") or "-edited" in file_name:
                continue

            input_path = os.path.join(input_folder, file_name)
            output_path = os.path.join(
                output_folder, file_name.replace(".mp4", ".wav")
            )

            if os.path.exists(output_path):
                continue

            try:
                video = VideoFileClip(input_path)
                audio = video.audio
                audio.write_audiofile(
                    output_path, fps=16000, codec='pcm_s16le',
                    logger=None,
                )
            except Exception as e:
                print(f"\n[WARN] {input_path}: {e}")
                if "-edited.mp4" in input_path:
                    try:
                        fallback = input_path.replace("-edited.mp4", ".mp4")
                        video = VideoFileClip(fallback)
                        video.audio.write_audiofile(
                            output_path, fps=16000, codec='pcm_s16le',
                            logger=None,
                        )
                    except Exception:
                        print(f"  fallback also failed for {fallback}")


def fix_mosei_video_duration(raw_dir):
    """
    MOSEI 数据集有部分视频末尾帧损坏, 需要先截断到有效帧长度.
    复制自 CAGMamba extract_audio.py::preprocess_video_file.
    """
    invalid_files = [
        '3aIQUQgawaI/12', '94ULum9MYX0/2',
        'mRnEJOLkhp8/24', 'aE-X_QdDaqQ/3',
        '94ULum9MYX0/11', 'mRnEJOLkhp8/26',
    ]

    print("Fixing MOSEI video files...")
    for folder in sorted(os.listdir(raw_dir)):
        folder_path = os.path.join(raw_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        for file_name in sorted(os.listdir(folder_path)):
            fpath = os.path.join(folder_path, file_name)
            if "-edited.mp4" in fpath:
                continue
            if os.path.exists(fpath.replace(".mp4", "-edited.mp4")):
                continue
            if os.path.join(folder, file_name.split(".")[0]) in invalid_files:
                continue

            cap = cv2.VideoCapture(fpath)
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_counter = 0
            for _ in range(n_frames):
                ret, _ = cap.read()
                frame_counter += 1
                if not ret:
                    break
            cap.release()

            if frame_counter < n_frames:
                duration = (frame_counter - 1) / fps
                print(f"  Fixing {fpath}: truncate to {duration:.1f}s")
                with VideoFileClip(fpath) as video:
                    new = video.subclip(0, duration)
                    out = fpath.replace(".mp4", "-edited.mp4")
                    new.write_videofile(
                        out, logger=None,
                    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw_dir", type=str, required=True,
        help="MP4 源文件目录 (如 .../CMU-MOSI/Raw)"
    )
    parser.add_argument(
        "--wav_dir", type=str, required=True,
        help="WAV 输出目录 (如 .../CMU-MOSI/wav)"
    )
    parser.add_argument(
        "--dataset", type=str, default="mosi",
        choices=["mosi", "mosei", "sims"],
        help="数据集名 (mosei 需要先修复损坏帧)"
    )
    args = parser.parse_args()

    # MOSEI 需要先修复视频帧
    if args.dataset.lower() == "mosei":
        fix_mosei_video_duration(args.raw_dir)

    extract(args.raw_dir, args.wav_dir)
    print("Done.")


if __name__ == "__main__":
    main()
