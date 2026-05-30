#!/usr/bin/env python3
"""
统一视频特征提取脚本 — 保留原始帧数版
=====================================

在 extract_video_features.py 基础上新增:
  - `--num_frames 0` 时保留视频的全部原始帧, 不做均匀采样或池化
  - 支持 `--csv_path` 按 train/valid/test 拆分保存 (对齐 extract_audio_data2vec.py 管线)
  - 输出格式与 data_loader.py 的 _load_external_feat 兼容

支持提取器: openface / clip / videomae / dinov3

用法示例:
    # 1. OpenFace 3.0, 保留原始帧数, 按 label.csv 拆分
    python scripts/extract_video_features_full.py \
        --extractor openface \
        --video_dir /path/to/videos \
        --csv_path /path/to/label.csv \
        --output ./features/vision_openface3_full.pkl \
        --num_frames 0

    # 2. CLIP, 保留原始帧数
    python scripts/extract_video_features_full.py \
        --extractor clip \
        --video_dir /path/to/videos \
        --csv_path /path/to/label.csv \
        --output ./features/vision_clip_full.pkl \
        --num_frames 0

    # 3. VideoMAE, 保留原始帧数
    python scripts/extract_video_features_full.py \
        --extractor videomae \
        --video_dir /path/to/videos \
        --csv_path /path/to/label.csv \
        --output ./features/vision_videomae_full.pkl \
        --num_frames 0

    # 4. DINOv3, 保留原始帧数
    python scripts/extract_video_features_full.py \
        --extractor dinov3 \
        --video_dir /path/to/videos \
        --csv_path /path/to/label.csv \
        --output ./features/vision_dinov3_full.pkl \
        --num_frames 0

    # 5. 固定 50 帧 (与原脚本行为一致)
    python scripts/extract_video_features_full.py \
        --extractor clip \
        --video_dir /path/to/videos \
        --output ./features/vision_clip.pkl \
        --num_frames 50

依赖:
    pip install transformers torchvision opencv-python pillow tqdm pandas open-clip-torch
"""

import argparse
import os
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import pandas as pd
except ImportError:
    pd = None


# ===========================================================================
# 帧提取工具函数
# ===========================================================================

def extract_video_frames(video_path: str, num_frames: int = 50, rgb: bool = True) -> list:
    """
    从视频中提取帧, 返回 PIL Image 列表.

    - num_frames > 0: 均匀采样 num_frames 帧
    - num_frames <= 0: 保留全部原始帧
    """
    assert cv2 is not None, "Please install opencv-python: pip install opencv-python"

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        n = num_frames if num_frames > 0 else 1
        return [Image.new('RGB', (224, 224))] * n

    if num_frames > 0:
        # 均匀采样
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    else:
        # 保留全部帧
        indices = np.arange(total_frames)

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            if rgb:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))

    cap.release()

    # 补充至目标帧数 (仅当 num_frames > 0 时)
    if num_frames > 0:
        while len(frames) < num_frames:
            frames.append(frames[-1] if frames else Image.new('RGB', (224, 224)))
        frames = frames[:num_frames]

    return frames


def get_video_total_frames(video_path: str) -> int:
    """获取视频的总帧数"""
    assert cv2 is not None, "Please install opencv-python"
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total


def get_video_id(video_path: str, video_dir: str) -> str:
    """从视频路径推导 video_id, 兼容扁平和多级目录结构"""
    parent_dir = os.path.basename(os.path.dirname(video_path))
    file_stem = os.path.splitext(os.path.basename(video_path))[0]
    if parent_dir == os.path.basename(os.path.normpath(video_dir)):
        return file_stem  # 扁平结构
    else:
        return f"{parent_dir}_{file_stem}"  # 多级目录结构


def collect_video_files(video_dir: str) -> list:
    """递归收集所有视频文件"""
    video_files = []
    for root, dirs, files in os.walk(video_dir):
        for f in files:
            if f.endswith(('.mp4', '.avi', '.mkv')):
                video_files.append(os.path.join(root, f))
    return sorted(video_files)


# ===========================================================================
# 1. CLIP 提取器
# ===========================================================================
def run_clip(args, device):
    print(">>> 正在初始化 CLIP 提取器...")
    import open_clip
    import torchvision.transforms as T

    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    model = model.visual.to(device)
    model.eval()

    transform = T.Compose([
        T.Resize(args.image_size, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(args.image_size),
        T.ToTensor(),
        T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

    results = {}
    video_files = collect_video_files(args.video_dir)

    with torch.no_grad():
        for vid_path in tqdm(video_files, desc="Extracting CLIP"):
            vid_id = get_video_id(vid_path, args.video_dir)
            frames = extract_video_frames(vid_path, args.num_frames)

            # (N_frames, C, H, W)
            tensors = torch.stack([transform(img) for img in frames]).to(device)

            # 分批提取避免 OOM
            feats = []
            for i in range(0, len(tensors), args.batch_size):
                batch_tensors = tensors[i:i + args.batch_size]
                feat = model(batch_tensors)
                feats.append(feat.cpu())

            final_feat = torch.cat(feats, dim=0).numpy()  # (N_frames, 768)
            results[vid_id] = final_feat

    return results


# ===========================================================================
# 2. VideoMAE 提取器
# ===========================================================================
def run_videomae(args, device):
    """
    VideoMAE 滑窗提取.

    当 num_frames > 0: 取 num_frames 个关键观测点, 每个用 16 连续帧
    当 num_frames <= 0: 对全部帧做滑窗, 保留原始帧数
    """
    print(">>> 正在初始化 VideoMAE 提取器...")
    from transformers import VideoMAEImageProcessor, VideoMAEModel

    processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base")
    model = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base").to(device)
    model.eval()

    results = {}
    video_files = collect_video_files(args.video_dir)

    clip_len = 16  # VideoMAE 预训练位置编码长度

    with torch.no_grad():
        for vid_path in tqdm(video_files, desc="Extracting VideoMAE"):
            vid_id = get_video_id(vid_path, args.video_dir)

            # 读取全部帧
            cap = cv2.VideoCapture(vid_path)
            all_frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.resize(frame, (args.image_size, args.image_size))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                all_frames.append(Image.fromarray(frame))
            cap.release()

            if not all_frames:
                all_frames = [Image.new('RGB', (args.image_size, args.image_size))]

            total = len(all_frames)

            if args.num_frames > 0:
                # 固定 num_frames 个关键观测点
                centers = np.linspace(0, total - 1, args.num_frames, dtype=int)
            else:
                # 保留原始帧数: 每帧作为一个观测点
                centers = np.arange(total)

            clips = []
            for c in centers:
                start = max(0, c - 8)
                end = start + clip_len
                if end > total:
                    end = total
                    start = max(0, end - clip_len)
                clip = all_frames[start:end]
                # 补齐 16 帧
                while len(clip) < clip_len:
                    clip.append(clip[-1] if clip else all_frames[0])
                clips.append([np.array(img) for img in clip])

            # 分批送入网络
            feats = []
            for i in range(0, len(clips), args.batch_size):
                batch_clips = clips[i:i + args.batch_size]
                inputs = processor(batch_clips, return_tensors="pt").to(device)
                outputs = model(**inputs)
                # mean pooling over temporal dimension -> (batch, 768)
                pool_feat = outputs.last_hidden_state.mean(dim=1)
                feats.append(pool_feat.cpu())

            seq_feat = torch.cat(feats, dim=0).numpy()  # (N_frames, 768)
            results[vid_id] = seq_feat

    return results


# ===========================================================================
# 3. OpenFace 3.0 提取器
# ===========================================================================
def run_openface(args, device):
    """
    OpenFace 3.0 端到端提取面部行为特征.

    输出 18 维/帧: AU(8) + Gaze(2) + Emotion(8)
    当 num_frames > 0: 均匀采样 num_frames 帧
    当 num_frames <= 0: 处理全部原始帧
    """
    print(">>> 正在初始化 OpenFace 3.0 提取器...")
    import os as _os

    try:
        from openface.face_detection import FaceDetector
        from openface.multitask_model import MultitaskPredictor
        from openface.Pytorch_Retinaface.layers.functions.prior_box import PriorBox
        from openface.Pytorch_Retinaface.utils.box_utils import decode, decode_landm
        from openface.Pytorch_Retinaface.utils.nms.py_cpu_nms import py_cpu_nms
    except ImportError:
        print("=" * 60)
        print("OpenFace 3.0 未安装。请执行以下命令安装:")
        print("  pip install openface-test")
        print("  openface download   # 下载预训练权重 (约 500MB)")
        print("=" * 60)
        raise

    # ---- 模型权重路径查找 ----
    script_dir = _os.path.dirname(_os.path.abspath(__file__))
    project_root = _os.path.dirname(script_dir)
    candidate_paths = [
        args.openface_weights_dir if hasattr(args, 'openface_weights_dir') and args.openface_weights_dir else "",
        _os.path.join(project_root, "weights"),
        _os.path.join(_os.getcwd(), "weights"),
        "./weights",
    ]

    weights_dir = None
    for p in candidate_paths:
        if p and _os.path.isdir(p) and _os.path.isfile(_os.path.join(p, "Alignment_RetinaFace.pth")):
            weights_dir = p
            break
    if weights_dir is None:
        for p in candidate_paths:
            if p and _os.path.isdir(p):
                weights_dir = p
                break
    if weights_dir is None:
        print("=" * 60)
        print("未找到 OpenFace 3.0 权重文件!")
        print("请运行: openface download")
        for p in candidate_paths:
            if p:
                print(f"  - {p}")
        print("=" * 60)
        raise FileNotFoundError("OpenFace 3.0 weights not found")

    face_model_path = _os.path.join(weights_dir, "Alignment_RetinaFace.pth")
    multitask_model_path = _os.path.join(weights_dir, "MTL_backbone.pth")

    print(f">>> 使用权重: {weights_dir}")
    face_detector = FaceDetector(model_path=face_model_path, device=device)
    multitask_predictor = MultitaskPredictor(model_path=multitask_model_path, device=device)
    print(">>> OpenFace 3.0 模型加载完成!")

    # ---------- 辅助函数: 用 RetinaFace 在内存中检测人脸 ----------
    def _detect_face_in_memory(frame_bgr: np.ndarray, resize: float = 1.0):
        """不写磁盘, 直接在内存中对 numpy 帧做 RetinaFace 人脸检测"""
        cfg = face_detector.cfg
        img_raw = frame_bgr.copy()
        img = np.float32(img_raw)
        if resize != 1.0:
            img = cv2.resize(img, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img_tensor = torch.from_numpy(img).unsqueeze(0).to(device)

        with torch.no_grad():
            loc, conf, landms = face_detector.model(img_tensor)

        im_height, im_width, _ = img_raw.shape
        scale = torch.Tensor([img_tensor.shape[3], img_tensor.shape[2],
                              img_tensor.shape[3], img_tensor.shape[2]]).to(device)

        priorbox = PriorBox(cfg, image_size=(im_height, im_width))
        priors = priorbox.forward().to(device)
        prior_data = priors.data

        boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms_d = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
        scale1 = torch.Tensor([img_tensor.shape[3], img_tensor.shape[2]] * 5).to(device)
        landms_d = landms_d * scale1 / resize
        landms_d = landms_d.cpu().numpy()

        inds = np.where(scores > face_detector.confidence_threshold)[0]
        boxes, landms_d, scores = boxes[inds], landms_d[inds], scores[inds]

        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, face_detector.nms_threshold)
        dets = dets[keep]
        dets = np.concatenate((dets, landms_d[keep]), axis=1)
        return dets, img_raw

    def _crop_face_from_dets(dets, img_raw):
        """从检测结果中裁剪最高置信度的人脸区域"""
        if dets is None or len(dets) == 0:
            return None
        det = dets[0]
        if det[4] < face_detector.vis_threshold:
            return None
        x1, y1, x2, y2 = det[:4].astype(int)
        h, w = img_raw.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        return img_raw[y1:y2, x1:x2]

    # ----------------------------------------------------------------

    results = {}
    video_files = collect_video_files(args.video_dir)

    with torch.no_grad():
        for vid_path in tqdm(video_files, desc="Extracting OpenFace 3.0"):
            vid_id = get_video_id(vid_path, args.video_dir)

            cap = cv2.VideoCapture(vid_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                cap.release()
                n_fallback = args.num_frames if args.num_frames > 0 else 50
                results[vid_id] = np.zeros((n_fallback, 18), dtype=np.float32)
                continue

            if args.num_frames > 0:
                indices = np.linspace(0, total_frames - 1, args.num_frames, dtype=int)
            else:
                indices = np.arange(total_frames)

            frame_feats = []
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret:
                    frame_feats.append(np.zeros(18, dtype=np.float32))
                    continue

                try:
                    dets, img_raw = _detect_face_in_memory(frame)
                    cropped_face = _crop_face_from_dets(dets, img_raw)

                    if cropped_face is not None and cropped_face.shape[0] > 0 and cropped_face.shape[1] > 0:
                        emotion_logits, gaze_output, au_output = multitask_predictor.predict(cropped_face)
                        feat = np.concatenate([
                            au_output.cpu().numpy().flatten(),
                            gaze_output.cpu().numpy().flatten(),
                            emotion_logits.cpu().numpy().flatten(),
                        ]).astype(np.float32)
                    else:
                        feat = frame_feats[-1] if frame_feats else np.zeros(18, dtype=np.float32)
                except Exception as e:
                    print(f"  [WARNING] {vid_id} 帧 {idx} 提取失败: {e}")
                    feat = frame_feats[-1] if frame_feats else np.zeros(18, dtype=np.float32)

                frame_feats.append(feat)

            cap.release()

            # 确保输出帧数一致
            n_out = len(indices)
            while len(frame_feats) < n_out:
                frame_feats.append(frame_feats[-1] if frame_feats else np.zeros(18, dtype=np.float32))

            results[vid_id] = np.stack(frame_feats[:n_out]).astype(np.float32)

    return results


# ===========================================================================
# 4. DINOv3 提取器
# ===========================================================================
def run_dinov3(args, device):
    """
    使用 DINOv3 视觉大模型提取特征.
    优先级: ModelScope 本地缓存 > ModelScope 自动下载 > HuggingFace Hub

    当 num_frames > 0: 均匀采样 num_frames 帧
    当 num_frames <= 0: 处理全部原始帧
    """
    print(">>> 正在初始化 DINOv3 提取器...")
    from transformers import AutoImageProcessor, AutoModel

    model_name = "facebook/dinov3-vitb16-pretrain-lvd1689m"
    modelscope_cache = os.path.expanduser(
        "~/.cache/modelscope/hub/models/facebook/dinov3-vitb16-pretrain-lvd1689m"
    )

    # ---- 智能路径选择 ----
    if os.path.isdir(modelscope_cache):
        model_path = modelscope_cache
        print(f">>> 使用 ModelScope 本地缓存: {model_path}")
    else:
        try:
            from modelscope import snapshot_download
            print(">>> 正在从 ModelScope (魔搭社区) 下载 DINOv3 权重 (~327MB)...")
            model_path = snapshot_download('facebook/dinov3-vitb16-pretrain-lvd1689m')
            print(f">>> 下载完成: {model_path}")
        except Exception as e:
            print(f"[WARNING] ModelScope 不可用: {e}")
            print(">>> 回退到 HuggingFace Hub (需先接受许可)")
            model_path = model_name

    processor = AutoImageProcessor.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path).to(device)
    model.eval()

    results = {}
    video_files = collect_video_files(args.video_dir)

    with torch.no_grad():
        for vid_path in tqdm(video_files, desc="Extracting DINOv3"):
            vid_id = get_video_id(vid_path, args.video_dir)
            frames = extract_video_frames(vid_path, args.num_frames)

            feats = []
            for i in range(0, len(frames), args.batch_size):
                batch_frames = frames[i:i + args.batch_size]
                inputs = processor(images=batch_frames, return_tensors="pt").to(device)
                outputs = model(**inputs)
                # CLS token 特征 -> (batch, hidden_dim)
                pool_feat = outputs.pooler_output
                feats.append(pool_feat.cpu())

            final_feat = torch.cat(feats, dim=0).numpy()  # (N_frames, 768)
            results[vid_id] = final_feat

    return results


# ===========================================================================
# 主函数
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(description="多模态视频特征提取 (保留原始帧数版)")
    parser.add_argument("--extractor", type=str, required=True,
                        choices=["clip", "videomae", "openface", "dinov3"],
                        help="选择特征提取器")
    parser.add_argument("--video_dir", type=str, required=True,
                        help="视频文件目录")
    parser.add_argument("--output", type=str, required=True,
                        help="保存的 pkl 路径")
    parser.add_argument("--csv_path", type=str, default=None,
                        help="label.csv 路径 (CAGMamba 格式, 用于 train/valid/test 拆分)")
    parser.add_argument("--num_frames", type=int, default=50,
                        help="目标帧数; 0 或负数 = 不采样, 保留原始帧率")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="GPU 推理批量大小")
    parser.add_argument("--image_size", type=int, default=224,
                        help="图像重建尺寸")
    parser.add_argument("--device", type=str, default="cuda",
                        help="推理设备")
    parser.add_argument("--openface_weights_dir", type=str, default=None,
                        help="OpenFace 3.0 权重目录 (默认自动查找)")

    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # ---- 提取特征 ----
    extractor_map = {
        "clip": run_clip,
        "videomae": run_videomae,
        "openface": run_openface,
        "dinov3": run_dinov3,
    }
    results = extractor_map[args.extractor](args, device)

    print(f">>> 提取完成! 成功提取了 {len(results)} 个视频的特征。")

    # ---- 统计帧数分布 ----
    frame_counts = {vid_id: feat.shape[0] for vid_id, feat in results.items()}
    if frame_counts:
        fc = list(frame_counts.values())
        print(f">>> 帧数统计: min={min(fc)}, max={max(fc)}, mean={np.mean(fc):.1f}, median={np.median(fc):.0f}")

    # ---- 保存 ----
    if args.csv_path and os.path.isfile(args.csv_path):
        # 按 label.csv 的 mode 列拆分保存 (对齐 data_loader 格式)
        df = pd.read_csv(args.csv_path)
        df = df.sort_values(by=['video_id', 'clip_id']).reset_index(drop=True)

        output_data = {}
        feat_key = "vision"
        dim = None

        for mode in ['train', 'valid', 'test']:
            mask = df['mode'] == mode
            if not mask.any():
                continue

            mode_feats = []
            mode_ids = []
            for _, row in df[mask].iterrows():
                video_id = str(row['video_id'])
                clip_id = str(row['clip_id'])
                vid_key_candidates = [
                    f"{video_id}_{clip_id}",
                    video_id,
                ]
                feat = None
                for vk in vid_key_candidates:
                    if vk in results:
                        feat = results[vk]
                        break
                if feat is None:
                    # 找不到对应特征, 用零向量填充
                    if dim is None:
                        dim = 18 if args.extractor == "openface" else 768
                    n_frames = args.num_frames if args.num_frames > 0 else 50
                    feat = np.zeros((n_frames, dim), dtype=np.float32)
                    print(f"  [WARNING] {video_id}_{clip_id} 特征缺失, 用零向量填充")
                else:
                    if dim is None:
                        dim = feat.shape[1]

                mode_feats.append(feat)
                mode_ids.append(f"{video_id}_{clip_id}")

            # 注意: 保留原始帧数时不同样本帧数可能不同, 需要对齐到最长
            # 策略: pad 到该 split 中最长的帧数
            max_frames = max(f.shape[0] for f in mode_feats)
            aligned_feats = []
            for f in mode_feats:
                if f.shape[0] < max_frames:
                    # pad 末尾帧
                    pad_len = max_frames - f.shape[0]
                    pad = np.repeat(f[-1:], pad_len, axis=0)
                    aligned_feats.append(np.concatenate([f, pad], axis=0))
                else:
                    aligned_feats.append(f)

            output_data[mode] = {
                feat_key: np.stack(aligned_feats, axis=0).astype(np.float32),
                f"{feat_key}_ids": mode_ids,
            }
            print(f"  {mode}: {output_data[mode][feat_key].shape}")

        print(f">>> 正在保存至 {args.output} ...")
        with open(args.output, "wb") as f:
            pickle.dump(output_data, f)
    else:
        # 无 label.csv: 保存为扁平格式 (兼容原脚本)
        flat_output = {}
        for vid_id, feat in results.items():
            flat_output[vid_id] = feat

        print(f">>> 正在保存至 {args.output} (扁平格式, {len(flat_output)} 个视频) ...")
        with open(args.output, "wb") as f:
            pickle.dump(flat_output, f)

    print(">>> 保存完毕!")


if __name__ == "__main__":
    main()
