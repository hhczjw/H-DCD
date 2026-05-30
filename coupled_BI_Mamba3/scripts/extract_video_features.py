#!/usr/bin/env python3
"""
统一视频特征提取脚本 (支持 CLIP, VideoMAE, OpenFace, DINOv3)

提取的特征形如: { 'video_id': np.ndarray(N_frames, D_dim) }
最后保存为 .pkl 文件，提供给 Coupled-BI-Mamba3 或其他模型使用。

用法示例:
    # 1. CLIP 提取
    python scripts/extract_video_features.py --extractor clip --video_dir /path/to/videos --output features/vision_clip.pkl

    # 2. VideoMAE 提取
    python scripts/extract_video_features.py --extractor videomae --video_dir /path/to/videos --output features/vision_videomae.pkl

    # 3. OpenFace 提取 (需自行安装 OpenFace 2.0 并提取 csv 到某目录，这里只做读取合并)
    python scripts/extract_video_features.py --extractor openface --video_dir /path/to/openface_csvs --output features/vision_openface.pkl

    # 4. DINOv3 提取 (利用自监督纯视觉骨干)
    python scripts/extract_video_features.py --extractor dinov3 --video_dir /path/to/videos --output features/vision_dinov3.pkl

依赖安装:
    pip install transformers torchvision opencv-python pillow tqdm pandas open-clip-torch
"""

import argparse
import os
import pickle
import numpy as np
import torch
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


def extract_video_frames(video_path: str, num_frames: int = 50, rgb: bool = True) -> list:
    """均匀采样视频帧，返回 PIL Image 列表"""
    assert cv2 is not None, "Please install opencv-python: pip install opencv-python"
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames <= 0:
        cap.release()
        return [Image.new('RGB', (224, 224))] * num_frames

    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = []
    
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            if rgb:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))
    
    cap.release()

    # 补充至指定帧数
    while len(frames) < num_frames:
        frames.append(frames[-1] if frames else Image.new('RGB', (224, 224)))
        
    return frames[:num_frames]


# ==========================================
# 1. 方案A: CLIP 提取器
# ==========================================
def run_clip(args, device):
    print(">>> 正在初始化 CLIP 提取器...")
    import open_clip
    import torchvision.transforms as T
    
    # 默认使用 ViT-B-32
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
    
    video_files = []
    for root, dirs, files in os.walk(args.video_dir):
        for f in files:
            if f.endswith(('.mp4', '.avi', '.mkv')):
                video_files.append(os.path.join(root, f))
    
    with torch.no_grad():
        for vid_path in tqdm(video_files, desc="Extracting CLIP"):
            parent_dir = os.path.basename(os.path.dirname(vid_path))
            file_stem = os.path.splitext(os.path.basename(vid_path))[0]
            if parent_dir == os.path.basename(os.path.normpath(args.video_dir)):
                vid_id = file_stem
            else:
                vid_id = f"{parent_dir}_{file_stem}"
            frames = extract_video_frames(vid_path, args.num_frames)
            
            # (num_frames, C, H, W)
            tensors = torch.stack([transform(img) for img in frames]).to(device)
            
            # Batch extraction to avoid OOM
            feats = []
            for i in range(0, args.num_frames, args.batch_size):
                batch_tensors = tensors[i:i+args.batch_size]
                feat = model(batch_tensors)
                feats.append(feat.cpu())
            
            final_feat = torch.cat(feats, dim=0).numpy() # (50, 768)
            results[vid_id] = final_feat
            
    return results

# ==========================================
# 2. 方案B: VideoMAE 提取器
# ==========================================
def run_videomae(args, device):
    """
    VideoMAE 是专为视频训练的时空 Transformer。
    它默认吸收 16 帧作为一个输入块 (clip)。
    如果我们希望获得 50 帧的逐帧对应或者更压缩的特征，需要分治或池化。
    这里展示直接将多帧传给模型获得 sequence feature 的基础思路。
    """
    print(">>> 正在初始化 VideoMAE 提取器...")
    from transformers import VideoMAEImageProcessor, VideoMAEModel
    
    # model_name = "MCG-NJU/videomae-base"
    processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base")
    model = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base").to(device)
    model.eval()

    results = {}
    video_files = []
    for root, dirs, files in os.walk(args.video_dir):
        for f in files:
            if f.endswith(('.mp4', '.avi', '.mkv')):
                video_files.append(os.path.join(root, f))

    with torch.no_grad():
        for vid_path in tqdm(video_files, desc="Extracting VideoMAE"):
            parent_dir = os.path.basename(os.path.dirname(vid_path))
            file_stem = os.path.splitext(os.path.basename(vid_path))[0]
            if parent_dir == os.path.basename(os.path.normpath(args.video_dir)):
                vid_id = file_stem
            else:
                vid_id = f"{parent_dir}_{file_stem}"
            # 【重要改进】: 针对 VideoMAE 进行“滑窗切块提取”
            # 为保留局部的微表情动态和全局的时间跨度，我们在全视频生命周期中取 `args.num_frames` (50) 个观测点
            # 并在每个观测点局域提取连续的 16 帧（满足预训练位置编码），再分别单独提取特征进行降维 pooling
            cap = cv2.VideoCapture(vid_path)
            all_frames = []
            while True:
                ret, frame = cap.read()
                if not ret: break
                frame = cv2.resize(frame, (args.image_size, args.image_size))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                all_frames.append(Image.fromarray(frame))
            cap.release()
            
            if not all_frames:
                all_frames = [Image.new('RGB', (args.image_size, args.image_size))]
                
            total = len(all_frames)
            # 全局取 50 个关键观测点
            centers = np.linspace(0, total - 1, args.num_frames, dtype=int)
            
            clips = []
            for c in centers:
                start = max(0, c - 8)
                end = start + 8
                if end > total:
                    end = total
                    start = max(0, end - 16)
                clip = all_frames[start:end]
                # 若视频本身都不足16帧则用最后一帧补齐
                while len(clip) < 16:
                    clip.append(clip[-1] if clip else all_frames[0])
                clips.append([np.array(img) for img in clip])
                
            # 分批将 50 个(每个包含16连续帧)送入网络
            feats = []
            for i in range(0, args.num_frames, args.batch_size):
                batch_clips = clips[i:i+args.batch_size]
                inputs = processor(batch_clips, return_tensors="pt").to(device)
                outputs = model(**inputs)
                
                # outputs.last_hidden_state -> (batch, 1568, 768) -> pooling -> (batch, 768)
                pool_feat = outputs.last_hidden_state.mean(dim=1)
                feats.append(pool_feat.cpu())
                
            seq_feat = torch.cat(feats, dim=0).numpy() # (50, 768)
            results[vid_id] = seq_feat
            
    return results

# ==========================================
# 3. 方案C: OpenFace 3.0 提取器
# ==========================================
def run_openface(args, device):
    """
    使用 OpenFace 3.0 直接从视频中端到端提取面部行为特征。
    
    ★ 与 OpenFace 2.0 的重要区别:
       OpenFace 2.0 是一个 C++ 工具, 需要手动编译安装, 并在终端中
       对每个视频运行 ./FeatureExtraction 命令来生成 CSV 文件,
       我们的脚本只能被动"读取合并"那些已经产生的 CSV。
       
       OpenFace 3.0 (pip install openface-test) 是纯 Python 包,
       无需编译, 通过原生 API 即可直接处理视频帧, 端到端提取特征。
       
    提取内容 (18维): 
       - Action Units (8) : AU1, AU2, AU4, AU6, AU9, AU12, AU25, AU26
       - Gaze (2)         : yaw (水平), pitch (垂直)
       - Emotion (8)      : neutral, happy, sad, surprise, fear, disgust, anger, contempt
    输出: (num_frames, 18)  ∈ float32
    
    ★ 实现细节:
       为了避免 OpenFace 3.0 的 FaceDetector.get_face() 必须读取磁盘文件
       这一低效设计, 本函数直接调用底层的 RetinaFace 模型在内存中处理帧,
       绕过临时文件读写, 大幅提升提取速度。
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
            if p: print(f"  - {p}")
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
        # 边界保护
        h, w = img_raw.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        return img_raw[y1:y2, x1:x2]
    # ----------------------------------------------------------------
    
    results = {}
    video_files = []
    for root, dirs, files in _os.walk(args.video_dir):
        for f in files:
            if f.endswith(('.mp4', '.avi', '.mkv')):
                video_files.append(_os.path.join(root, f))
    
    with torch.no_grad():
        for vid_path in tqdm(video_files, desc="Extracting OpenFace 3.0"):
            parent_dir = _os.path.basename(_os.path.dirname(vid_path))
            file_stem = _os.path.splitext(_os.path.basename(vid_path))[0]
            if parent_dir == _os.path.basename(_os.path.normpath(args.video_dir)):
                vid_id = file_stem
            else:
                vid_id = f"{parent_dir}_{file_stem}"
            
            cap = cv2.VideoCapture(vid_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                cap.release()
                n_fallback = args.num_frames if args.num_frames > 0 else 50
                results[vid_id] = np.zeros((n_fallback, 18), dtype=np.float32)
                continue
            
            indices = np.linspace(0, total_frames - 1, 
                                   args.num_frames if args.num_frames > 0 else total_frames, 
                                   dtype=int)
            frame_feats = []
            
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret:
                    frame_feats.append(np.zeros(18, dtype=np.float32))
                    continue
                
                # ---- 全内存流水线: 不再写临时文件 ----
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
            
            n_out = len(indices)  # 实际输出帧数
            while len(frame_feats) < n_out:
                frame_feats.append(frame_feats[-1] if frame_feats else np.zeros(18, dtype=np.float32))
            
            results[vid_id] = np.stack(frame_feats[:n_out]).astype(np.float32)
    
    return results

# ==========================================
# 4. 方案D: DINOv3 (新一代纯视觉提取器)
# ==========================================
def run_dinov3(args, device):
    """
    使用由 Meta 发布的 DINOv3 视觉大模型。
    优先级: ModelScope 本地缓存 > ModelScope 自动下载 > HuggingFace Hub
    
    DINOv3 在 HuggingFace 上是 gated model (需 Meta 审批),
    但 ModelScope (魔搭社区) 提供了国内可直接下载的镜像, 无需申请。
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
            print(">>> 回退到 HuggingFace Hub (需先接受许可: https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m)")
            model_path = model_name
    
    processor = AutoImageProcessor.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path).to(device)
    model.eval()
    
    results = {}
    
    # 递归查找视频文件以适应多级目录 (如 CMU-MOSI/Raw/视频ID/切割序列.mp4)
    video_files = []
    for root, dirs, files in os.walk(args.video_dir):
        for f in files:
            if f.endswith(('.mp4', '.avi', '.mkv')):
                video_files.append(os.path.join(root, f))
    
    with torch.no_grad():
        for vid_path in tqdm(video_files, desc="Extracting DINOv3"):
            # 兼容 MOSI 的 video_id_段号 结构命名方式
            # 假设目录名为视频ID (如 '03bSnISJMiM'), 文件名为片段 (如 '1.mp4')
            parent_dir = os.path.basename(os.path.dirname(vid_path))
            file_stem = os.path.splitext(os.path.basename(vid_path))[0]
            # 如果源文件本身就是完整的视频ID如 mosi_video_1.mp4 也可以兼容
            if parent_dir == os.path.basename(os.path.normpath(args.video_dir)):
                vid_id = file_stem # 扁平结构
            else:
                vid_id = f"{parent_dir}_{file_stem}" # 多级目录结构

            frames = extract_video_frames(vid_path, args.num_frames)
            
            feats = []
            for i in range(0, args.num_frames, args.batch_size):
                batch_frames = frames[i:i+args.batch_size]
                inputs = processor(images=batch_frames, return_tensors="pt").to(device)
                
                outputs = model(**inputs)
                # 取 CLS token 的特征作为该帧的全局表示 -> (batch, hidden_dim) 比如768
                pool_feat = outputs.pooler_output
                feats.append(pool_feat.cpu())
                
            final_feat = torch.cat(feats, dim=0).numpy() # (50, 768)
            results[vid_id] = final_feat
            
    return results


def main():
    parser = argparse.ArgumentParser(description="多模态-视频视觉特征提取脚本")
    parser.add_argument("--extractor", type=str, required=True, 
                        choices=["clip", "videomae", "openface", "dinov3"],
                        help="选择特征提取器")
    parser.add_argument("--video_dir", type=str, required=True, help="视频或 CSV 数据目录")
    parser.add_argument("--output", type=str, required=True, help="保存的 pkl 路径")
    parser.add_argument("--num_frames", type=int, default=50, help="提取的目标帧数或时序长度")
    parser.add_argument("--batch_size", type=int, default=16, help="GPU 推理的批量大小")
    parser.add_argument("--image_size", type=int, default=224, help="图像重建尺寸")
    parser.add_argument("--device", type=str, default="cuda", help="推理设备")
    parser.add_argument("--openface_weights_dir", type=str, default=None,
                        help="OpenFace 3.0 权重文件所在目录 (默认自动查找)")
    
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    
    if args.extractor == "clip":
        results = run_clip(args, device)
    elif args.extractor == "videomae":
        results = run_videomae(args, device)
    elif args.extractor == "openface":
        results = run_openface(args, device)
    elif args.extractor == "dinov3":
        results = run_dinov3(args, device)
    else:
        raise ValueError(f"Unknown extractor: {args.extractor}")
        
    print(f">>> 提取完成! 成功提取了 {len(results)} 个视频的特征。")
    print(f">>> 正在保存至 {args.output} ...")
    
    with open(args.output, "wb") as f:
        pickle.dump(results, f)
        
    print(">>> 保存完毕!")


if __name__ == "__main__":
    main()
