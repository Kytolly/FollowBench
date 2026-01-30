import pyiqa
from PIL import Image
import logging
logger = logging.getLogger(__name__)

import torch
from torchvision import transforms
from torchvision.transforms import functional as TF
from torchvision.models.detection import (
    keypointrcnn_resnet50_fpn, KeypointRCNN_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
)
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
from pytorchvideo.models.resnet import create_resnet
import torch.nn.functional as F
from transformers import (
    AutoImageProcessor, 
    AutoModelForDepthEstimation,
    CLIPProcessor,
    CLIPModel,
    VideoMAEModel
)
from ultralytics import YOLO
from .gpu import clear_gpu_memory

_MODEL_CACHE = {}

def _get_cached_model(key, loader_func, *args, **kwargs):
    if key not in _MODEL_CACHE:
        logger.info(f"Loading model into cache: {key} ...")
        _MODEL_CACHE[key] = loader_func(*args, **kwargs)
    return _MODEL_CACHE[key]

def clear_models_cache():
    clear_gpu_memory(_MODEL_CACHE)
    logger.info("Models cache cleared.")


def load_videomae_model(device):
    """
    加载 VideoMAE 模型用于提取时空特征 (SCCR指标)。
    使用 'MCG-NJU/videomae-base' (或 large)。
    """
    def _loader():
        model = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base").to(device).eval()
        return model
    return _get_cached_model(f"videomae_base_{device}", _loader)

def preprocess_videomae_tensor(video_tensor: torch.Tensor, num_frames=16, target_size=(224, 224)):
    """
    VideoMAE 专用预处理 (纯 Tensor)。
    VideoMAE 需要 [B, T, C, H, W]，通常 T=16。
    
    Args:
        video_tensor: [T_in, C, H, W] in [0, 1]
    Returns:
        input_tensor: [1, 16, 3, 224, 224] normalized
    """
    T_in, C, H, W = video_tensor.shape
    
    # 1. 时序采样 (Uniform Sampling) -> 16帧
    if T_in == num_frames:
        indices = torch.arange(T_in, device=video_tensor.device)
    else:
        # linspace 采样
        indices = torch.linspace(0, T_in - 1, num_frames, device=video_tensor.device).long()
    
    video_sampled = video_tensor[indices] # [16, C, H, W]
    
    # 2. 空间 Resize & CenterCrop (到 224x224)
    # 简单策略：直接 Resize 到目标尺寸 (或先 Resize 到 256 再 Crop，这里直接 Resize 效率更高)
    video_resized = torch.nn.functional.interpolate(
        video_sampled, 
        size=target_size, 
        mode='bicubic', 
        align_corners=False, 
        antialias=True
    )
    
    # 3. Normalize (ImageNet Mean/Std)
    mean = torch.tensor([0.485, 0.456, 0.406], device=video_tensor.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=video_tensor.device).view(1, 3, 1, 1)
    
    video_norm = (video_resized - mean) / std
    
    # 4. 增加 Batch 维度: [1, T, C, H, W]
    return video_norm.unsqueeze(0)

def extract_videomae_features(video_tensor: torch.Tensor, model: VideoMAEModel):
    """
    模仿 extract_i3d_features 的风格，针对 VideoMAE 的纯 Tensor 特征提取。
    
    Args:
        video_tensor: [T, C, H, W] in [0, 1]
        model: VideoMAEModel
    Returns:
        feature: [768] tensor (CPU or GPU based on config, recommend CPU for caching)
    """
    device = video_tensor.device
    T, C, H, W = video_tensor.shape
    
    # 1. 时序采样 (VideoMAE 需要固定的帧数，通常是 16)
    num_frames = 16
    if T == num_frames:
        indices = torch.arange(T, device=device)
    else:
        indices = torch.linspace(0, T - 1, num_frames, device=device).long()
    
    video = video_tensor[indices] # [16, C, H, W]
    
    # 2. 空间预处理 (Resize 224x224 + Normalize)
    # VideoMAE 使用 ImageNet Mean/Std
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    
    # Resize
    video = F.interpolate(video, size=(224, 224), mode='bicubic', align_corners=False, antialias=True)
    
    # Normalize
    video = (video - mean) / std
    
    # [16, 3, 224, 224] -> [1, 16, 3, 224, 224] (Batch dimension)
    inputs = video.unsqueeze(0)
    
    # 3. 推理
    with torch.no_grad():
        outputs = model(pixel_values=inputs)
        # last_hidden_state: [1, 1568, 768] (patches + cls)
        # Mean Pooling over tokens to get video representation
        feat = outputs.last_hidden_state.mean(dim=1).squeeze(0) # [768]
        
    return feat

def extract_videomae_sequence(video_tensor: torch.Tensor, model: VideoMAEModel):
    """
    提取视频的时序特征序列 (Sequence of Features).
    
    Args:
        video_tensor: [T_in, C, H, W] 原始视频张量
        model: VideoMAEModel
    
    Returns:
        seq_feats: [T_out, D]  (T_out通常为8, D为768)
        代表视频在 8 个时间步上的语义特征。
    """
    device = video_tensor.device
    T_in, C, H, W = video_tensor.shape
    
    # 1. 采样 16 帧 (VideoMAE 标准输入)
    num_frames = 16
    if T_in == num_frames:
        indices = torch.arange(T_in, device=device)
    else:
        indices = torch.linspace(0, T_in - 1, num_frames, device=device).long()
    
    video = video_tensor[indices] # [16, C, H, W]
    
    # 2. 预处理 (Resize 224x224 + Normalize)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    
    video = F.interpolate(video, size=(224, 224), mode='bicubic', align_corners=False, antialias=True)
    video = (video - mean) / std
    
    # Add Batch Dim: [1, 16, 3, 224, 224]
    inputs = video.unsqueeze(0)
    
    # 3. 推理
    with torch.no_grad():
        outputs = model(pixel_values=inputs)
        # last_hidden_state: [1, 1568, 768]
        # Token Layout: (Time // 2) * (H // 16) * (W // 16)
        # 对于 16x224x224 -> Time=8, H=14, W=14 -> 8*14*14 = 1568
        tokens = outputs.last_hidden_state.squeeze(0) # [1568, 768]
        
        # 4. 还原时空结构并池化
        # Reshape to [T', H', W', D]
        # VideoMAE tokens 顺序通常是 T 优先
        tokens = tokens.view(8, 14, 14, 768)
        
        # Spatial Mean Pooling: [8, 14, 14, 768] -> [8, 768]
        seq_feats = tokens.mean(dim=(1, 2))
        
    return seq_feats
   
def load_yolov8(device, model_path="yolov8x.pt"):
    '''
    load_yolov8 的 Docstring
    
    :param device: 说明
    '''
    def _loader():
        model = YOLO(model_path)
        return model
    return _get_cached_model(f"yolov8_{device}", _loader)

def clip_preprocess_tensor(images: torch.Tensor, size=(224, 224)):
    """
    CLIP 的全 GPU 预处理 (仅 Normalize)。
    假设输入已经是 224x224 的 Tensor。
    """
    if images.ndim == 3:
        images = images.unsqueeze(0)
    
    # 安全检查：如果尺寸不对才 Resize (通常不会触发)
    if images.shape[-2:] != size:
        # 如果必须 resize，这里用 squash 兜底，但 ac.py 应该已经保证了尺寸
        images = F.interpolate(images, size=size, mode='bicubic', align_corners=False, antialias=True)
    
    # Normalize (CLIP Mean/Std)
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=images.device).view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=images.device).view(1, 3, 1, 1)
    return (images - mean) / std

def get_crop_embedding(
    image_tensor: torch.Tensor, 
    bbox: torch.Tensor, 
    model: CLIPModel, 
    processor: CLIPProcessor, 
    device: str
):
    """
    工具函数：裁剪图像区域并提取 CLIP 特征。
    
    Args:
        image_tensor: [C, H, W] 单帧图像
        bbox: [x1, y1, x2, y2]
        model: CLIPModel
        processor: CLIPProcessor
        device: 计算设备
        
    Returns:
        normalized_embedding: [1, D] Tensor (on device) 或 None (若裁剪无效)
    """
    if bbox is None:
        return None
        
    x1, y1, x2, y2 = map(int, bbox.tolist())
    H, W = image_tensor.shape[1], image_tensor.shape[2]
    
    # 边界保护
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)
    
    if x2 <= x1 or y2 <= y1:
        return None
    
    # Crop
    crop = image_tensor[:, y1:y2, x1:x2] # [C, h, w]
    
    try:
        # Preprocess: Tensor -> PIL -> Processor -> Tensor
        # CLIPProcessor 需要 PIL (或者 List[PIL])
        pil_img = TF.to_pil_image(crop.cpu())
        inputs = processor(images=pil_img, return_tensors="pt").to(device)
        
        with torch.no_grad():
            features = model.get_image_features(**inputs) # [1, D]
            # 立即归一化，方便后续计算 Cosine Similarity
            return F.normalize(features, p=2, dim=-1)
    except Exception as e:
        logger.warning(f"CLIP embedding extraction failed: {e}")
        return None

def get_all_yolo_detections(video_tensor: torch.Tensor, detector: YOLO):
    """
    对视频帧进行检测，返回每一帧中检测到的【所有】人物框。
    
    Args:
        video_tensor: [T, C, H, W] 归一化后的 Tensor (支持 [0,1] 或 [-1,1])
        detector: YOLO 模型实例
        
    Returns:
        List[List[Tuple[tensor_bbox, float_conf]]]: 
            外层 List 长度为 T (帧数)。
            内层 List 包含该帧所有检测结果，每个结果为 (bbox, conf)。
            bbox 为 [x1, y1, x2, y2] (CPU Tensor)。
    """
    results = []
    batch_size = 16 
    
    # 确保是 [0, 1] 用于转 uint8
    if video_tensor.min() < 0:
        video_tensor = (video_tensor + 1.0) / 2.0
    video_tensor = video_tensor.clamp(0, 1)

    # Tensor -> List[numpy uint8] (Ultralytics 推理最快的方式)
    imgs_np = video_tensor.permute(0, 2, 3, 1).mul(255).byte().cpu().numpy()
    imgs_list = [img for img in imgs_np]

    for i in range(0, len(imgs_list), batch_size):
        batch = imgs_list[i : i + batch_size]
        # classes=0 仅检测人
        preds = detector(batch, verbose=False, classes=0)
        
        for r in preds:
            frame_detections = []
            if len(r.boxes) > 0:
                # 遍历该帧所有检测框
                # r.boxes 是一个 Boxes 对象，包含 xyxy, conf, cls
                boxes = r.boxes.xyxy.cpu()
                confs = r.boxes.conf.cpu()
                
                for k in range(len(boxes)):
                    frame_detections.append((boxes[k], confs[k].item()))
            
            results.append(frame_detections)
                
    return results

def get_yolo_detection_results(
    video_tensor: torch.Tensor, 
    detector: YOLO
):
    """
    使用 YOLO 对视频进行批量检测。
    
    Args:
        video_tensor: [T, C, H, W] 归一化后的 Tensor
        detector: 加载好的 YOLO 模型
        
    Returns:
        results: 长度为 T 的列表。
                 每项为 (bbox, score) 或 None。
                 bbox 为 [x1, y1, x2, y2] (CPU Tensor)
    """
    results = []
    # YOLO 推理非常快，可以使用较大的 batch_size
    batch_size = 16 
    
    # 1. 数据转换: Tensor -> List[numpy array]
    # 假设输入在 [-1, 1] 或 [0, 1]
    if video_tensor.min() < 0:
        video_tensor = (video_tensor + 1.0) / 2.0
    video_tensor = video_tensor.clamp(0, 1)
    
    # [T, C, H, W] -> [T, H, W, C] -> uint8 numpy
    imgs_np = video_tensor.permute(0, 2, 3, 1).mul(255).byte().cpu().numpy()
    
    # Ultralytics 支持直接传入 List[np.ndarray]
    imgs_list = [img for img in imgs_np]

    # 2. 批量推理
    # stream=True 节省内存; classes=0 仅检测“人”
    # verbose=False 关闭刷屏日志
    for i in range(0, len(imgs_list), batch_size):
        batch = imgs_list[i : i + batch_size]
        preds = detector(batch, verbose=False, classes=0)
        
        for r in preds:
            # 检查是否检测到人
            if len(r.boxes) > 0:
                # 默认取置信度最高的一个
                # boxes 默认按 conf 排序，取第一个即可
                box = r.boxes[0].xyxy[0].cpu() # [x1, y1, x2, y2]
                conf = r.boxes[0].conf[0].item()
                results.append((box, conf))
            else:
                results.append(None)
    return results

def load_dinov2(device):
    def _loader():
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').eval().to(device)
        transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return model, transform
    
    return _get_cached_model(f"dinov2_{device}", _loader)

def load_raft(device):
    def _loader():
        model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False).to(device)
        model.eval()
        return model
    return _get_cached_model(f"raft_{device}", _loader)

def load_faster_rcnn(device):
    def _loader():
        model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT).to(device)
        model.eval()
        return model
    return _get_cached_model(f"faster_rcnn_{device}", _loader)

def load_keypoint_rcnn(device):
    def _loader():
        model = keypointrcnn_resnet50_fpn(weights=KeypointRCNN_ResNet50_FPN_Weights.DEFAULT).to(device)
        model.eval()
        return model
    return _get_cached_model(f"keypoint_rcnn_{device}", _loader)

def load_i3d(device):
    def _loader():
        model = create_resnet(input_channel=3, model_depth=50, model_num_class=400)
        model.eval()
        model.to(device)
        return model
    return _get_cached_model(f"i3d_{device}", _loader)

def load_musiq(device):
    def _loader():
        try:
            metric = pyiqa.create_metric('musiq', device=device)
            metric.eval()
            return metric
        except Exception as e:
            logger.info(f"Failed to load MUSIQ: {e}")
            return None
    return _get_cached_model(f"musiq_{device}", _loader)
    
def load_laion_aes_vit(device):
    def _loader():
        try:
            metric = pyiqa.create_metric('laion_aes_pl', device=device)
            metric.eval()
            return metric
        except Exception as e:
            logger.info(f"Failed to load LAION-AES: {e}")
            return None
    return _get_cached_model(f"laion_aes_{device}", _loader)

def load_clip(device):
    def _loader():
        model_name = "openai/clip-vit-base-patch32"
        model = CLIPModel.from_pretrained(model_name).to(device).eval()
        proc = CLIPProcessor.from_pretrained(model_name)
        return model, proc
    return _get_cached_model(f"clip_{device}", _loader)

def get_detection_results(
    video_tensor, 
    detector, 
    device=None
):
    if device is None: device = video_tensor.device
    results = []
    batch_size = 4
    with torch.no_grad():
        for i in range(0, len(video_tensor), batch_size):
            batch = video_tensor[i : i + batch_size]
            preds = detector(batch)
            for pred in preds:
                valid = (pred['labels'] == 1) & (pred['scores'] > 0.7)
                if valid.any():
                    best_idx = torch.argmax(pred['scores'][valid])
                    box = pred['boxes'][valid][best_idx].cpu()
                    score = pred['scores'][valid][best_idx].cpu()
                    results.append((box, score))
                else:
                    results.append(None)
    return results

def get_keypoint_results(
    video_tensor, 
    keypoint_detector
):
    results = []
    batch_size = 4
    with torch.no_grad():
        for i in range(0, len(video_tensor), batch_size):
            batch = video_tensor[i : i + batch_size]
            preds = keypoint_detector(batch)
            for pred in preds:
                valid = (pred['labels'] == 1) & (pred['scores'] > 0.7)
                if valid.any():
                    best_idx = torch.argmax(pred['scores'][valid])
                    keypoints = pred['keypoints'][valid][best_idx].cpu()
                    score = pred['scores'][valid][best_idx].cpu()
                    results.append((keypoints, score))
                else:
                    results.append(None)
    return results

def load_depth_anything(device):
    def _loader():
        model_name: str = "LiheYoung/depth-anything-small-hf",
        processor = AutoImageProcessor.from_pretrained(model_name)
        model = AutoModelForDepthEstimation.from_pretrained(model_name).to(device).eval()
        return model, processor
    return _get_cached_model(f"depth-anything-small-hf_{device}", _loader)


def infer_depth(
    video_tensor: torch.Tensor,
    batch_size,
    processor: AutoImageProcessor,
    model: AutoModelForDepthEstimation,
    device
):
    """Helper: Extract depth maps from video tensor [T, C, H, W]"""
    # 1. Convert Tensor to PIL Images (Model expects RGB 0-255)
    # Assuming input video is normalized to [-1, 1] or [0, 1]
    if video_tensor.min() < 0:
        video_tensor = (video_tensor + 1.0) / 2.0
    
    # 获取原始尺寸
    original_h, original_w = video_tensor.shape[-2:]
    
    # 从 Processor 获取标准化参数 (Lazy setup)
    # Depth Anything/DPT 使用标准的 ImageNet mean/std
    # 获取 mean/std 并转为 [1, 3, 1, 1] 的 GPU Tensor 以便广播
    mean = getattr(processor, 'image_mean', [0.485, 0.456, 0.406])
    std = getattr(processor, 'image_std', [0.229, 0.224, 0.225])
    _norm_mean = torch.tensor(mean, device=device).view(1, 3, 1, 1)
    _norm_std = torch.tensor(std, device=device).view(1, 3, 1, 1)
    
    # 获取目标输入尺寸 (Depth Anything 默认为 518x518)
    # 如果 processor.size 是 {'height': 518, 'width': 518}
    size_conf = getattr(processor, 'size', {})
    h = size_conf.get('height', 518)
    w = size_conf.get('width', 518)
    _target_size = (h, w)
    
    depth_maps = []
    
    with torch.no_grad():
        for i in range(0, len(video_tensor), batch_size):
            # [B, C, H, W]
            batch = video_tensor[i : i + batch_size]
            
            # --- Preprocessing (GPU) ---
            # A. Resize 到模型输入尺寸
            batch_resized = torch.nn.functional.interpolate(
                batch, 
                size=_target_size, 
                mode='bilinear', 
                align_corners=False,
                antialias=True # PyTorch 1.11+ 推荐开启，防止混叠
            )
            
            # B. Normalize (标准 z-score)
            batch_norm = (batch_resized - _norm_mean) / _norm_std
            
            # --- Inference ---
            # 直接传入 pixel_values 跳过 processor 的处理逻辑
            outputs = model(pixel_values=batch_norm)
            predicted_depth = outputs.predicted_depth # [B, h_out, w_out]
            
            # --- Post-processing (GPU) ---
            # C. Resize 回原始分辨率
            prediction = torch.nn.functional.interpolate(
                predicted_depth.unsqueeze(1), # [B, 1, h, w]
                size=(original_h, original_w),
                mode="bicubic", # 深度图插值通常用 bicubic 效果更好
                align_corners=False,
            )
            
            depth_maps.append(prediction.squeeze(1)) # [B, H, W]
    return torch.cat(depth_maps, dim=0)