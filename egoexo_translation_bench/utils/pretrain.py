import pyiqa
from PIL import Image
import logging
logger = logging.getLogger(__name__)

import torch
from torchvision import transforms
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
    CLIPVisionModelWithProjection,
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

def load_yolov8(device):
    '''
    load_yolov8 的 Docstring
    
    :param device: 说明
    '''
    def _loader():
        model = YOLO("yolov8x.pt")
        return model
    return _get_cached_model(f"yolov8_{device}", _loader)

def clip_preprocess_tensor(images: torch.Tensor, size=(224, 224)):
    """
    CLIP 的标准预处理
    Args:
        images: [B, C, H, W] or [C, H, W], value range [0, 1]
    Returns:
        [B, C, H, W] normalized tensor
    """
    if images.ndim == 3:
        images = images.unsqueeze(0)
    
    # CLIP Mean/Std
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=images.device).view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=images.device).view(1, 3, 1, 1)
    
    # 1. Resize (Bicubic)
    images = F.interpolate(images, size=size, mode='bicubic', align_corners=False, antialias=True)
    
    # 2. Normalize
    images = (images - mean) / std
    
    return images

def get_yolo_detection_results(video_tensor: torch.Tensor, detector: YOLO):
    """
    Args:
        video_tensor: [T, C, H, W] float32 [0, 1] on GPU/CPU
    Returns:
        results: List of (bbox_tensor, score_float)
    """
    results = []
    # YOLOv8 支持直接输入 float32 [0-1] 的 BCHW Tensor
    batch_size = 16 
    
    # 确保是 [0, 1]
    if video_tensor.min() < 0:
        video_tensor = (video_tensor + 1.0) / 2.0
    video_tensor = video_tensor.clamp(0, 1)

    for i in range(0, len(video_tensor), batch_size):
        batch = video_tensor[i : i + batch_size]
        
        # YOLO 推理
        # verbose=False, classes=0 (Person)
        preds = detector(batch, verbose=False, classes=0)
        
        for r in preds:
            if len(r.boxes) > 0:
                # 获取最高置信度的框
                # Ultralytics 的 boxes.xyxy 已经是 Tensor
                # 默认在 r.boxes.data 所在的 device
                box = r.boxes[0].xyxy[0] # [x1, y1, x2, y2]
                conf = r.boxes[0].conf[0].item()
                results.append((box, conf))
            else:
                results.append(None)
                
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