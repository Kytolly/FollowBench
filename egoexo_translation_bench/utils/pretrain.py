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
from transformers import CLIPProcessor, CLIPModel
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

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
    """专门为 CCE, VV, AC, BSC 提供统一的检测模型"""
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