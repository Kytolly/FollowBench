import logging
import pyiqa
import torch
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel
from torchvision.models.detection import (
    keypointrcnn_resnet50_fpn, KeypointRCNN_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
)
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
from pytorchvideo.models.resnet import create_resnet

from .gpu import clear_gpu_memory

_MODEL_CACHE = {}

def _get_cached_model(key, loader_func, *args, **kwargs):
    if key not in _MODEL_CACHE:
        print(f"[System] Loading model into cache: {key} ...")
        _MODEL_CACHE[key] = loader_func(*args, **kwargs)
    return _MODEL_CACHE[key]

def clear_models_cache():
    clear_gpu_memory(_MODEL_CACHE)
    print("[System] Models cache cleared.")

def load_dinov2(device):
    def _loader():
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        model.eval()
        model.to(device)
        transform = transforms.Compose([
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
            print(f"Failed to load MUSIQ: {e}")
            return None
    return _get_cached_model(f"musiq_{device}", _loader)
    
def load_laion_aes_vit(device):
    def _loader():
        try:
            metric = pyiqa.create_metric('laion_aes_pl', device=device)
            metric.eval()
            return metric
        except Exception as e:
            print(f"Failed to load LAION-AES: {e}")
            return None
    return _get_cached_model(f"laion_aes_{device}", _loader)

def load_clip_model(device):
    def _loader():
        try:
            clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
            return clip
        except Exception as e:
            print(f"Failed to load CLIP: {e}")
            return None
    return _get_cached_model(f"clip_model_{device}", _loader)

def load_clip_processor(device):
    def _loader():
        try:
            proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32").to(device)
            return proc
        except Exception as e:
            print(f"Failed to load CLIP: {e}")
            return None
    return _get_cached_model(f"clip_processor_{device}", _loader)

def get_detection_results(video_tensor, detector, device=None):
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

def get_keypoint_results(video_tensor, keypoint_detector):
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