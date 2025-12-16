import logging
import pyiqa

import torch
from torch import Tensor
import torch.nn.functional as F
from pytorchvideo.models.resnet import create_resnet
from torchvision import transforms
from torchvision.models.detection import KeypointRCNN, keypointrcnn_resnet50_fpn, KeypointRCNN_ResNet50_FPN_Weights

def load_dinov2(device):
    print("Loading DINOv2 for Appearance Consistency...")
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    model.eval()
    model.to(device)
    transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return model, transform

def load_imaging_quality_metric(device):
    try:
        print("Loading Imaging Quality Metric (MUSIQ)...")
        metric = pyiqa.create_metric('musiq', device=device)
        metric.eval()
        return metric
    except Exception as e:
        print(f"Failed to load MUSIQ: {e}")
        return None
    
def load_aesthetic_metric(device):
    try:
        logging.info("Loading Aesthetic Metric (LAION-AES ViT-L/14)...")
        metric = pyiqa.create_metric('laion_aes_pl', device=device)
        metric.eval()
        return metric
    except Exception as e:
        print(f"Failed to load pyiqa: {e}")
        return None
    
def get_detection_results(video_tensor: Tensor, detector, device=None):
    """
    Faster R-CNN model 推理计算视频的检测结果 (Bounding Boxes)
    Args:
        video_tensor: [T, C, H, W]
        detector: Faster R-CNN model
    Returns:
        results: List of (box, score) for each frame. None if no detection.
                 box format: [x1, y1, x2, y2]
    """
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

def get_keypoint_results(video_tensor, keypoint_detector: KeypointRCNN):
    """
    计算视频的人体关键点 (Keypoint Detection)
    
    Args:
        video_tensor: [T, C, H, W] tensor on GPU/CPU
        keypoint_detector: Loaded Keypoint R-CNN model
        
    Returns:
        results: List of (keypoints, scores) or None.
                 keypoints format: Tensor [num_keypoints, 3] (x, y, confidence)
    """
    results = []
    batch_size = 4
    keypoint_detector.eval()
        
    with torch.no_grad():
        for i in range(0, len(video_tensor), batch_size):
            batch = video_tensor[i : i + batch_size]
            preds = keypoint_detector(batch)
            for pred in preds:
                valid = (pred['labels'] == 1) & (pred['scores'] > 0.7)
                if valid.any():
                    best_idx = torch.argmax(pred['scores'][valid])
                    # keypoints: [K, 3]
                    keypoints = pred['keypoints'][valid][best_idx].cpu()
                    score = pred['scores'][valid][best_idx].cpu()
                    results.append((keypoints, score))
                else:
                    results.append(None)
    return results

def load_i3d(device):
    """
    加载 I3D 模型。
    """
    model = create_resnet(input_channel=3, model_depth=50, model_num_class=400)
    model.eval()
    model.to(device)
    return model