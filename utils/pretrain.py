import logging
import pyiqa

import torch
from torch import Tensor
import torch.nn.functional as F
from torchvision import transforms

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