import logging
import pyiqa
import torch
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
        # pyiqa 会自动下载 Google 预训练的 MUSIQ 权重
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