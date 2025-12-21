import os
import json
import torch
import cv2
import numpy as np
import logging

from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from huggingface_hub import snapshot_download

from .option import Options
from src.utils.video_kit import load_video_to_gpu

MODAL_KEY_MAP = {
    "text_only": "for text only model",
    "text_image": "for text&image model",
    "fullymodal": "for fullymodal model",
}
class BenchmarkDataset(Dataset):
    def __init__(self, opt: Options):
        self.opt = opt
        self.caption = {}
        self.annotation = {}
        self.transform = transforms.Compose([
            transforms.Resize((opt.height, opt.width)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        # 1. 自动下载逻辑
        self._ensure_dataset_exists()
        
        # 2. 加载标注
        self._load_caption()
        self._load_annotation()
        self.dataset = self.caption if self.opt.phase == 'train' else self.annotation
        self.ids = list(self.dataset.keys())

    def _ensure_dataset_exists(self):
        """如果本地目录不存在或为空，尝试从 HF 下载"""
        if os.path.exists(self.opt.assets) and os.listdir(self.opt.assets):
            return
        hf_repo_id = getattr(self.opt, 'hf_repo_id', None)

        logging.info(f"Downloading dataset from HuggingFace: {hf_repo_id} -> {self.opt.assets}")
        try:
            snapshot_download(
                repo_id=hf_repo_id,
                repo_type="dataset",
                local_dir=self.opt.assets,
                local_dir_use_symlinks=False,
                resume_download=True
            )
        except Exception as e:
            raise RuntimeError(f"Failed to download dataset: {e}")
        
    def _load_caption(self):
        '''加载训练使用的 caption'''
        try:
            with open(self.opt.caption, 'r') as f:
                self.caption = json.load(f)
        except FileNotFoundError:
            logging.warning(f"Caption file not found: {self.opt.caption}")
            self.caption = None
            if self.opt.phase == 'train':
                raise RuntimeError("Caption file not found. Please provide a valid path.")
        finally:
            f.close()
            
    def _load_annotation(self):
        '''加载训练使用的 annotation'''
        try:
            with open(self.opt.annotation, 'r') as f:
                self.annotation = json.load(f)
        except FileNotFoundError:
            logging.warning(f"Annotation file not found: {self.opt.annotation}")
            self.annotation = None
            if self.opt.phase == 'test':
                raise RuntimeError("Annotation file not found. Please provide a valid path.")
        finally:
            f.close()
    
    def _get_prompt(self, id):
        if self.opt.phase == 'train':
            return (self.caption[id]['prompt']['positive'], 
                    self.caption[id]['prompt']['negative'])
        else:
            return (self.annotation[id]['prompt']['positive'][MODAL_KEY_MAP[self.opt.modal]], 
                    self.annotation[id]['prompt']['negative'])
    
    def _load_image(self, rel_path):
        """读取图片 -> [C, H, W]"""
        path = os.path.join(self.opt.assets, rel_path)
        try:
            img = Image.open(path).convert('RGB')
            return self.transform(img) # [Fix] Apply transform
        except Exception as e:
            logging.error(f"Failed to load image {path}: {e}")
            return torch.zeros(3, self.opt.height, self.opt.width)
    
    def _load_video(self, rel_path):
        """读取视频 -> [T, C, H, W]"""
        path = os.path.join(self.opt.assets, rel_path)
        try:
            return load_video_to_gpu(path, device='cpu') 
        except Exception as e:
            logging.error(f"Failed to load video {path}: {e}")
            return torch.zeros(self.opt.clip_len, 3, self.opt.height, self.opt.width)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        vid_id = self.ids[index]
        data = self.dataset[vid_id]
        
        ego_video = self._load_video(data['the first view'])
        exo_video = self._load_video(data['the third view']) # GT
        ref_img = self._load_image(data['reference'])
        pos_p, neg_p = self._get_prompt(vid_id)
        
        return {
            'video_id': vid_id,
            'ego_video': ego_video,
            'exo_video': exo_video, # GT
            'ref_image': ref_img,
            'pos_prompt': pos_p,
            'neg_prompt': neg_p
        }