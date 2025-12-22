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
from ..utils.video_kit import load_video_to_device

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
        
        # 1. 自动下载逻辑 下载到 assets/
        self._ensure_dataset_exists()
        self.data_root = os.path.join(self.opt.assets, self.opt.phase)
        
        # 2. 加载标注
        self._load_metadata()
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
        
    def _load_metadata(self):
        '''根据 phase 加载对应的 json 文件'''
        if self.opt.phase == 'train':
            # caption.json 位于 assets/train/caption.json
            json_path = self.opt.caption if self.opt.caption else os.path.join(self.data_root, 'caption.json')
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    self.caption = json.load(f)
            except FileNotFoundError:
                raise RuntimeError(f"Caption file not found at {json_path}")
        else:
            # annotation.json 位于 assets/test/annotation.json
            json_path = self.opt.annotation if self.opt.annotation else os.path.join(self.data_root, 'annotation.json')
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    self.annotation = json.load(f)
            except FileNotFoundError:
                raise RuntimeError(f"Annotation file not found at {json_path}")
    
    def _get_prompt(self, id):
        data: dict = self.caption[id] if self.opt.phase == 'train' else self.annotation[id]
        prompts_dict = data['prompts']
        if self.opt.modal in prompts_dict:
            pos_p = prompts_dict[self.opt.modal]
        else:
            # Fallback: 如果指定的 modal key 不存在，取第一个可用的 prompt
            logging.warning(f"Modal '{self.opt.modal}' not found in prompts for {id}.")
            pos_p = list(prompts_dict.values())[0] if prompts_dict else ""

        neg_p = data.get('negative_prompt', "")      
        return pos_p, neg_p
    
    def _load_image(self, rel_path):
        """读取图片 -> [C, H, W]"""
        path = os.path.join(self.data_root, rel_path)
        try:
            img = Image.open(path).convert('RGB')
            return self.transform(img) # [Fix] Apply transform
        except Exception as e:
            logging.error(f"Failed to load image {path}: {e}")
            return torch.zeros(3, self.opt.height, self.opt.width)
    
    def _load_video(self, rel_path):
        """读取视频 -> [T, C, H, W]"""
        path = os.path.join(self.data_root, rel_path)
        try:
            return load_video_to_device(path, device='cpu') 
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