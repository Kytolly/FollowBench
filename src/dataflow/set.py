import os
import json
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from huggingface_hub import snapshot_download

class Ego2ExoBenchmarkDataset(Dataset):
    def __init__(self, opt):
        """
        Args:
            opt: 配置对象
                - dataroot: 数据集本地根目录
                - hf_repo_id: (可选) HuggingFace 仓库 ID
                - json_path: annotation.json 的路径 (相对或绝对)
                - phase: 'train' 或 'test'
                - prompt_mode: 'text_only', 'text_image', 'fullymodal'
                - clip_len: 视频片段长度
                - load_size: 图像 resize 大小
        """
        self.opt = opt
        self.root = opt.dataroot
        self.phase = opt.phase
        self.clip_len = opt.clip_len
        
        # 1. 自动下载逻辑
        self._ensure_dataset_exists()
        
        # 2. 加载标注
        if not os.path.isabs(opt.json_path):
            self.json_path = os.path.join(self.root, opt.json_path)
        else:
            self.json_path = opt.json_path

        if not os.path.exists(self.json_path):
            raise FileNotFoundError(f"Annotation file not found: {self.json_path}")

        with open(self.json_path, 'r') as f:
            self.annotations = json.load(f)
        
        self.ids = list(self.annotations.keys())
        
        # 3. 预处理
        self.transform = transforms.Compose([
            transforms.Resize((opt.load_size, opt.load_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def _ensure_dataset_exists(self):
        """如果本地目录不存在或为空，尝试从 HF 下载"""
        if os.path.exists(self.root) and os.listdir(self.root):
            return

        hf_repo_id = getattr(self.opt, 'hf_repo_id', None)
        if not hf_repo_id:
            # 如果是本地模式且没配 HF ID，只报 Warning 或 Error
            print(f"[Info] Local dataset not found at {self.root} and no hf_repo_id provided.")
            return

        print(f"Downloading dataset from HuggingFace: {hf_repo_id} -> {self.root}")
        try:
            snapshot_download(
                repo_id=hf_repo_id,
                repo_type="dataset",
                local_dir=self.root,
                local_dir_use_symlinks=False,
                resume_download=True
            )
        except Exception as e:
            raise RuntimeError(f"Failed to download dataset: {e}")

    def _load_video_clip(self, rel_path):
        """读取视频 -> [T, C, H, W]"""
        path = os.path.join(self.root, rel_path)
        cap = cv2.VideoCapture(path)
        frames = []
        
        # 简单策略: 读取前 clip_len 帧
        for _ in range(self.clip_len):
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            frames.append(self.transform(frame))
            
        cap.release()
        
        if len(frames) == 0:
            # Fallback for broken video
            return torch.zeros(self.clip_len, 3, self.opt.load_size, self.opt.load_size)
            
        # Padding
        if len(frames) < self.clip_len:
            frames += [frames[-1]] * (self.clip_len - len(frames))
            
        return torch.stack(frames) # [T, C, H, W]

    def _get_prompt(self, prompt_data):
        if self.phase == 'train':
            return prompt_data['positive'], prompt_data['negative']
        else:
            mode_map = {
                'text_only': 'for text only model',
                'text_image': 'for_text_image model',
                'fullymodal': 'for fullymodal model'
            }
            key = mode_map.get(self.opt.prompt_mode, 'for fullymodal model')
            return prompt_data['positive'].get(key, ""), prompt_data['negative']

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        vid_id = self.ids[index]
        data = self.annotations[vid_id]
        
        ego_video = self._load_video_clip(data['the first view'])
        exo_video = self._load_video_clip(data['the third view']) # GT
        
        ref_path = os.path.join(self.root, data['reference image'])
        ref_img = Image.open(ref_path).convert('RGB')
        ref_tensor = self.transform(ref_img)
        
        pos_p, neg_p = self._get_prompt(data['prompt'])
        
        return {
            'video_id': vid_id,
            'ego_video': ego_video,
            'exo_video': exo_video, # GT
            'ref_image': ref_tensor,
            'pos_prompt': pos_p,
            'neg_prompt': neg_p
        }