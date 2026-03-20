"""Dataset and utilities for loading benchmark data.

Provides :class:`BenchmarkDataset` used by the data loader to load images and videos,
and to prepare samples for model input.
"""

import os
import json
import torch
import cv2
import numpy as np
import logging
from typing import Dict, Any

from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from huggingface_hub import snapshot_download

from .option import Options
from ..utils.video_kit import load_video_to_device

class BenchmarkDataset(Dataset):
    def __init__(self, opt: Options) -> None:
        """Initialize dataset.

        Args:
            opt: Options object with runtime configuration.
        """
        self.opt = opt
        self.index: Dict[str, Any] = {}
        self.transform = transforms.Compose([
            transforms.Resize((opt.height, opt.width)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        # 1. 自动下载逻辑 下载到 assets/
        self._ensure_dataset_exists()
        self.data_root = os.path.join(self.opt.assets, self.opt.phase)
        
        # 2. 加载标注
        self._load_index()
        self.ids = list(self.index.keys())

    def _ensure_dataset_exists(self) -> None:
        """Ensure the dataset exists locally, downloading from HuggingFace if needed.

        Raises:
            RuntimeError: If downloading fails.
        """
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
        
    def _load_index(self) -> None:
        """Load index for the selected phase.

        Raises:
            RuntimeError: If Index file is not found.
        """
        json_path = os.path.join(self.data_root, 'index.json')
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                self.index = json.load(f)
        except FileNotFoundError:
            raise RuntimeError(f"Index file not found at {json_path}")
            
    def _load_image(self, rel_path: str) -> "torch.Tensor":
        """Load an image and apply transforms.

        Args:
            rel_path: Relative path to the image inside data root.

        Returns:
            Transformed image tensor of shape [C, H, W].
        """
        path = os.path.join(self.data_root, rel_path)
        try:
            img = Image.open(path).convert('RGB')
            return self.transform(img)
        except Exception as e:
            logging.error(f"Failed to load image {path}: {e}")
            return torch.zeros(3, self.opt.height, self.opt.width)
    
    def _load_video(self, rel_path: str) -> "torch.Tensor":
        """Load a video and return a tensor.

        Args:
            rel_path: Relative path to the video inside data root.

        Returns:
            Video tensor of shape [T, C, H, W].
        """
        path = os.path.join(self.data_root, rel_path)
        try:
            return load_video_to_device(path, device='cpu') 
        except Exception as e:
            logging.error(f"Failed to load video {path}: {e}")
            return torch.zeros(self.opt.clip_len, 3, self.opt.height, self.opt.width)

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Return a single sample by index.

        Args:
            index: Index of the sample.

        Returns:
            A dict containing 'video_id', 'ego_video', 'exo_video', and 'ref_image'.
        """
        vid_id = self.ids[index]
        data = self.index[vid_id]
        
        ego_video = self._load_video(data['ego video path'])
        exo_video = self._load_video(data['exo video path']) # GT
        ref_img = self._load_image(data['reference image path'])
        
        return {
            'video_id': vid_id,
            'ego_video': ego_video,
            'exo_video': exo_video, # GT
            'ref_image': ref_img,
        }