import json
import cv2
import numpy as np
from pathlib import Path
from typing import Union
import logging
logger = logging.getLogger(__name__)

from src.utils.video_kit import (
    load_video_to_device,
    validate_video_properties
)
from . import (
    REQUIRED_META_KEYS,
    ALLOWED_EXTENSIONS,
    TOTAL_CASES_NUM,
    STANDARD_RESOLUTION,
    STANDARD_CLIP_LEN,
    STANDARD_CLIP_FPS,
)

class Submission:
    """
    专门负责 Submission 数据的 IO 和 Tensor 转换。
    它作为一个只读的 Data Mapping 将 Video ID 映射为显存中的 Video Tensor。
    """
    def __init__(self,
                 submission_path: Union[str, Path],
                 source_path: Union[str, Path],
                 device: str = 'cpu'):
        """
        Args:
            submission_path: submission.json 的路径
            source_path: 生成视频所在的根目录 (所有相对路径均基于此)
            device: 预加载的设备
        """
        self.device = device
        self.source_path = Path(source_path)
        self.submission_path = Path(submission_path)
        self.meta_info = {}
        self.mapping = {}
        self._load()
    
    def _load(self):
        """加载 JSON 并解析 meta/results"""
        if not self.submission_path.exists():
            raise FileNotFoundError(f"Submission file not found: {self.submission_path}")
        if not self.source_path.exists():
            logger.warning(f"Source path root not found: {self.source_path}")

        try:
            with open(self.submission_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {e}")

        self.meta_info = data.get('meta', {})
        self.mapping = data.get('results', {})
        logger.info(f"Loaded submission with {len(self.mapping)} cases.")
        
    def validate_all(self):
        """执行全量检查 Meta, 数量, 每个视频的物理属性"""
        errors = []

        # A. Meta Check
        missing = REQUIRED_META_KEYS - self.meta_info.keys()
        if missing:
            errors.append(f"[Meta] Missing keys: {missing}")

        # B. Count Check
        if len(self.mapping) != TOTAL_CASES_NUM:
            errors.append(f"[Count] Expected {TOTAL_CASES_NUM} cases, found {len(self.mapping)}")

        # C. Per-Case Check
        for case_id, entry in self.mapping.items():
            # 1. Structure
            if not isinstance(entry, dict) or "generated video" not in entry:
                errors.append(f"[{case_id}] Missing 'generated video' key.")
                continue
                
            rel_path = entry["generated video"]
            full_path = self.source_path / rel_path
            
            # 2. Existence
            if not full_path.exists():
                errors.append(f"[{case_id}] File not found: {full_path}")
                continue
                
            # 3. Extension
            if full_path.suffix.lower() not in ALLOWED_EXTENSIONS:
                errors.append(f"[{case_id}] Invalid extension: {full_path.suffix}")
                
            # 4. Video Properties (调用 video_kit)
            vid_errors = validate_video_properties(
                full_path, 
                STANDARD_RESOLUTION, 
                STANDARD_CLIP_FPS, 
                STANDARD_CLIP_LEN
            )
            if vid_errors:
                errors.append(f"[{case_id}] Properties Invalid: {'; '.join(vid_errors)}")

        if errors:
            msg = f"Submission Validation Failed with {len(errors)} errors:\n" + "\n".join(errors[:20])
            if len(errors) > 20: msg += f"\n...and {len(errors)-20} more."
            raise ValueError(msg)
            
        logger.info("✅ Submission validation passed successfully.")
        
    def get_generated_video(self, video_id: str):
        """获取视频 Tensor [T, C, H, W]"""
        if video_id not in self.mapping:
            return None
        rel_path = self.mapping[video_id]["generated video"]
        full_path = self.source_path / rel_path
        try:
            return load_video_to_device(str(full_path), device=self.device)
        except Exception as e:
            logger.error(f"Failed to load video {full_path}: {e}")
            return None

    def __getitem__(self, item):
        return self.get_generated_video(item)

    def __len__(self):
        return len(self.mapping)