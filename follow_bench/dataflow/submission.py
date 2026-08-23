"""Data structures and IO utilities for handling submission files.

This module provides the :class:`Submission` class which loads a submission JSON
and offers helpers for validation and loading generated videos into memory.
"""
import os
import torch
import decord
import numpy as np
import torch.nn.functional as F
import json
from pathlib import Path
from typing import Union, Optional, Dict, Any
import logging
logger = logging.getLogger(__name__)

from .option import Options
REQUIRED_META_KEYS = set([
    'team_name', 
    'model_name', 
    # 'modal', 
    # 'mode', 
    'contact'
])
ALLOWED_EXTENSIONS = [
    '.mp4', 
    '.avi', 
    '.mov',
]
# STANDARD_RESOLUTION = (CONFIG['rules']['resolution_height'], CONFIG['rules']['resolution_width'])
# STANDARD_CLIP_LEN = CONFIG['rules']['clip_len']
# STANDARD_CLIP_FPS = CONFIG['rules']['fps']

class Submission:
    """Submission data IO and access helper.

    The class loads a submission JSON file and maps video IDs to on-disk video paths
    and provides convenience methods to validate the submission and to load videos
    as tensors.

    Attributes:
        meta_info: Metadata from the submission file.
        mapping: Mapping from video id to result entries.
    """
    meta_info: Dict[str, Any]
    mapping: Dict[str, Dict[str, Any]]

    def __init__(
        self,
        opt: Options, 
        submission_path: Union[str, Path],
        source_path: Union[str, Path],
        device: str = 'cpu'
    ) -> None:
        """Initialize the Submission.

        Args:
            submission_path: Path to the submission JSON file.
            source_path: Root directory containing generated videos (relative paths in JSON).
            device: Device to load tensors onto (e.g., 'cpu' or 'cuda').
        """
        self.opt = opt
        self.device = device
        self.source_path = Path(source_path)
        self.submission_path = Path(submission_path)
        self.meta_info = {}
        self.mapping = {}
        self._load()
    
    def _load(self) -> None:
        """Load and parse the submission JSON file.

        Raises:
            FileNotFoundError: If submission file is not found.
            ValueError: If submission JSON is invalid.
        """
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
        
    def validate_all(self) -> None:
        """Validate submission meta, case count, and per-video properties.

        Raises:
            ValueError: Aggregated validation errors if any check fails.
        """
        errors: list[str] = []

        # A. Meta Check
        missing = REQUIRED_META_KEYS - set(self.meta_info.keys())
        if missing:
            errors.append(f"[Meta] Missing keys: {missing}")

        # B. Count Check
        # if len(self.mapping) != TOTAL_CASES_NUM:
        #     errors.append(f"[Count] Expected {TOTAL_CASES_NUM} cases, found {len(self.mapping)}")

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
            # vid_errors = validate_video_properties(
            #     full_path, 
            #     STANDARD_RESOLUTION, 
            #     STANDARD_CLIP_FPS, 
            #     STANDARD_CLIP_LEN
            # )
            # if vid_errors:
            #     errors.append(f"[{case_id}] Properties Invalid: {'; '.join(vid_errors)}")

        if errors:
            msg = f"Submission Validation Failed with {len(errors)} errors:\n" + "\n".join(errors[:20])
            if len(errors) > 20: msg += f"\n...and {len(errors)-20} more."
            raise ValueError(msg)
            
        logger.info("✅ Submission validation passed successfully.")
        
    def get_generated_video(self, video_id: str) -> Optional[Any]:
        """Load and return the generated video tensor for the given ID.

        Args:
            video_id: ID of the video to load.

        Returns:
            A video tensor or None if the video is missing or failed to load.
        """
        if video_id not in self.mapping:
            return None
        rel_path = self.mapping[video_id]["generated video"]
        full_path = str(self.source_path / rel_path)
        
        return self._load_video(full_path)

    def __getitem__(self, item: str) -> Optional[Any]:
        """Get generated video tensor by video ID.
        
        Convenience method that delegates to get_generated_video().
        
        Args:
            item: Video ID string
            
        Returns:
            Video tensor or None if not found or failed to load
        """
        return self.get_generated_video(item)

    def __len__(self) -> int:
        """Return the number of video cases in the submission.
        
        Returns:
            Number of video cases in the mapping
        """
        return len(self.mapping)

    def _load_video(self, path: str) -> "torch.Tensor":
        """
        高内存效率的视频读取器：按需抽帧 + 极速降采样
        """
        vr = decord.VideoReader(path, ctx=decord.cpu(0))
        
        total_frames = len(vr)
        target_frames = self.opt.num_frames
        target_h = self.opt.height
        target_w = self.opt.width
        indices = np.linspace(0, total_frames - 1, target_frames).round().astype(np.int64)
        frames = vr.get_batch(indices).asnumpy()
        tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
        if tensor.shape[2] != target_h or tensor.shape[3] != target_w:
            tensor = F.interpolate(tensor, size=(target_h, target_w), mode='bilinear', align_corners=False)
        return tensor