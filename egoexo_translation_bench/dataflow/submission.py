"""Data structures and IO utilities for handling submission files.

This module provides the :class:`Submission` class which loads a submission JSON
and offers helpers for validation and loading generated videos into memory.
"""

import json
from pathlib import Path
from typing import Union, Optional, Dict, Any
import logging
logger = logging.getLogger(__name__)

from ..utils.video_kit import (
    load_video_to_device,
    validate_video_properties
)
from ..configs import CONFIG
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
TOTAL_CASES_NUM = CONFIG['submission']['total_cases_num']
STANDARD_RESOLUTION = (CONFIG['rules']['resolution_height'], CONFIG['rules']['resolution_width'])
STANDARD_CLIP_LEN = CONFIG['rules']['clip_len']
STANDARD_CLIP_FPS = CONFIG['rules']['fps']

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

    def __init__(self: "Submission",
                 submission_path: Union[str, Path],
                 source_path: Union[str, Path],
                 device: str = 'cpu') -> None:
        """Initialize the Submission.

        Args:
            submission_path: Path to the submission JSON file.
            source_path: Root directory containing generated videos (relative paths in JSON).
            device: Device to load tensors onto (e.g., 'cpu' or 'cuda').
        """
        self.device = device
        self.source_path = Path(source_path)
        self.submission_path = Path(submission_path)
        self.meta_info = {}
        self.mapping = {}
        self._load()
    
    def _load(self: "Submission") -> None:
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
        
    def validate_all(self: "Submission") -> None:
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
        
    def get_generated_video(self: "Submission", video_id: str) -> Optional[Any]:
        """Load and return the generated video tensor for the given ID.

        Args:
            video_id: ID of the video to load.

        Returns:
            A video tensor or None if the video is missing or failed to load.
        """
        if video_id not in self.mapping:
            return None
        rel_path = self.mapping[video_id]["generated video"]
        full_path = self.source_path / rel_path
        try:
            return load_video_to_device(str(full_path), device=self.device)
        except Exception as e:
            logger.error(f"Failed to load video {full_path}: {e}")
            return None

    def __getitem__(self: "Submission", item: str) -> Optional[Any]:
        return self.get_generated_video(item)

    def __len__(self: "Submission") -> int:
        return len(self.mapping)