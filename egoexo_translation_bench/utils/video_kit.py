"""Video processing utilities for the EgoExo Translation Benchmark.

This module provides comprehensive video I/O, processing, and validation
utilities including frame extraction, optical flow computation, trajectory
analysis, and video property validation.
"""

import cv2
import numpy as np
from scipy.stats import pearsonr
import torch
from torch import Tensor
from typing import List, Optional, Dict, Any, Union, Tuple
from pathlib import Path


def get_video_frames(video_path: Union[str, Path], max_frames: int = 60) -> List[np.ndarray]:
    """Extract frames from a video file.
    
    Args:
        video_path: Path to the video file
        max_frames: Maximum number of frames to extract
        
    Returns:
        List of frames as numpy arrays in RGB format [H, W, C]
        
    Raises:
        cv2.error: If video file cannot be opened or read
    """
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret or len(frames) >= max_frames:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    finally:
        cap.release()
        
    return frames

def calculate_optical_flow_magnitude(frames: List[np.ndarray]) -> List[float]:
    """Calculate optical flow magnitude for video frames.
    
    Computes dense optical flow between consecutive frames using Farneback method
    and returns the mean flow magnitude for each frame transition.
    
    Args:
        frames: List of video frames as numpy arrays in RGB format [H, W, C]
        
    Returns:
        List of mean optical flow magnitudes for each frame transition
        
    Raises:
        IndexError: If frames list is empty
    """
    if not frames:
        return []
        
    mags = []
    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_RGB2GRAY)
    
    for i in range(1, len(frames)):
        curr_gray = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        mags.append(np.mean(mag))
        prev_gray = curr_gray
        
    return mags


def get_motion_series(frames: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Extract global motion series from video frames.
    
    Computes dense optical flow between consecutive frames and extracts
    the average motion in x and y directions, representing global camera motion.
    
    Args:
        frames: List of video frames as numpy arrays in RGB format [H, W, C]
        
    Returns:
        Tuple of (motion_x, motion_y) as numpy arrays representing
        average motion in x and y directions for each frame transition
        
    Raises:
        IndexError: If frames list is empty
    """
    if not frames:
        return np.array([]), np.array([])
        
    # Convert first frame to grayscale
    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_RGB2GRAY)
    motion_x = []
    motion_y = []
    
    for i in range(1, len(frames)):
        curr_gray = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
        
        # Compute dense optical flow using Farneback algorithm
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None, 
            0.5, 3, 15, 3, 5, 1.2, 0
        )
        
        # Calculate average optical flow (represents global/camera motion)
        avg_dx = np.mean(flow[..., 0])
        avg_dy = np.mean(flow[..., 1])
        
        motion_x.append(avg_dx)
        motion_y.append(avg_dy)
        
        prev_gray = curr_gray
        
    return np.array(motion_x), np.array(motion_y)


def load_video_as_tensor(video_path: Union[str, Path]) -> Tensor:
    """Load video as a PyTorch tensor.
    
    Reads video frames and converts them to a tensor format [T, C, H, W]
    with values normalized to [0, 1].
    
    Args:
        video_path: Path to the video file
        
    Returns:
        Video tensor of shape [T, C, H, W] normalized to [0, 1],
        or empty tensor if no frames found
    """
    frames = get_video_frames(video_path)  # [H, W, C]
    if not frames:
        return torch.empty(0)
        
    tensor = torch.stack([torch.from_numpy(f) for f in frames])
    tensor = tensor.permute(0, 3, 1, 2).float() / 255.0
    return tensor


def load_video_to_device(
    video_path: Union[str, Path], 
    target_size: Optional[Tuple[int, int]] = None, 
    device: str = 'cuda'
) -> Optional[Tensor]:
    """Load video directly to specified device as tensor.
    
    Reads video frames, optionally resizes them, and converts to tensor
    format [T, C, H, W] on the specified device.
    
    Args:
        video_path: Path to the video file
        target_size: Optional target size as (height, width) for resizing
        device: Target device ('cuda' or 'cpu')
        
    Returns:
        Video tensor of shape [T, C, H, W] normalized to [0, 1] on specified device,
        or None if video cannot be loaded
    """
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if target_size is not None:
                frame = cv2.resize(frame, (target_size[1], target_size[0]))
            frames.append(frame)
    finally:
        cap.release()
        
    if not frames:
        return None
    
    video_np = np.stack(frames)  # [T, H, W, C]
    tensor = torch.from_numpy(video_np).to(device, non_blocking=True)
    
    # [T, H, W, C] -> [T, C, H, W] & Normalize to [0, 1]
    tensor = tensor.permute(0, 3, 1, 2).float() / 255.0
    return tensor


def tensor_to_numpy(tensor: Optional[Tensor]) -> List[np.ndarray]:
    """Convert GPU tensor to CPU numpy arrays.
    
    Converts tensor from [T, C, H, W] format (0-1) to list of numpy arrays
    in [H, W, C] format (0-255).
    
    Args:
        tensor: Input tensor of shape [T, C, H, W] with values in [0, 1]
        
    Returns:
        List of numpy arrays in [H, W, C] format with values in [0, 255]
    """
    if tensor is None:
        return []
        
    # [T, C, H, W] -> [T, H, W, C]
    arr = tensor.permute(0, 2, 3, 1).cpu().numpy()
    # 0-1 -> 0-255
    return [(frame * 255).astype(np.uint8) for frame in arr]


def compute_flow(frames: Tensor, flow_model: Any) -> Optional[Tensor]:
    """Compute optical flow using a flow model.
    
    Args:
        frames: Input frames tensor of shape [T, C, H, W]
        flow_model: Optical flow model (e.g., RAFT)
        
    Returns:
        Optical flow tensor of shape [T-1, 2, H, W] or None if insufficient frames
    """
    if len(frames) < 2:
        return None
        
    img1 = frames[:-1]
    img2 = frames[1:]
    
    with torch.no_grad():
        flows = flow_model(img1, img2)[-1]  # [T-1, 2, H, W]
        
    return flows


def get_traj(results: List[Optional[Tuple[Any, Any]]]) -> List[Optional[np.ndarray]]:
    """Extract trajectory from detection results.
    
    Computes center points from bounding box detections to form a trajectory.
    
    Args:
        results: List of detection results, each containing (box, confidence) or None
        
    Returns:
        List of trajectory points as numpy arrays [x, y] or None for missing detections
    """
    traj = []
    for res in results:
        if res is None:
            traj.append(None)
        else:
            box, _ = res
            cx = (box[0] + box[2]) / 2.0
            cy = (box[1] + box[3]) / 2.0
            traj.append(np.array([cx, cy]))
    return traj


def extract_i3d_features(video_tensor: Tensor, i3d_model: Any) -> np.ndarray:
    """Extract I3D features from video tensor.
    
    Args:
        video_tensor: Video tensor of shape [T, 3, H, W] with values in [0, 1]
        i3d_model: Pre-trained I3D model
        
    Returns:
        Feature array of shape [1, D] where D is the feature dimension
    """
    # Reshape to [1, C, T, H, W] and normalize to [-1, 1]
    video_input = video_tensor.permute(1, 0, 2, 3).unsqueeze(0)  # [1, C, T, H, W]
    video_input = (video_input * 2.0) - 1.0 
    
    with torch.no_grad():
        features = i3d_model(video_input)  # [1, D]
        
    return features.cpu().numpy()


def inspect_video(video_path: Union[str, Path]) -> Optional[Dict[str, Union[int, float]]]:
    """Inspect video properties and return metadata.
    
    Args:
        video_path: Path to the video file
        
    Returns:
        Dictionary containing video properties:
        - 'width': Video width in pixels
        - 'height': Video height in pixels  
        - 'fps': Frames per second
        - 'frame_count': Total number of frames
        Returns None if video cannot be opened
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        return {
            'width': width,
            'height': height,
            'fps': fps,
            'frame_count': frame_count
        }
    finally:
        cap.release()


def validate_video_properties(
    video_path: Union[str, Path], 
    standard_res: Tuple[int, int], 
    standard_fps: float, 
    standard_len: int
) -> List[str]:
    """Validate video properties against standards.
    
    Checks video resolution, frame rate, and frame count against expected values.
    
    Args:
        video_path: Path to the video file
        standard_res: Expected resolution as (height, width)
        standard_fps: Expected frame rate
        standard_len: Expected number of frames
        
    Returns:
        List of error messages. Empty list indicates validation passed.
    """
    props = inspect_video(video_path)
    if props is None:
        return ["Failed to open or corrupted video file."]
    
    errors = []
    
    # 1. Resolution Check (H, W)
    std_h, std_w = standard_res
    if (props['height'], props['width']) != (std_h, std_w):
        errors.append(
            f"Resolution mismatch: Got {props['width']}x{props['height']}, "
            f"expected {std_w}x{std_h}"
        )
        
    # 2. FPS Check (Allow small float error)
    if abs(props['fps'] - standard_fps) > 0.1:
        errors.append(f"FPS mismatch: Got {props['fps']:.2f}, expected {standard_fps}")
        
    # 3. Frame Count Check
    if props['frame_count'] != standard_len:
        errors.append(
            f"Frame count mismatch: Got {props['frame_count']}, expected {standard_len}"
        )
        
    return errors
