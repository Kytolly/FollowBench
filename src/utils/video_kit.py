import cv2
import numpy as np
from scipy.stats import pearsonr
import torch
from torch import Tensor

def get_video_frames(video_path, max_frames=60):
    '''获取视频所有帧'''
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret or len(frames) >= max_frames:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    return frames

def calculate_optical_flow_magnitude(frames):
    '''计算视频光流'''
    mags = []
    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_RGB2GRAY)
    for i in range(1, len(frames)):
        curr_gray = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        mags.append(np.mean(mag))
        prev_gray = curr_gray
    return mags

def get_motion_series(frames):
    # 转换第一帧为灰度
    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_RGB2GRAY)
    motion_x = []
    motion_y = []
    
    for i in range(1, len(frames)):
        curr_gray = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
        
        # 使用 Farneback 光流算法计算稠密光流
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None, 
            0.5, 3, 15, 3, 5, 1.2, 0
        )
        
        # 计算全图平均光流 (代表该帧的全局/相机运动)
        avg_dx = np.mean(flow[..., 0])
        avg_dy = np.mean(flow[..., 1])
        
        motion_x.append(avg_dx)
        motion_y.append(avg_dy)
        
        prev_gray = curr_gray
        
    return np.array(motion_x), np.array(motion_y)

def load_video_as_tensor(video_path):
    '''读取视频并返回 [T, C, H, W] 格式的 Tensor 归一化到 [0, 1]'''
    frames = get_video_frames(video_path) # [H, W, C]
    if not frames:
        return torch.empty(0)
    tensor = torch.stack([torch.from_numpy(f) for f in frames])
    tensor = tensor.permute(0, 3, 1, 2).float() / 255.0
    return tensor

def load_video_to_device(video_path, target_size=None, device='cuda'):
    """读取视频并直接转换为 GPU Tensor [T, C, H, W]"""
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if target_size is not None:
            frame = cv2.resize(frame, (target_size[1], target_size[0]))
        frames.append(frame)
        count += 1
    cap.release()
    if not frames:
        return None
    
    video_np = np.stack(frames) # [T, H, W, C]
    tensor = torch.from_numpy(video_np).to(device, non_blocking=True)
    
    # [T, H, W, C] -> [T, C, H, W] & Normalize to [0, 1]
    tensor = tensor.permute(0, 3, 1, 2).float() / 255.0
    return tensor

def tensor_to_numpy(tensor):
    """
    GPU Tensor [T, C, H, W] (0-1) -> CPU Numpy List of [H, W, C] (0-255)
    """
    if tensor is None: return []
    # [T, C, H, W] -> [T, H, W, C]
    arr = tensor.permute(0, 2, 3, 1).cpu().numpy()
    # 0-1 -> 0-255
    return [(frame * 255).astype(np.uint8) for frame in arr]

def compute_flow(frames, flow_model):
    """计算光流"""
    if len(frames) < 2: return None
    img1 = frames[:-1]
    img2 = frames[1:]
    with torch.no_grad():
        flows = flow_model(img1, img2)[-1] # [T-1, 2, H, W]
    return flows

def get_traj(results):
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

def extract_i3d_features(video_tensor: Tensor, i3d_model):
    """
    提取 I3D 特征
    Args:
        video_tensor: [T, 3, H, W] (0-1 float)
    Returns:
        features: numpy array [1, D]
    """
    video_input = video_tensor.permute(1, 0, 2, 3).unsqueeze(0) # [1, C, T, H, W]
    video_input = (video_input * 2.0) - 1.0 
    with torch.no_grad():
        features = i3d_model(video_input) # [1, D]
    return features.cpu().numpy()

def inspect_video(video_path):
    """
    检查视频完整性并返回属性。
    Returns:
        dict: {'width', 'height', 'fps', 'frame_count'} or None if invalid
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

def validate_video_properties(video_path, standard_res, standard_fps, standard_len):
    """
    根据标准校验视频属性。
    Args:
        video_path: 视频路径
        standard_res: (H, W)
        standard_fps: float
        standard_len: int
    Returns:
        list[str]: 错误信息列表。为空表示校验通过。
    """
    props = inspect_video(video_path)
    if props is None:
        return ["Failed to open or corrupted video file."]
    
    errors = []
    
    # 1. Resolution Check (H, W)
    std_h, std_w = standard_res
    if (props['height'], props['width']) != (std_h, std_w):
        errors.append(f"Resolution mismatch: Got {props['width']}x{props['height']}, expected {std_w}x{std_h}")
        
    # 2. FPS Check (Allow small float error)
    if abs(props['fps'] - standard_fps) > 0.1:
        errors.append(f"FPS mismatch: Got {props['fps']:.2f}, expected {standard_fps}")
        
    # 3. Frame Count Check
    if props['frame_count'] != standard_len:
        errors.append(f"Frame count mismatch: Got {props['frame_count']}, expected {standard_len}")
        
    return errors