import cv2
import numpy as np
from scipy.stats import pearsonr
import torch

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

def load_video_to_gpu(video_path, device='cuda', target_size=None):
    """读取视频并直接转换为 GPU Tensor [T, C, H, W]"""
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if target_size is not None:
            # 如果显存紧张，可以在这里 resize成 (244, 244)
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