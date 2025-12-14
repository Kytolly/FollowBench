import cv2
import numpy as np
from scipy.stats import pearsonr

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