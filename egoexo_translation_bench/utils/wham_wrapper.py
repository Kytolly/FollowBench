import numpy as np
import logging
logger = logging.getLogger(__name__)

import torch

from wham.api.wrapper import WHAM_API

from ..configs import CONFIG
WHAM_VIT_H_CHECKPOINT_PATH = CONFIG.models.wham_vit_h

class WhamWrapper():
    """
    3D Human Mesh Recovery Solver (Wrapper for WHAM/4DHumans).
    用于提取 Subject-Camera Distance (Z-translation).
    """
    def __init__(self, device='cuda', checkpoint_path=WHAM_VIT_H_CHECKPOINT_PATH):
        self.device = device
        self.checkpoint = checkpoint_path
        self.model = None
        if WHAM_API is None:
            self.logger.warning("WHAM library not found.")

    def _init_model(self):
        """Lazy load WHAM model"""
        if self.model is None and WHAM_API is not None:
            logger.info(f"Loading WHAM model from {self.checkpoint}...")
            # 初始化 WHAM Pipeline
            self.model = WHAM_API(checkpoint=self.checkpoint, device=self.device)
    
    def _run_inference(self, video_tensor: torch.Tensor):
        """
        内部通用推理函数。
        Returns raw WHAM output dictionary.
        """
        self._init_model()
        
        # 1. 转换数据格式
        # WHAM API 通常接受 numpy image list [H, W, 3] in 0-255
        if video_tensor.min() < 0:
            video_tensor = (video_tensor + 1.0) / 2.0
        
        # [T, C, H, W] -> [T, H, W, C] numpy
        video_np = video_tensor.permute(0, 2, 3, 1).cpu().numpy()
        video_np = (video_np * 255).astype(np.uint8)
        
        # 2. 运行 WHAM (包含 Detection -> DPVO -> Optimization)
        # 注意: 如果 WHAM 检测不到人，可能会返回 None 或空字典，需要处理
        try:
            # estimate_video 内部会处理 batching 和 tracking
            results = self.model.estimate_video(video_np)
            return results
        except Exception as e:
            self.logger.error(f"WHAM Inference Error: {e}")
            return None
        
    def extract_camera_distance(self, video_tensor: torch.Tensor):
        """
        输入视频张量，输出每一帧的人机距离 Z。
        
        Args:
            video_tensor: [T, C, H, W], normalized [0, 1]
            
        Returns:
            depth_seq: [T] tensor containing Z-values.
        """
        if WHAM_API is None:
            return torch.zeros(video_tensor.shape[0], device=self.device)

        self._init_model()
        
        # 1. 预处理: Tensor -> Numpy (WHAM 通常接收 CV2 images 或 path)
        # WHAM 的处理流程通常包含: SLAM/VIO -> Detection -> HMR -> Optimization
        # 为了高效，我们这里假设直接输入 Tensor 供模型推理
        
        # 假设 video_tensor 已经在 [0, 1]
        T, C, H, W = video_tensor.shape
        
        # 将 Tensor 转换为 WHAM 需要的格式 (通常是 batch images)
        # 注意：这里简化了流程，实际 WHAM 需要 bbox。
        # 如果视频是 Ego2Exo 生成的，人物通常在画面中心。
        
        with torch.no_grad():
            # 调用 WHAM 推理
            # outputs = self.model(video_tensor) 
            # outputs 包含 'cam_trans' (camera translation in world frame or relative)
            # 或 'pred_cam' (weak perspective [s, tx, ty])
            
            # --- 模拟 WHAM 输出 ---
            # 真实 WHAM 输出通常是 Global Trajectory。
            # 我们需要的是 Camera 系下 Root 的 Z 坐标，即 T_cam_to_root 的 Z 分量。
            # 若 WHAM 输出的是 World 系下的 Camera 和 Body：
            # Z = (Body_Pos - Cam_Pos) dot Cam_Forward_Vector
            
            # 简化：假设 model 直接返回估计的相机坐标系下的平移向量 pred_cam_t [T, 3]
            results = self.model.estimate_video(video_tensor) 
            pred_cam_t = results['cam_trans'] # [T, 3]
            
            # 提取 Z 分量 (深度)
            depth_seq = pred_cam_t[:, 2] # [T]
            
            # 如果是弱透视投影 (s, tx, ty)，深度 Z 约为 f / s
            # depth_seq = 1.0 / (results['pred_cam'][:, 0] + 1e-5)

        return depth_seq
    
    def extract_trajectory(self, video_tensor: torch.Tensor):
        """
        For CTE: Extract full camera translation trajectory [T, 3].
        """
        results = self._run_inference(video_tensor)
        if results is None or 'cam_trans' not in results:
            self.logger.warning("Failed to extract trajectory.")
            return torch.zeros((len(video_tensor), 3), device=self.device)
        
        # WHAM 输出的 cam_trans 已经是优化后的世界坐标系下的相机位置
        # Shape: [T, 3]
        traj = torch.tensor(results['cam_trans'], device=self.device).float()
        return traj
    
    def extract_full_geometry(self, video_tensor: torch.Tensor):
        """
        提取用于 CSHA 计算的全套 3D 几何信息。
        
        Returns:
            dict or None: {
                'cam_pos': [T, 3],      # 相机世界坐标
                'subj_pos': [T, 3],     # 主角 Root 世界坐标
                'subj_orient': [T, 3]   # 主角 Root 旋转 (Axis-Angle)
            }
        """
        results = self._run_inference(video_tensor)
        if results is None:
            return None
            
        # WHAM 输出解析
        # results 通常包含:
        # 'cam_trans': [T, 3] -> 相机在世界系下的位置 (WHAM 将第一帧定为世界原点或由 SLAM 确定)
        # 'pose_world': [T, 72] -> 包含 Global Orient (前3位) 和 Body Pose
        # 'trans_world': [T, 3] -> 主角 Root 在世界系下的平移
        
        if 'pose_world' not in results or 'trans_world' not in results:
            self.logger.error("WHAM output missing 'pose_world' or 'trans_world'")
            return None

        # 1. 相机位置
        # 注意: WHAM 的 cam_trans 有时表示 "Camera Translation in World"
        cam_pos = torch.tensor(results['cam_trans'], device=self.device).float()
        
        # 2. 主角位置
        subj_pos = torch.tensor(results['trans_world'], device=self.device).float()
        
        # 3. 主角朝向 (Global Orient, Axis-Angle)
        # SMPL pose 的前 3 个值是 root orientation
        poses = torch.tensor(results['pose_world'], device=self.device).float()
        subj_orient = poses[:, :3] # [T, 3] Axis-Angle
        
        return {
            'cam_pos': cam_pos,
            'subj_pos': subj_pos,
            'subj_orient': subj_orient
        }
        
    def clear(self):
        if self.model is not None:
            del self.model
            self.model = None
        torch.cuda.empty_cache()