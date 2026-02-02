import os
import shutil
import tempfile
import logging
import torch
import numpy as np
import cv2
import contextlib

# 尝试导入 WHAM_API
try:
    from wham.wham_api import WHAM_API
except ImportError:
    try:
        from wham.api.wrapper import WHAM_API
    except ImportError:
        pass

# 尝试定位 WHAM 库的根目录
try:
    import wham
    WHAM_ROOT = os.path.dirname(os.path.dirname(wham.__file__))
except Exception:
    WHAM_ROOT = None

logger = logging.getLogger(__name__)

@contextlib.contextmanager
def temporary_chdir(path):
    """
    上下文管理器：临时切换工作目录，退出时自动切回原目录。
    """
    prev_cwd = os.getcwd()
    if path is not None and os.path.exists(path):
        os.chdir(path)
        try:
            yield
        finally:
            os.chdir(prev_cwd)
    else:
        yield

class WhamWrapper():
    """
    Wrapper for WHAM API to handle tensor-to-video conversion and geometry extraction.
    """
    def __init__(self, device='cuda', checkpoint_path=None):
        self.device = device
        
        if 'WHAM_API' in globals():
            try:
                logger.info(f"Initializing WHAM_API with root context: {WHAM_ROOT}")
                
                # [FIX] 1. 切换目录以找到配置文件
                # [FIX] 2. 临时 Monkey-patch torch.load 以绕过 PyTorch 2.6+ 的安全检查
                with temporary_chdir(WHAM_ROOT):
                    self._init_model_safely()
                    
                logger.info("WHAM_API initialized successfully.")
            except Exception as e:
                logger.error(f"Failed to initialize WHAM_API: {e}")
                if WHAM_ROOT:
                    config_path = os.path.join(WHAM_ROOT, 'configs/yamls/demo.yaml')
                    logger.error(f"Debug: Expected config path: {config_path}, Exists: {os.path.exists(config_path)}")
                self.model = None
        else:
            self.model = None
            logger.warning("WHAM_API class not found. Please ensure 'wham' is installed/in python path.")

    def _init_model_safely(self):
        """
        Helper to initialize WHAM_API while patching torch.load for legacy checkpoints.
        """
        original_load = torch.load

        def unsafe_load(*args, **kwargs):
            # 强制设置 weights_only=False 以支持旧版 Checkpoint (numpy scalars 等)
            if 'weights_only' not in kwargs:
                kwargs['weights_only'] = False
            return original_load(*args, **kwargs)

        try:
            # 应用补丁
            torch.load = unsafe_load
            self.model = WHAM_API()
        finally:
            # 恢复原始函数，避免影响其他模块
            torch.load = original_load

    def _save_tensor_to_video(self, video_tensor, output_path):
        """
        Helper: Save [T, C, H, W] tensor (0-1) to an mp4 file for WHAM input.
        """
        if video_tensor.min() < 0: 
            video_tensor = (video_tensor + 1.0) / 2.0
            
        video_np = video_tensor.permute(0, 2, 3, 1).detach().cpu().numpy()
        video_np = (np.clip(video_np, 0, 1) * 255).astype(np.uint8)
        
        T, H, W, C = video_np.shape
        fps = 24 
        
        try:
            writer = cv2.VideoWriter(
                output_path, 
                cv2.VideoWriter_fourcc(*'mp4v'), 
                fps, 
                (W, H)
            )
            for frame in video_np:
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            writer.release()
        except Exception as e:
            logger.error(f"Failed to save temp video: {e}")
            return False
        return True

    def extract_full_geometry(self, video_tensor: torch.Tensor):
        """
        提取 CSHA 所需的几何信息。
        通过 run_global=False 强制在相机坐标系下计算。
        """
        if self.model is None:
            return None
            
        # 1. 创建临时目录处理文件 I/O
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_vid_path = os.path.join(temp_dir, "temp_input.mp4")
            temp_out_dir = os.path.join(temp_dir, "output")
            
            # 保存 Tensor 为视频文件
            if not self._save_tensor_to_video(video_tensor, temp_vid_path):
                return None
            
            # 2. 调用 WHAM API (同样需要切换目录环境)
            try:
                with temporary_chdir(WHAM_ROOT):
                    # 同样需要 patch torch.load 吗？通常推理不需要 load checkpoint，
                    # 但为了保险起见，如果内部有 lazy load，可以复用 patch 逻辑。
                    # 这里假设推理阶段不再 load 权重。
                    results, _, _ = self.model(
                        video=temp_vid_path,
                        output_dir=temp_out_dir,
                        run_global=False, 
                        visualize=False
                    )
            except Exception as e:
                logger.error(f"WHAM inference failed: {e}")
                return None
                
            if not results:
                return None
            
            # 3. 解析结果
            try:
                subject_id = list(results.keys())[0]
                data = results[subject_id]
                
                # [T, 3]
                subj_pos = torch.tensor(data['trans_world'], device=self.device).float()
                # [T, 3] Axis-Angle
                subj_orient = torch.tensor(data['poses_root_world'], device=self.device).float()
                
                # 构造静态相机坐标 (0,0,0)
                cam_pos = torch.zeros_like(subj_pos) 
                
                return {
                    'cam_pos': cam_pos,
                    'subj_pos': subj_pos,
                    'subj_orient': subj_orient
                }
                
            except Exception as e:
                logger.error(f"Error parsing WHAM results: {e}")
                return None

    def clear(self):
        if self.model is not None:
            del self.model
            self.model = None
        torch.cuda.empty_cache()