from typing import Any
import logging
logger = logging.getLogger(__name__)

import torch

from .metric import CameraTrajectoryError
from ..dimension import DimensionEvaluator
from ..utils.wham_wrapper import WhamWrapper

class CameraTrajectoryErrorEvaluator(DimensionEvaluator):
    """
    Camera Trajectory Error (CTE) Evaluator powered by WHAM.
    
    Metric:
        RMSE of the Sim3-aligned camera trajectories.
        Uses WHAM (which includes DPVO + Human Priors) to estimate robust trajectories.
    """
    def prepare(self):
        self.solver = WhamWrapper(device=self.device)
        super().prepare()

    def compute(self, **kwargs: Any):
        """
        Args:
            tensor_gen: [T, C, H, W]
            tensor_gt:  [T, C, H, W]
        """
        prediction = kwargs.get('tensor_gen')
        target = kwargs.get('tensor_gt')

        if prediction is None or target is None:
            return 0.0
        
        if self.solver is None:
            self.prepare()

        # 1. 提取轨迹 (使用 WHAM)
        # Gen 和 GT 分别通过 WHAM 得到各自世界坐标系下的轨迹
        traj_gen = self.solver.extract_trajectory(prediction) # [T, 3]
        traj_gt = self.solver.extract_trajectory(target)      # [T, 3]

        # 如果提取失败 (全0)，返回错误值 (通常 CTE 越大越差，这里返回 None 或 0 需由外层逻辑决定)
        if traj_gen.abs().sum() == 0 or traj_gt.abs().sum() == 0:
            logger.warning("Invalid trajectory extracted (no human detected?). Skipping.")
            return None

        # 2. 长度对齐
        min_len = min(len(traj_gen), len(traj_gt))
        if min_len < 4:
            return 0.0
            
        traj_gen = traj_gen[:min_len]
        traj_gt = traj_gt[:min_len]

        # 3. Sim3 对齐与误差计算
        # 尽管 WHAM 利用人体先验修正了部分尺度，但 Gen 和 GT 之间仍可能存在
        # 全局坐标系定义的不同 (原点、旋转方向)，因此 Sim3 (Umeyama) 依然是必须的。
        return CameraTrajectoryError(traj_gen, traj_gt)
    
    def clear(self):
        if self.solver is not None:
            self.solver.clear()
            self.solver = None
        torch.cuda.empty_cache()
        super().clear()