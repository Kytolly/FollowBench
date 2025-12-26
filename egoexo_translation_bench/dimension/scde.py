from typing import Any
import logging
logger = logging.getLogger(__name__)

import torch

from ..dimension import DimensionEvaluator
from .metric import SubjectCameraDistanceError
from ..utils.wham_wrapper import WhamWrapper
from ..utils.video_kit import load_video_to_device

class SCDE(DimensionEvaluator):
    """
    Subject-Camera Distance Error (SCDE) Evaluator.
    
    Metric:
        Measures the accuracy and stability of the follow distance.
        Uses HMR (WHAM) to recover depth, aligns scale using Least Squares,
        and computes RMSE.
    """
    def prepare(self):
        """Prepare the HMR solver."""
        self.solver = WhamWrapper(device=self.device)
        super().prepare()

    def compute(self, **kwargs: Any):
        """
        Execute calculation.
        Expected kwargs:
            tensor_gen: [T, C, H, W]
            tensor_gt:  [T, C, H, W]
        """
        prediction = kwargs.get('tensor_gen')
        target = kwargs.get('tensor_gt')

        if prediction is None or target is None:
            logger.warning("SCDE inputs missing.")
            return 0.0
        
        if self.solver is None:
            self.prepare()

        # 1. 提取深度序列 (Extract Depth Sequences)
        # 对称处理：Gen 和 GT 都过一遍模型，消除模型本身的 bias
        try:
            # [T] vectors
            z_gen = self.solver.extract_camera_distance(prediction)
            z_gt = self.solver.extract_camera_distance(target)
        except Exception as e:
            self.logger.error(f"HMR Extraction Failed: {e}")
            return 0.0

        # 2. 长度对齐
        min_len = min(len(z_gen), len(z_gt))
        if min_len < 2:
            return 0.0
            
        z_gen = z_gen[:min_len]
        z_gt = z_gt[:min_len]

        # 3. 计算对齐后的误差 (Math Calculation)
        # 内部包含 Scale Alignment: z_gt approx s * z_gen
        score = SubjectCameraDistanceError(z_gen, z_gt)
        
        return score

    def clear(self):
        """Free VRAM."""
        if self.solver is not None:
            self.solver.clear()
            self.solver = None
        torch.cuda.empty_cache()