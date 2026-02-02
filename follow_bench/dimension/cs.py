from typing import Any
import logging
logger = logging.getLogger(__name__)

import torch

from .metric import CameraStability
from . import DimensionEvaluator
# from ..utils.wham_wrapper import WhamWrapper

class CameraStabilityEvaluator(DimensionEvaluator):
    """
    Camera Trajectory Error (CTE) Evaluator powered by WHAM.
    
    Metric:
        RMSE of the Sim3-aligned camera trajectories.
        Uses WHAM (which includes DPVO + Human Priors) to estimate robust trajectories.
    """
    def prepare(self):
        # self.solver = WhamWrapper(device=self.device)
        super().prepare()

    def compute(self, tensor_gen: torch.Tensor, **kwargs: Any) -> float:
        """
        Compute CTE score.
        
        Args:
            tensor_gen: Generated video tensor
            kwargs: 'tensor_exo' (Ground Truth) is accepted but ignored 
                    as this is a blind stability metric.
        """
        # 调用 metric.py 中的函数
        return CameraStability(tensor_gen)
    
    def clear(self):
        if self.solver is not None:
            self.solver.clear()
            self.solver = None
        torch.cuda.empty_cache()
        super().clear()