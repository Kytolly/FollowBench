from typing import Any, Union
import logging
logger = logging.getLogger(__name__)

from torch import Tensor

from ..dimension import DimensionEvaluator
from ..utils.pretrain import (
    load_depth_anything, 
    infer_depth
)
from .metric import SideBySideDepthConsistency

class SideBySideDepthConsistencyEvaluator(DimensionEvaluator):
    def prepare(self):
        self.model, self.processor = load_depth_anything(self.device)
        super().prepare()
    
    def compute(self, **kwargs: Any):  # noqa: ANN201, ANN101
        # 1. Load Data
        tensor_gen = kwargs.get('tensor_gen')
        tensor_gt = kwargs.get('tensor_gt')
        
        # 2. Align Lengths
        min_len = min(len(tensor_gen), len(tensor_gt))
        tensor_gen = tensor_gen[:min_len]
        tensor_gt = tensor_gt[:min_len]

        # 3. Extract Depth (Proxy Model)
        depth_pred = infer_depth(tensor_gen)
        depth_gt = infer_depth(tensor_gt)

        # 4. Calculate Metric (Affine-Invariant Error)
        # We calculate frame-wise scores and average them
        scores = []
        for i in range(min_len):
            score = SideBySideDepthConsistency(depth_pred[i], depth_gt[i])
            scores.append(score)
            
        return sum(scores) / len(scores)
     
    def clear(self):
        """
        Free VRAM resources. 
        Called by BenchRouter after finishing all cases for this metric.
        """
        del self.model
        del self.processor
        super().clear()