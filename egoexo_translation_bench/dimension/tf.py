"""Evaluators related to temporal/flow-based metrics.

This module provides a RAFT-based Temporal Flickering evaluator which
computes a photometric flickering score by warping using estimated optical
flow.
"""

from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
from typing import Any

from . import DimensionEvaluator
from .metric import calculate_metrics_based_flow_model
from ..utils.pretrain import load_raft

class TemporalFlickeringEvaluator(DimensionEvaluator):
    """Evaluator for Temporal Flickering (TF) using optical-flow warping."""

    def prepare(self) -> None:
        """Load RAFT model and call base prepare."""
        self.model = load_raft(self.device)
        super().prepare()

    def compute(self, **kwargs: Any) -> float:
        """Compute TF metric for the given video using cached or computed flows.

        Expected kwargs: 'tensor_gen', 'video_id', 'global_cache', optional
        'metrics_to_compute'. Returns the 'tf' value from
        `calculate_metrics_based_flow_model`.
        """
        # parse kwargs
        video_gen = kwargs.get('tensor_gen')
        video_id = kwargs.get('video_id')
        global_cache = kwargs.get('global_cache')
        metrics_to_compute = kwargs.get('metrics_to_compute', {'tf'}) # 默认只算自己
        
        results = calculate_metrics_based_flow_model(
            gen_frames=video_gen,
            metrics_to_compute=metrics_to_compute,
            flow_model=self.model,
            device=self.device,
            video_id=video_id,
            global_cache=global_cache
        )
        return float(results['tf'])