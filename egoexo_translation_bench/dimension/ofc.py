"""Optical-flow based evaluators utilities."""

from typing import Any, Set

from . import DimensionEvaluator
from .metric import calculate_metrics_based_flow_model
from ..utils.pretrain import load_raft


class OpticalFlowCorrelationEvaluator(DimensionEvaluator):
    """Evaluator for Optical Flow Correlation (OFC) between generated and GT videos."""

    def prepare(self) -> None:  # noqa: ANN201, ANN101
        """Load RAFT model and call base prepare."""
        self.model = load_raft(self.device)
        super().prepare()

    def compute(self, **kwargs: Any) -> float:  # noqa: ANN201, ANN101
        """Compute OFC for a generated video, optionally using cached flows.

        Expected kwargs: 'tensor_gen', 'gt_frames', 'video_id', 'global_cache'.
        Returns the 'ofc' value from `calculate_metrics_based_flow_model`.
        """
        # parse kwargs
        video_gen = kwargs.get('tensor_gen')
        video_id = kwargs.get('video_id')
        global_cache = kwargs.get('global_cache')
        metrics_to_compute: Set[str] = kwargs.get('metrics_to_compute', {'ofc'})  # 默认只算自己

        results = calculate_metrics_based_flow_model(
            gen_frames=video_gen,
            metrics_to_compute=metrics_to_compute,
            flow_model=self.model,
            device=self.device,
            video_id=video_id,
            global_cache=global_cache,
        )
        return float(results['ofc'])
