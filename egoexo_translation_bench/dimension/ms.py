"""Motion-smoothness related evaluator wrapping RAFT-based flows."""

from typing import Any, Set

from . import DimensionEvaluator
from .metric import calculate_metrics_based_flow_model
from ..utils.pretrain import load_raft


class MotionSmoothnessEvaluator(DimensionEvaluator):
    """Evaluator for Motion Smoothness (MS) using a RAFT flow model."""

    def prepare(self) -> None:  # noqa: ANN201, ANN101
        """Load RAFT model and call base prepare."""
        self.model = load_raft(self.device)
        super().prepare()

    def compute(self, **kwargs: Any) -> float:  # noqa: ANN201, ANN101
        """Compute the MS metric for the generated video.

        Expected kwargs: 'tensor_gen', 'video_id', 'global_cache', optional
        'metrics_to_compute'. Returns the 'ms' entry from
        `calculate_metrics_based_flow_model`.
        """
        # parse kwargs
        video_gen = kwargs.get('tensor_gen')
        video_id = kwargs.get('video_id')
        global_cache = kwargs.get('global_cache')
        metrics_to_compute: Set[str] = kwargs.get('metrics_to_compute', {'ms'})  # 默认只算自己

        results = calculate_metrics_based_flow_model(
            gen_frames=video_gen,
            metrics_to_compute=metrics_to_compute,
            flow_model=self.model,
            device=self.device,
            video_id=video_id,
            global_cache=global_cache,
        )
        return float(results['ms'])
