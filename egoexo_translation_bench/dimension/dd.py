"""Dynamic-degree evaluator utilities (DD)."""

from typing import Any, Set
import logging
logger = logging.getLogger(__name__)

from ..dimension import DimensionEvaluator
from .metric import calculate_metrics_based_flow_model
from ..utils.pretrain import load_raft


class DynamicDegreeEvaluator(DimensionEvaluator):
    """Evaluate the Dynamic Degree (DD) dimension using an optical-flow model (RAFT).

    This evaluator loads a pretrained RAFT optical-flow model in `prepare`, then
    computes DD by calling `calculate_metrics_based_flow_model` with proper
    kwargs. The result returned is the 'dd' metric extracted from the
    result dict.
    """

    def prepare(self):  # noqa: ANN201, ANN101
        """Load RAFT model onto the configured device and call superclass prepare.

        Notes:
            - Uses `load_raft(self.device)` to obtain the model.
            - Does not change the evaluation pipeline; merely ensures the
              `self.model` attribute is set.
        """
        self.model = load_raft(self.device)
        super().prepare()

    def compute(self, **kwargs: Any):  # noqa: ANN201, ANN101
        """Compute DD for a single video.

        Args:
            **kwargs: Keyword arguments forwarded from the evaluation pipeline.
                Expected keys:
                    - 'tensor_gen': generator that yields frames (video frames).
                    - 'video_id': identifier for the current video.
                    - 'global_cache': shared cache dict for reused computations.
                    - 'metrics_to_compute': optional set of metric names to compute.

        Returns:
            The numeric DD value (results['dd']) produced by
            `calculate_metrics_based_flow_model`.
        """
        # parse kwargs
        video_gen = kwargs.get('tensor_gen')
        video_id = kwargs.get('video_id')
        global_cache = kwargs.get('global_cache')
        metrics_to_compute: Set[str] = kwargs.get('metrics_to_compute', {'dd'})  # 默认只算自己

        results = calculate_metrics_based_flow_model(
            gen_frames=video_gen,
            metrics_to_compute=metrics_to_compute,
            flow_model=self.model,
            device=self.device,
            video_id=video_id,
            global_cache=global_cache,
        )
        return float(results['dd'])
