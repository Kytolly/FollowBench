"""Camera-centering evaluator utilities."""

from typing import Any
from torch import Tensor

from . import DimensionEvaluator
from .metric import CameraCenteringError
from ..utils.pretrain import get_detection_results, load_faster_rcnn


class CameraCenteringErrorEvaluator(DimensionEvaluator):
    """Evaluator for Camera Centering Error (CCE).

    Uses a person detector to compute per-frame detections and measures how
    centrally the detected person is positioned in each frame.
    """

    def prepare(self) -> None:  # noqa: ANN201, ANN101
        """Load the detector and call base prepare."""
        self.model = load_faster_rcnn(self.device)
        super().prepare()

    def compute(self, **kwargs: Any) -> float:  # noqa: ANN201, ANN101
        """Compute CCE for generated video.

        Expected kwargs: 'tensor_gen', 'video_id', 'global_cache'. Returns a float
        score in [0.0, 1.0], where higher indicates worse centering.
        """
        # parse kwargs
        video_gen: Tensor = kwargs.get('tensor_gen')
        video_id = kwargs.get('video_id')
        global_cache = kwargs.get('global_cache')

        cache_key = f"detection_gen_{video_id}"
        if global_cache is not None and cache_key in global_cache:
            # cache hits
            detections = global_cache[cache_key]
        else:  # cache not hits
            detections = get_detection_results(video_gen, self.model)
            if global_cache is not None:
                global_cache[cache_key] = detections

        H, W = video_gen.shape[2], video_gen.shape[3]
        return float(CameraCenteringError(detections, H, W))
