"""Viewpoint- and detection-based evaluator utilities."""

from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from typing import Any

from ..utils.pretrain import get_detection_results, load_faster_rcnn
from .metric import ViewpointValidity
from . import DimensionEvaluator

class ViewpointValidityEvaluator(DimensionEvaluator):
    """Evaluator for viewpoint validity measuring how often a person is detected."""

    def prepare(self) -> None:
        """Load the detector and call base prepare."""
        self.model = load_faster_rcnn(self.device)
        super().prepare()

    def compute(self, **kwargs: Any) -> float:
        """Compute viewpoint validity over sampled frames.

        Expected kwargs: 'tensor_gen', 'video_id', 'global_cache'. The evaluator
        samples frames (every 5th frame) to speed up detection and then computes
        the fraction of frames with person detections via `ViewpointValidity`.

        Returns a float in [0.0, 1.0].
        """
        # parse kwargs
        video_gen = kwargs.get('tensor_gen')
        video_id = kwargs.get('video_id')
        global_cache = kwargs.get('global_cache')
        
        cache_key = f"detection_gen_{video_id}"
        if global_cache is not None and cache_key in global_cache:
            # cache hits
            detections = global_cache[cache_key]
        else: # cache not hits
            sampled_video = video_gen[::5]
            detections = get_detection_results(sampled_video, self.model)
            if global_cache is not None:
                global_cache[cache_key] = detections

        return float(ViewpointValidity(detections))