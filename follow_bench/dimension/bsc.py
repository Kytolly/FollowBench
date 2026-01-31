"""Background CLIP-based evaluator utilities."""

from PIL import Image
from typing import Any
import logging
logger = logging.getLogger(__name__)

from transformers import CLIPProcessor, CLIPModel
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights

from ..dimension import DimensionEvaluator
from .metric import BackgroundSemanticContinuity
from ..utils.pretrain import get_detection_results, load_clip, load_faster_rcnn

class BackgroundSemanticConsistencyEvaluator(DimensionEvaluator):
    """Evaluator for Background Semantic Consistency (BSC).

    Uses CLIP to compare frame backgrounds to a reference image; person regions
    are masked out using a detector to focus on background semantics.
    """

    def prepare(self):
        """Load CLIP and detector models and call base prepare."""
        self.clip, self.proc = load_clip(self.device)
        self.det = load_faster_rcnn(self.device)
        super().prepare()

    def compute(self, **kwargs: Any):
        """Compute background continuity.

        Expected kwargs: 'tensor_gen', 'video_id', 'global_cache'
        Note: 'pillow_ref' is no longer required.
        """
        video_gen = kwargs.get('tensor_gen')
        video_id = kwargs.get('video_id')
        global_cache = kwargs.get('global_cache')
        
        # 1. Detection (带缓存)
        cache_key = f"detection_gen_{video_id}"
        if global_cache is not None and cache_key in global_cache:
            detections = global_cache[cache_key]
        else:
            detections = get_detection_results(video_gen, self.det)
            if global_cache is not None:
                global_cache[cache_key] = detections

        # 2. Compute Metric 
        return float(BackgroundSemanticContinuity(
            video_gen=video_gen,
            clip_model=self.clip,
            detection_results=detections,
            device=self.device,
            batch_size=64
        ))