"""Background CLIP-based evaluator utilities."""

from PIL import Image
from typing import Any

from transformers import CLIPProcessor, CLIPModel
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights

from . import DimensionEvaluator
from .metric import BackgroundSemanticConsistency
from ..utils.pretrain import get_detection_results, load_clip, load_faster_rcnn

class BackgroundSemanticConsistencyEvaluator(DimensionEvaluator):
    """Evaluator for Background Semantic Consistency (BSC).

    Uses CLIP to compare frame backgrounds to a reference image; person regions
    are masked out using a detector to focus on background semantics.
    """

    def prepare(self) -> None:
        """Load CLIP and detector models and call base prepare."""
        self.clip, self.proc = load_clip(self.device)
        self.det = load_faster_rcnn(self.device)
        super().prepare()

    def compute(self, **kwargs: Any) -> float:
        """Compute background semantic consistency for a generated video.

        Expected kwargs: 'tensor_gen', 'pillow_ref', 'video_id', 'global_cache'
        Returns the mean CLIP similarity between masked-frame backgrounds and
        the reference image embedding.
        """
        # parse kwargs
        video_gen = kwargs.get('tensor_gen')
        pillow_ref: Image = kwargs.get('pillow_ref')
        video_id = kwargs.get('video_id')
        global_cache = kwargs.get('global_cache')
        

        cache_key = f"detection_gen_{video_id}"
        if global_cache is not None and cache_key in global_cache:
            # cache hits
            detections = global_cache[cache_key]
        else: # cache not hits
            detections = get_detection_results(video_gen, self.det)
            if global_cache is not None:
                global_cache[cache_key] = detections

        return float(BackgroundSemanticConsistency(
            ref_img_pil=pillow_ref,
            video_gen=video_gen,
            clip_model=self.clip,
            clip_proc=self.proc,
            detection_results=detections,
            device=self.device
        ))