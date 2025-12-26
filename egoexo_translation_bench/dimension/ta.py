from torch import Tensor
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from typing import Any
import logging
logger = logging.getLogger(__name__)

from ..dimension import DimensionEvaluator
from .metric import TrajectoryAlignment
from ..utils.pretrain import get_detection_results, load_faster_rcnn

class TrajectoryAlignmentEvaluator(DimensionEvaluator):
    """Evaluator for trajectory alignment between generated and GT videos."""

    def prepare(self):
        """Load detector and call base prepare."""
        self.det = load_faster_rcnn(self.device)
        super().prepare()

    def compute(self, **kwargs: Any):
        """Compute Trajectory Alignment using detection/keypoint results.

        Expected kwargs: 'tensor_gen', 'tensor_gt', 'video_id', 'global_cache'.
        Returns the mean normalized trajectory alignment error (float).
        """
        video_gen = kwargs.get('tensor_gen')
        video_gt = kwargs.get('tensor_gt')
        video_id = kwargs.get('video_id')
        global_cache = kwargs.get('global_cache')
        
        # 1. Gen Detection (Shared Cache)
        gen_key = f"detection_gen_{video_id}"
        if global_cache is not None and gen_key in global_cache:
            det_gen = global_cache[gen_key]
        else:
            det_gen = get_detection_results(video_gen, self.det)
            if global_cache is not None: global_cache[gen_key] = det_gen
            
        # 2. GT Detection (Shared Cache)
        gt_key = f"detection_gt_{video_id}"
        if global_cache is not None and gt_key in global_cache:
            det_gt = global_cache[gt_key]
        else:
            det_gt = get_detection_results(video_gt, self.det)
            if global_cache is not None: global_cache[gt_key] = det_gt

        H, W = video_gen.shape[2], video_gen.shape[3]
        return float(TrajectoryAlignment(det_gen, det_gt, H, W))