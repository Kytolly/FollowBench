from typing import Any
from torch import Tensor

from . import DimensionEvaluator
from .metric import HumanActionAlignment
from ..utils.pretrain import get_keypoint_results, load_keypoint_rcnn


class HumanActionAlignmentEvaluator(DimensionEvaluator):
    """Evaluator for Human Action Alignment (HAA).

    This evaluator uses a keypoint detection model (Keypoint R-CNN) to extract
    keypoint results from generated and ground-truth videos and computes the
    HAA metric using `HumanActionAlignment`.
    """

    def prepare(self) -> None:  # noqa: ANN201, ANN101
        """Load the keypoint detection model and call superclass prepare.

        Notes:
            - Model is obtained via `load_keypoint_rcnn(self.device)`.
        """
        self.model = load_keypoint_rcnn(self.device)
        super().prepare()

    def compute(self, **kwargs: Any) -> float:  # noqa: ANN201, ANN101
        """Compute HAA for a single video pair.

        Args:
            **kwargs: Keyword arguments forwarded from the evaluation pipeline.
                Expected keys:
                    - 'tensor_gen': Tensor frames for generated video.
                    - 'tensor_gt': Tensor frames for ground-truth video.
                    - 'video_id': identifier for the current video.
                    - 'global_cache': optional cache for intermediate results.

        Returns:
            The result of `HumanActionAlignment(kp_gen, kp_gt, H, W)`.
        """
        video_gen: Tensor = kwargs.get('tensor_gen')
        video_gt: Tensor = kwargs.get('tensor_gt')
        video_id = kwargs.get('video_id')
        global_cache = kwargs.get('global_cache')

        gen_key = f"keypoint_gen_{video_id}"
        if global_cache is not None and gen_key in global_cache:
            # cache hits
            kp_gen = global_cache[gen_key]
        else:  # cache not hits
            kp_gen = get_keypoint_results(video_gen, self.model)
            if global_cache is not None:
                global_cache[gen_key] = kp_gen

        gt_key = f"keypoint_gt_{video_id}"
        if global_cache is not None and gt_key in global_cache:
            # cache hits
            kp_gt = global_cache[gt_key]
        else:  # cache not hits
            kp_gt = get_keypoint_results(video_gt, self.model)
            if global_cache is not None:
                global_cache[gt_key] = kp_gt

        H, W = video_gen.shape[2], video_gen.shape[3]
        return float(HumanActionAlignment(kp_gen, kp_gt, H, W))
