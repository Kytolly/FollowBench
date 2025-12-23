from typing import Any

from . import DimensionEvaluator
from ..utils.pretrain import load_i3d
from ..utils.video_kit import extract_i3d_features
from .metric import FrechetVideoDistance


class FrechetVideoDistanceEvaluator(DimensionEvaluator):
    """Evaluator that computes Frechet Video Distance (FVD) between two videos.

    The evaluator uses a pretrained I3D model to extract per-video features and
    computes FVD on the resulting activations. Features are cached in
    `global_cache` if provided to avoid redundant computation.
    """

    def prepare(self):  # noqa: ANN201, ANN101
        """Load the I3D model onto the evaluator device and call superclass prepare.

        Notes:
            - Model is obtained via `load_i3d(self.device)`.
        """
        self.model = load_i3d(self.device)
        super().prepare()

    def compute(self, **kwargs: Any):  # noqa: ANN201, ANN101
        """Compute FVD between generated and ground-truth videos.

        Args:
            **kwargs: Keyword arguments forwarded from the evaluation pipeline.
                Expected keys:
                    - 'tensor_gen': frames generator for generated video.
                    - 'tensor_gt': frames generator for ground-truth video.
                    - 'video_id': identifier for the current video.
                    - 'global_cache': optional dict-like cache for intermediate results.

        Returns:
            A `FrechetVideoDistance` object constructed from the extracted features.
        """
        video_gen = kwargs.get('tensor_gen')
        video_gt = kwargs.get('tensor_gt')
        video_id = kwargs.get('video_id')
        global_cache = kwargs.get('global_cache')

        gen_key = f"i3d_feat_gen_{video_id}"
        if global_cache is not None and gen_key in global_cache:
            # cache hits
            feat_gen = global_cache[gen_key]
        else:  # cache not hits
            feat_gen = extract_i3d_features(video_gen, self.model)
            if global_cache is not None:
                global_cache[gen_key] = feat_gen

        gt_key = f"i3d_feat_gt_{video_id}"
        if global_cache is not None and gt_key in global_cache:
            # cache hits
            feat_gt = global_cache[gt_key]
        else:  # cache not hits
            feat_gt = extract_i3d_features(video_gt, self.model)
            if global_cache is not None:
                global_cache[gt_key] = feat_gt

        return FrechetVideoDistance(feat_gen, feat_gt)
