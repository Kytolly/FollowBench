import numpy as np

from . import DimensionEvaluator
from src.utils.pretrain import load_i3d
from src.utils.video_kit import extract_i3d_features
from .metric import FrechetVideoDistance

class FrechetVideoDistanceEvaluator(DimensionEvaluator):
    def prepare(self):
        self.model = load_i3d(self.device)

    def compute(self, **kwargs):
        video_gen = kwargs.get('tensor_gen')
        video_gt = kwargs.get('tensor_gt')
        video_id = kwargs.get('video_id')
        global_cache = kwargs.get('global_cache')

        gen_key = f"i3d_feat_gen_{video_id}"
        if global_cache is not None and gen_key in global_cache:
            # cache hits
            feat_gen = global_cache[gen_key]
        else: # cache not hits
            feat_gen = extract_i3d_features(video_gen, self.model)
            if global_cache is not None: global_cache[gen_key] = feat_gen

        gt_key = f"i3d_feat_gt_{video_id}"
        if global_cache is not None and gt_key in global_cache:
            # cache hits
            feat_gt = global_cache[gt_key]
        else: # cache not hits
            feat_gt = extract_i3d_features(video_gt, self.model)
            if global_cache is not None: global_cache[gt_key] = feat_gt

        return FrechetVideoDistance(feat_gen, feat_gt)