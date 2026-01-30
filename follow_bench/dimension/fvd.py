from typing import Any
import numpy as np
import logging
logger = logging.getLogger(__name__)


from ..dimension import DimensionEvaluator
from ..utils.pretrain import load_i3d
from ..utils.video_kit import extract_i3d_features
from .metric import FrechetVideoDistance


class FrechetVideoDistanceEvaluator(DimensionEvaluator):
    """
    FVD Evaluator (Refactored for Global Calculation).
    """

    def prepare(self):
        self.model = load_i3d(self.device)
        super().prepare()

    def compute(self, **kwargs: Any):
        """
        提取特征并收集到 global_cache，不返回单项分数。
        """
        video_gen = kwargs.get('tensor_gen')
        video_gt = kwargs.get('tensor_gt')
        video_id = kwargs.get('video_id')
        global_cache = kwargs.get('global_cache')

        if video_gen is None or video_gt is None or global_cache is None:
            return 0.0

        # 1. 提取生成视频特征
        gen_key = f"i3d_feat_gen_{video_id}"
        if gen_key in global_cache:
            feat_gen = global_cache[gen_key]
        else:
            feat_gen = extract_i3d_features(video_gen, self.model)
            global_cache[gen_key] = feat_gen

        # 2. 提取真值视频特征
        gt_key = f"i3d_feat_gt_{video_id}"
        if gt_key in global_cache:
            feat_gt = global_cache[gt_key]
        else:
            feat_gt = extract_i3d_features(video_gt, self.model)
            global_cache[gt_key] = feat_gt

        # 3. 收集到列表 (Accumulate)
        # 存入 CPU numpy array 以节省显存
        global_cache.setdefault('fvd_gen_list', []).append(feat_gen)
        global_cache.setdefault('fvd_gt_list', []).append(feat_gt)

        return 0.0

    @staticmethod
    def finalize_metric(global_cache: dict):
        """
        在所有视频遍历结束后调用，计算 Dataset-level FVD。
        """
        gen_list = global_cache.get('fvd_gen_list', [])
        gt_list = global_cache.get('fvd_gt_list', [])

        if not gen_list or not gt_list:
            logger.warning("No features collected for FVD.")
            return 0.0

        logger.info(f"Finalizing FVD with {len(gen_list)} samples...")
        
        # 堆叠特征 [N, D]
        feats_gen = np.stack(gen_list, axis=0)
        feats_gt = np.stack(gt_list, axis=0)
        
        # 计算
        score = FrechetVideoDistance(feats_gen, feats_gt)
        return float(score)