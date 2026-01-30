from typing import Any, List
import logging
logger = logging.getLogger(__name__)

import torch

from ..dimension import DimensionEvaluator
from .metric import SourceControlConditionRecall
from ..utils.pretrain import (
    load_videomae_model,
    extract_videomae_features
)

class SourceControlConditionRecallEvaluator(DimensionEvaluator):
    """
    SCCR Evaluator.
    衡量生成模型在跨视角关联中的“源控制能力”。
    
    注意：这是一个 Global Metric。Compute 阶段只提取特征，不返回有意义的分数。
    必须在所有视频处理完毕后，对收集到的特征矩阵进行计算。
    """
    def prepare(self):
        """Load VideoMAE model."""
        self.model = load_videomae_model(self.device)
        super().prepare()

    def compute(self, **kwargs: Any):
        """
        Extract features for Gen and GT, cache them, and accumulate for global metric.
        """
        video_gen = kwargs.get('tensor_gen')
        video_gt = kwargs.get('tensor_gt')
        video_id = kwargs.get('video_id')
        global_cache = kwargs.get('global_cache')

        # === 1. 生成视频特征提取 (模仿 FVD 的缓存逻辑) ===
        gen_key = f"videomae_feat_gen_{video_id}"
        if global_cache is not None and gen_key in global_cache:
            feat_gen = global_cache[gen_key]
        else:
            # 使用工具函数提取特征 (纯 Tensor)
            feat_gen = extract_videomae_features(video_gen, self.model)
            if global_cache is not None:
                global_cache[gen_key] = feat_gen

        # === 2. 真值视频特征提取 (作为 Gallery) ===
        gt_key = f"videomae_feat_gt_{video_id}"
        if global_cache is not None and gt_key in global_cache:
            feat_gt = global_cache[gt_key]
        else:
            feat_gt = extract_videomae_features(video_gt, self.model)
            if global_cache is not None:
                global_cache[gt_key] = feat_gt

        # === 3. 数据收集 (Accumulation) ===
        # 将提取好的特征放入全局列表，供最后计算 Rank 使用
        if global_cache is not None:
            # 必须使用 cpu() 存储以免爆显存
            global_cache.setdefault('sccr_gen_list', []).append(feat_gen.cpu())
            global_cache.setdefault('sccr_gt_list', []).append(feat_gt.cpu())

        # 返回 0.0，因为单个视频无法计算排他性 Recall
        return 0.0

    @staticmethod
    def finalize_metric(global_cache: dict, device='cuda'):
        """
        计算 SCCR@K
        """
        gen_list = global_cache.get('sccr_gen_list', [])
        gt_list = global_cache.get('sccr_gt_list', [])
        
        if not gen_list: return {'SCCR@1': 0.0, 'SCCR@5': 0.0}

        # 堆叠 & 转 GPU
        gen_feats = torch.cat(gen_list, dim=0).to(device)
        gt_feats = torch.cat(gt_list, dim=0).to(device)
        
        return SourceControlConditionRecall(gen_feats, gt_feats)