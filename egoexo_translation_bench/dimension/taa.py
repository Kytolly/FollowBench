# egoexo_translation_bench/dimension/taa.py

from typing import Any
import torch
import logging
logger = logging.getLogger(__name__)

from ..dimension import DimensionEvaluator
from .metric import TemporalAttentionAlignment
from ..utils.pretrain import load_videomae_model, extract_videomae_sequence

class TemporalAttentionAlignmentEvaluator(DimensionEvaluator):
    """
    TAA Evaluator: Measures temporal alignment/synchronization.
    
    Uses VideoMAE to extract frame-wise (temporal) features and computes
    the diagonal trace of the attention matrix between Gen and GT.
    """

    def prepare(self):
        self.model = load_videomae_model(self.device)
        super().prepare()

    def compute(self, **kwargs: Any):
        video_gen = kwargs.get('tensor_gen')
        video_gt = kwargs.get('tensor_gt')
        video_id = kwargs.get('video_id')
        global_cache = kwargs.get('global_cache')

        if video_gen is None or video_gt is None:
            return 0.0

        # === 1. 生成视频时序特征 ===
        # 缓存 Key 使用 'seq' 以区别于 SCCR 可能使用的全局特征
        gen_key = f"videomae_seq_gen_{video_id}"
        
        if global_cache is not None and gen_key in global_cache:
            seq_gen = global_cache[gen_key]
        else:
            seq_gen = extract_videomae_sequence(video_gen, self.model)
            if global_cache is not None:
                global_cache[gen_key] = seq_gen.cpu() # Cache on CPU

        # === 2. 真值视频时序特征 ===
        gt_key = f"videomae_seq_gt_{video_id}"
        
        if global_cache is not None and gt_key in global_cache:
            seq_gt = global_cache[gt_key]
        else:
            seq_gt = extract_videomae_sequence(video_gt, self.model)
            if global_cache is not None:
                global_cache[gt_key] = seq_gt.cpu() # Cache on CPU
        
        # === 3. 计算 TAA ===
        # 确保在同一 Device 计算
        seq_gen = seq_gen.to(self.device)
        seq_gt = seq_gt.to(self.device)
        
        score = TemporalAttentionAlignment(seq_gen, seq_gt)
        
        return score