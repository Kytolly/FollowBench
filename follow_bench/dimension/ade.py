from typing import Any
import logging
import torch

from ..dimension import DimensionEvaluator
from ..dimension.metric import AverageDisplacementError
from ..utils.pretrain import load_yolov8, extract_trajectory_detections
from ..configs import CONFIG

logger = logging.getLogger(__name__)

class AverageDisplacementErrorEvaluator(DimensionEvaluator):
    """
    Average Displacement Error (ADE) Evaluator.
    Computes the deviation between the generated subject's trajectory and the GT trajectory.
    """
    def prepare(self):
        model_path = CONFIG.models.yolo
        self.detector = load_yolov8(self.device, model_path=model_path)
        super().prepare()

    def compute(self, **kwargs: Any) -> float:
        video_gen = kwargs.get('tensor_gen')
        video_gt = kwargs.get('tensor_gt')
        video_id = kwargs.get('video_id')
        global_cache = kwargs.get('global_cache')
        
        if video_gen is None or video_gt is None:
            return 1.0 # 无法计算时给最大误差

        H, W = video_gen.shape[2], video_gen.shape[3]

        # 1. 获取生成视频检测结果 (带缓存)
        # 注意：这里缓存 Key 用 _traj_ 区分，因为这是筛选过的单人结果
        cache_key_gen = f"detection_traj_gen_{video_id}"
        if global_cache is not None and cache_key_gen in global_cache:
            gen_results = global_cache[cache_key_gen]
        else:
            gen_results = extract_trajectory_detections(video_gen, self.detector)
            if global_cache is not None: global_cache[cache_key_gen] = gen_results
            
        # 2. 获取 GT 视频检测结果 (带缓存)
        cache_key_gt = f"detection_traj_gt_{video_id}"
        if global_cache is not None and cache_key_gt in global_cache:
            gt_results = global_cache[cache_key_gt]
        else:
            gt_results = extract_trajectory_detections(video_gt, self.detector)
            if global_cache is not None: global_cache[cache_key_gt] = gt_results

        # 3. 计算 ADE
        score = AverageDisplacementError(gen_results, gt_results, H, W)
        return score
        
    def clear(self):
        if hasattr(self, 'detector'): del self.detector
        torch.cuda.empty_cache()
        super().clear()