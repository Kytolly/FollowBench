"""Camera-centering evaluator utilities."""

from typing import Any
from torch import Tensor
import logging
logger = logging.getLogger(__name__)

import torch

from ..dimension import DimensionEvaluator
from .metric import CameraCenteringError
from ..utils.pretrain import (
    get_all_yolo_detections, 
    load_yolov8
)
from ..configs import CONFIG

class CameraCenteringErrorEvaluator(DimensionEvaluator):
    """Evaluator for Camera Centering Error (CCE).

    Uses a person detector to compute per-frame detections and measures how
    centrally the detected person is positioned in each frame.
    """

    def prepare(self):  # noqa: ANN201, ANN101
        """Load the detector and call base prepare."""
        model_path = CONFIG.models.yolo
        self.detector = load_yolov8(self.device, model_path=model_path)
        super().prepare()

    def compute(self, **kwargs: Any) -> float:
        """
        Execute calculation.
        """
        video_gen = kwargs.get('tensor_gen') # [T, 3, H, W]
        video_id = kwargs.get('video_id')
        global_cache = kwargs.get('global_cache')
        
        if video_gen is None: return 1.0

        H, W = video_gen.shape[2], video_gen.shape[3]

        # 1. 获取每一帧的所有检测结果
        cache_key = f"detection_all_gen_{video_id}"
        if global_cache is not None and cache_key in global_cache:
            all_detections = global_cache[cache_key]
        else:
            all_detections = get_all_yolo_detections(video_gen, self.detector)
            if global_cache is not None: global_cache[cache_key] = all_detections

        # 2. 筛选出每帧的主角 (Largest BBox)
        # metric.py 中的 CameraCenteringError 期望输入是 List[(bbox, conf) or None]
        final_detections = []
        
        for candidates in all_detections:
            if not candidates:
                final_detections.append(None)
                continue
            
            # 策略：选择面积最大的框作为主角
            # bbox: [x1, y1, x2, y2]
            best_candidate = max(candidates, key=lambda x: (x[0][2]-x[0][0]) * (x[0][3]-x[0][1]))
            
            # metric.py 需要的格式通常是一个列表或元组，包含 bbox
            # 这里我们传入 [(bbox, conf)] 这种 list 形式，适配 metric.py 的逻辑
            final_detections.append([best_candidate[0]]) 

        # 3. 调用 metric.py 计算
        # 注意: metric.py 的 CameraCenteringError 函数签名是 (detection_results, H, W)
        # 它内部循环 for res in detection_results: box = res[0]
        # 所以我们上面 append([best_candidate[0]]) 是正确的
        score = CameraCenteringError(final_detections, H, W)
        
        return score
    
    def clear(self):
        """释放局部模型引用，但不清除 global_cache"""
        self.model = None
        torch.cuda.empty_cache()