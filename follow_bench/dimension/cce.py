"""Camera-centering evaluator utilities."""

from typing import Any
from torch import Tensor
import logging
logger = logging.getLogger(__name__)

import torch

from ..dimension import DimensionEvaluator
from .metric import CameraCenteringError
from ..utils.pretrain import (
    get_yolo_detection_results, 
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
        self.model = load_yolov8(self.device, model_path=model_path)
        super().prepare()

    def compute(self, **kwargs: Any):  # noqa: ANN201, ANN101
        """Compute CCE for generated video.

        Expected kwargs: 'tensor_gen', 'video_id', 'global_cache'. Returns a float
        score in [0.0, 1.0], where higher indicates worse centering.
        """
        video_gen: Tensor = kwargs.get('tensor_gen')
        video_id = kwargs.get('video_id')
        global_cache = kwargs.get('global_cache')

        # === 核心：检测结果缓存逻辑 ===
        cache_key = f"detection_gen_{video_id}"
        detections = None

        # 1. 尝试从全局缓存读取 (可能由 SDR 指标先生成)
        if global_cache is not None and cache_key in global_cache:
            detections = global_cache[cache_key]
            logger.info(f"CCE: Detection cache hit for {video_id}")
        
        # 2. 若未命中，则运行 YOLOv8 批量检测
        if detections is None:
            # 内部调用 Ultralytics YOLO 的 Tensor 推理接口
            detections = get_yolo_detection_results(video_gen, self.model)
            if global_cache is not None:
                global_cache[cache_key] = detections
                logger.info(f"CCE: Detection cache updated for {video_id}")

        # 3. 调用纯 Tensor Metric 进行计算
        H, W = video_gen.shape[2], video_gen.shape[3]
        score = CameraCenteringError(detections, H, W)
        
        return float(score)
    
    def clear(self):
        """释放局部模型引用，但不清除 global_cache"""
        self.model = None
        torch.cuda.empty_cache()