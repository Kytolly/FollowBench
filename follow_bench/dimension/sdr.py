"""Viewpoint- and detection-based evaluator utilities."""

from typing import Any
import logging
logger = logging.getLogger(__name__)

import torch

from ..utils.pretrain import (
    get_yolo_detection_results, 
    load_yolov8
)
from .metric import SubjectDetectionRate
from ..dimension import DimensionEvaluator

class SubjectDetectionRateEvaluator(DimensionEvaluator):
    """Evaluator for viewpoint validity measuring how often a person is detected."""

    def prepare(self):
        """Load the detector and call base prepare."""
        self.model = load_yolov8(self.device)
        super().prepare()

    def compute(self, **kwargs: Any):
        """Compute viewpoint validity over sampled frames.

        Expected kwargs: 'tensor_gen', 'video_id', 'global_cache'. The evaluator
        samples frames (every 5th frame) to speed up detection and then computes
        the fraction of frames with person detections via `ViewpointValidity`.

        Returns a float in [0.0, 1.0].
        """
        # parse kwargs
        video_gen = kwargs.get('tensor_gen') # [T, C, H, W]
        video_id = kwargs.get('video_id')
        global_cache = kwargs.get('global_cache')

        # === 核心：检测结果缓存逻辑 ===
        # 统一使用与 CCE, AC 相同的 Cache Key
        cache_key = f"detection_gen_{video_id}"
        detections = None

        # 1. 尝试从全局缓存读取
        if global_cache is not None and cache_key in global_cache:
            detections = global_cache[cache_key]
            logger.info(f"SDR: Using cached detection results for {video_id}")
        
        # 2. 若未命中（例如 SDR 是第一个运行的检测类指标），则运行 YOLOv8 检测
        if detections is None:
            # 确保输入在正确的 device 上
            video_gen = video_gen.to(self.device)
            detections = get_yolo_detection_results(video_gen, self.model)
            
            # 将结果写入缓存供后续指标（如 CCE, AC）使用
            if global_cache is not None:
                global_cache[cache_key] = detections
                logger.info(f"SDR: Detection results cached for {video_id}")

        # 3. 计算指标
        score = SubjectDetectionRate(detections)
        
        return float(score)
    
    def clear(self):
        """释放局部模型引用，但不清除 global_cache"""
        self.model = None
        torch.cuda.empty_cache()