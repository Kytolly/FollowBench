from typing import Any
import logging
logger = logging.getLogger(__name__)

import torch
from torchvision.transforms import functional as F

from ..dimension import DimensionEvaluator
from ..utils.pretrain import (
    load_yolov8,
    load_clip,
    get_yolo_detection_results
)

class AppearanceConsistencyEvaluator(DimensionEvaluator):
    """
    Appearance Consistency / Fidelity Evaluator (AC_CLIP).
    
    Metric:
        Measures the preservation of subject identity/appearance over time.
        1. Detect subject (Person) using YOLOv8.
        2. Crop subject from the frame.
        3. Extract features using CLIP Image Encoder.
        4. Compute Cosine Similarity w.r.t Reference Image (GT Frame 0).
    """
    def prepare(self):
        """Load YOLO detector and CLIP model."""
        self.detector = load_yolov8(self.device)
        self.clip_model, self.clip_processor = load_clip(self.device)
        super().prepare()
        
    def _get_crop_embedding(self, frame_tensor, bbox):
        """执行 Tensor 裁剪并调用 Processor 预处理"""
        # Tensor 切片操作 (x1, y1, x2, y2)
        x1, y1, x2, y2 = map(int, bbox.tolist())
        crop = frame_tensor[:, y1:y2, x1:x2]
        inputs = self.clip_processor(images=F.to_pil_image(crop.cpu()), return_tensors="pt").to(self.device)
        with torch.no_grad():
            return self.clip_model.get_image_features(**inputs)
        
    def compute(self, **kwargs: Any):
        """
        Execute calculation.
        Expected kwargs:
            tensor_gen: [T, C, H, W] Generated Video
            tensor_gt:  [T, C, H, W] GT Video (Used for Reference Frame)
        """
        video_gen = kwargs.get('tensor_gen') # [T, 3, H, W]
        video_gt = kwargs.get('tensor_gt')
        video_id = kwargs.get('video_id')
        global_cache = kwargs.get('global_cache')

        # 1. 获取检测结果（优先从 global_cache 读取 CCE 等指标存入的 YOLO 结果）
        cache_key = f"detection_gen_{video_id}"
        if global_cache is not None and cache_key in global_cache:
            detections_gen = global_cache[cache_key]
        else:
            detections_gen = get_yolo_detection_results(video_gen, self.detector)
            if global_cache is not None: global_cache[cache_key] = detections_gen

        # 2. 提取参考特征 (Reference Embedding)
        # 使用 GT 第一帧作为 Appearance 的基准
        ref_frame = video_gt[0:1] 
        ref_det = get_yolo_detection_results(ref_frame, self.detector)
        if not ref_det or ref_det[0] is None: return 0.0
        
        ref_emb = self._get_crop_embedding(video_gt[0], ref_det[0][0])
        if ref_emb is None: return 0.0
        
    def clear(self):
        """Free VRAM."""
        del self.detector
        del self.clip_model
        del self.clip_processor
        torch.cuda.empty_cache()
        super().clear()
    