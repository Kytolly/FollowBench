from typing import Any, List, Optional, Tuple
import logging
logger = logging.getLogger(__name__)

import torch
from torchvision import transforms
from torchvision.transforms import functional as TF
import torch.nn.functional as F_torch

from ..dimension import DimensionEvaluator
from ..utils.pretrain import (
    load_yolov8,
    load_clip,
    clip_preprocess_tensor,
    get_all_yolo_detections,
    get_crop_embedding
)
from .metric import AppearanceConsistency
from ..configs import CONFIG

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
        model_path = CONFIG.models.yolo
        self.detector = load_yolov8(self.device, model_path=model_path)
        self.clip_model, self.clip_processor = load_clip(self.device)
        super().prepare()
    
    def _extract_reference_embedding(self, ref_tensor: torch.Tensor) -> Optional[torch.Tensor]:
        """从参考图中提取最显著人物的特征"""
        detections = get_all_yolo_detections(ref_tensor.unsqueeze(0), self.detector)
        
        if not detections or not detections[0]:
            return None
            
        # 启发式策略：Reference 中取面积最大的人物
        candidates = detections[0]
        best_candidate = max(candidates, key=lambda x: (x[0][2]-x[0][0]) * (x[0][3]-x[0][1]))
        return get_crop_embedding(ref_tensor, best_candidate[0], self.clip_model, self.clip_processor, self.device)

    def _collect_crops(self, video_gen: torch.Tensor, detections: List[List]) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
        """
        收集所有帧的人物 Crop，并调整到统一尺寸 (224x224)。
        修复：使用 Resize(shortest) + CenterCrop 以保持长宽比，避免变形导致的指标下降。
        """
        crop_list = []
        metadata = []
        H, W = video_gen.shape[2], video_gen.shape[3]
        
        # 使用 torchvision 的 functional API，它支持 Tensor 操作且行为与 CLIPProcessor 一致
        from torchvision.transforms import functional as TF
        from torchvision.transforms import InterpolationMode
        
        for t, candidates in enumerate(detections):
            if not candidates: continue
            
            for idx, (bbox, _) in enumerate(candidates):
                x1, y1, x2, y2 = map(int, bbox.tolist())
                # 边界保护
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(W, x2), min(H, y2)
                
                if x2 <= x1 or y2 <= y1: continue
                
                # 1. Crop on GPU [C, h, w]
                crop = video_gen[t, :, y1:y2, x1:x2] 
                
                # 2. Resize Shortest Edge to 224 (保持长宽比)
                # TF.resize 输入为 Tensor [C, H, W]，若 size 为 int，则缩放短边
                # 使用 BICUBIC 插值以匹配 CLIP 官方预处理
                crop_resized = TF.resize(
                    crop, 
                    size=224, 
                    interpolation=InterpolationMode.BICUBIC, 
                    antialias=True
                )
                
                # 3. Center Crop to 224x224
                # 这一步去除了多余的边缘，保留中心，防止变形
                crop_final = TF.center_crop(crop_resized, output_size=(224, 224))
                
                # 增加 Batch 维度 [1, C, 224, 224]
                crop_list.append(crop_final.unsqueeze(0))
                metadata.append((t, idx))
                
        if not crop_list:
            return None, []
            
        return torch.cat(crop_list, dim=0), metadata

    def _batch_inference(self, batch_crops: torch.Tensor) -> torch.Tensor:
        """[Inference Phase] 批量 CLIP 特征提取"""
        # 1. Normalize (使用 pretrain.py 中的 GPU 函数)
        batch_input = clip_preprocess_tensor(batch_crops)
        
        # 2. Batch Forward
        all_feats = []
        batch_size = 64 # 根据显存调整
        
        with torch.no_grad():
            for i in range(0, len(batch_input), batch_size):
                batch = batch_input[i : i + batch_size]
                feats = self.clip_model.get_image_features(pixel_values=batch)
                feats = F_torch.normalize(feats, p=2, dim=-1)
                all_feats.append(feats)
        
        return torch.cat(all_feats, dim=0)

    def compute(self, **kwargs: Any) -> float:
        """
        Execute calculation.
        """
        video_gen = kwargs.get('tensor_gen')
        video_gt = kwargs.get('tensor_gt')
        pillow_ref = kwargs.get('pillow_ref')
        video_id = kwargs.get('video_id')
        global_cache = kwargs.get('global_cache')

        # 1. Detect (Batch)
        cache_key = f"detection_all_gen_{video_id}"
        if global_cache is not None and cache_key in global_cache:
            detections = global_cache[cache_key]
        else:
            detections = get_all_yolo_detections(video_gen, self.detector)
            if global_cache is not None: global_cache[cache_key] = detections

        # 2. Reference Embedding (With Fallback)
        ref_emb = None
        if pillow_ref is not None:
            try:
                ref_tensor = TF.to_tensor(pillow_ref).to(self.device)
                ref_emb = self._extract_reference_embedding(ref_tensor)
            except Exception as e:
                logger.warning(f"Error with pillow_ref: {e}")
        
        if ref_emb is None and video_gt is not None and len(video_gt) > 0:
            ref_emb = self._extract_reference_embedding(video_gt[0])
            
        if ref_emb is None: return 0.0
        
        # 3. Batch Processing (Speed Optimization)
        # 3.1 Gather Crops
        batch_crops, metadata = self._collect_crops(video_gen, detections)
        if batch_crops is None: return 0.0
        
        # 3.2 Batch Inference
        all_features = self._batch_inference(batch_crops)
        
        # 3.3 Scatter & Score (Best Match Strategy)
        valid_gen_embeddings = []
        
        # 按帧聚合特征，找出每帧的最佳匹配
        frame_feat_map = {} # { t: [feat1, feat2, ...] }
        for idx, (t, _) in enumerate(metadata):
            if t not in frame_feat_map: frame_feat_map[t] = []
            frame_feat_map[t].append(all_features[idx])
            
        for t, feats in frame_feat_map.items():
            # Stack candidates: [M, D]
            feats_tensor = torch.stack(feats)
            # Similarity: [M, D] @ [D, 1] -> [M, 1]
            sims = torch.mm(feats_tensor, ref_emb.t())
            # Winner-Takes-All
            best_idx = torch.argmax(sims)
            valid_gen_embeddings.append(feats_tensor[best_idx].unsqueeze(0))
            
        if not valid_gen_embeddings: return 0.0
        
        # 4. Final Metric
        tensor_gen_embs = torch.cat(valid_gen_embeddings, dim=0)
        return AppearanceConsistency(tensor_gen_embs, ref_emb)
    
    def clear(self):
        """Free VRAM."""
        del self.detector
        del self.clip_model
        del self.clip_processor
        torch.cuda.empty_cache()
        super().clear()
    