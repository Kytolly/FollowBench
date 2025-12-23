"""Dimension evaluators and routing utilities.

This module exposes the :class:`BenchRouter` which orchestrates per-metric
computations by dynamically loading metric-specific evaluators. It also
contains the base :class:`DimensionEvaluator` class used by concrete
implementations in the `dimension` package.
"""

import importlib
import numpy as np
import gc
from pathlib import Path
from tqdm import tqdm
import logging
logger = logging.getLogger(__name__)

import torch
from torchvision.transforms.functional import to_pil_image
from typing import Any, Dict, Optional, Union, Iterator

from ..utils import gpu
from ..utils.video_kit import extract_i3d_features
from .metric import FrechetVideoDistance
from ..dataflow.set import BenchmarkDataset
from ..dataflow.submission import Submission

DIMENSION_NAMES = [
    'FrechetVideoDistance', 'AestheticQuality', 'ImagingQuality', 
    'TemporalFlickering', 'MotionSmoothness', 'DynamicDegree', 
    'CameraCenteringError', 'AppearanceConsistency', 'ViewpointValidity', 
    'BackgroundSemanticConsistency', 'HumanActionAlignment', 
    'OpticalFlowCorrelation', 'TrajectoryAlignment'
]
DIMENSION_NAMES_IN_SHORT = ['fvd','aq','iq','tf','ms','dd','cce','ac','vv','bsc','haa','ofc','ta']
DIMENSION_MODULE_MAP = dict(zip(DIMENSION_NAMES, DIMENSION_NAMES_IN_SHORT))
SHORT_TO_FULL_MAP = {v: k for k, v in DIMENSION_MODULE_MAP.items()}

class DimensionEvaluator():
    """Base class for all metric evaluators.

    Subclasses should implement :meth:`compute` and may override :meth:`prepare`.
    """
    def __init__(self, device: str) -> None:
        self.device: str = device
        self.model: Optional[Any] = None
        
    def prepare(self) -> None:
        """Prepare internal models or state for evaluation."""
        logger.info(f"{self.__class__.__name__} prepared!")
    
    def compute(self, **kwargs) -> float:
        """Compute the metric for a single case.

        Returns:
            A numeric score for the provided inputs.
        """
        raise NotImplementedError
    
    def clear(self) -> None:
        """Release any heavy resources (models, GPU memory)."""
        if self.model is not None:
            del self.model
            self.model = None
        gpu.clear_gpu_memory()

class BenchRouter():
    """
    路由所有计算逻辑：
    1. DimensionEvaluator 
    2. 管理显存 (加载/卸载模型)
    """
    def __init__(self, device, assets_root):
        self.device = device
        self.assets_root = Path(assets_root)
        self.global_cache = {} # 用于在单次指标计算中共享中间结果 (如检测框)

    def _load_evaluator(self, metric_name):
        """动态加载评估器类"""
        short_name = DIMENSION_MODULE_MAP.get(metric_name, metric_name)
        module_name = f"ego2exo_bench.dimension.{short_name}"
        
        possible_classes = [
            f"{metric_name}Evaluator",
            f"{short_name.upper()}Evaluator", 
            f"{metric_name}" # Fallback
        ]
        
        try:
            module = importlib.import_module(module_name)
            for cls_name in possible_classes:
                if hasattr(module, cls_name):
                    logger.info(f"Loaded {cls_name} from {module_name}")
                    return getattr(module, cls_name)(self.device)
            logger.warning(f"Class not found for {metric_name} in {module_name}")
            return None
        except Exception as e:
            logger.error(f"Failed to load module {module_name}: {e}")
            return None

    def compute_metric_with_loader(self, 
                                   metric_name, 
                                   submission, 
                                   dataloader):
        """
        通用计算入口: 遍历 DataLoader 计算指标
        """
        # 1. FVD 特殊处理 (需累积特征)
        if metric_name in ['FrechetVideoDistance', 'fvd']:
            return self._compute_fvd_with_loader(submission, dataloader)

        # 2. 加载评估器
        full_name = SHORT_TO_FULL_MAP.get(metric_name, metric_name)
        evaluator :DimensionEvaluator= self._load_evaluator(full_name)
        # if not evaluator: return {}

        results = {}
        try:
            evaluator.prepare()
            
            # 3. 遍历 DataLoader
            for batch in tqdm(dataloader, desc=f"Evaluating {full_name}"):
                # 从 Dataflow 获取 GT 和 Ego (已经预处理为 Tensor)
                ids = batch['video_id']
                ego_videos = batch['ego_video'].to(self.device) # [B, T, C, H, W]
                gt_videos = batch['exo_video'].to(self.device)  # [B, T, C, H, W]
                ref_images = batch['ref_image'] # [B, C, H, W] (Normalized)

                for i, vid_id in enumerate(ids):
                    # 获取生成视频
                    gen_video = submission.get_generated_video(vid_id)
                    if gen_video is None:
                        results[vid_id] = 0.0
                        continue
                    
                    # 维度适配: Submission 返回 [T, C, H, W]
                    tensor_gen = gen_video.to(self.device)
                    
                    # Dataset 返回 [T, C, H, W] (如果是从 load_video_to_gpu 加载)
                    # 假设 Dataset 返回 5D [B, T, C, H, W]，取 [i] 变成 [T, C, H, W]
                    tensor_gt = gt_videos[i]
                    tensor_ego = ego_videos[i]
                    
                    # Ref 图片反归一化 (Tensor -> PIL) 用于 AC/BSC 指标
                    # Dataset 通常做了 Normalize((0.5,), (0.5,))
                    pillow_ref = None
                    if ref_images is not None:
                        # 反归一化: val * 0.5 + 0.5
                        ref_tensor = ref_images[i].clone() * 0.5 + 0.5
                        ref_tensor = torch.clamp(ref_tensor, 0, 1)
                        pillow_ref = to_pil_image(ref_tensor)

                    # 构造参数
                    compute_kwargs = {
                        'tensor_gen': tensor_gen,
                        'tensor_gt': tensor_gt,
                        'tensor_ego': tensor_ego,
                        'pillow_ref': pillow_ref,
                        'video_id': vid_id,
                        'global_cache': self.global_cache,
                        'metrics_to_compute': {DIMENSION_MODULE_MAP.get(full_name, full_name)}
                    }
                    
                    score = evaluator.compute(**compute_kwargs)
                    results[vid_id] = score
                    
                    del tensor_gen

        except Exception as e:
            logger.error(f"Error computing {full_name}: {e}")
            import traceback; traceback.print_exc()
        finally:
            evaluator.clear()
            self._clear_cache_for_metric(full_name)
            gc.collect()
            
        return results

    def _compute_fvd_with_loader(self, submission: Submission, dataloader):
        """FVD 计算: 从 Loader 提取 GT 特征，从 Submission 提取 Gen 特征"""
        try:
            from .fvd import FrechetVideoDistanceEvaluator
            evaluator = FrechetVideoDistanceEvaluator(self.device)
            evaluator.prepare()
            
            feats_gen, feats_gt = [], []
            
            # 1. 提取 GT 特征 (遍历 Loader)
            for batch in tqdm(dataloader, desc="FVD: Extracting GT Features"):
                gt_videos = batch['exo_video'].to(self.device) # [B, T, C, H, W]
                for i in range(len(gt_videos)):
                    # extract_i3d_features 需要 [T, C, H, W]
                    feat = extract_i3d_features(gt_videos[i], evaluator.model)
                    feats_gt.append(feat)

            # 2. 提取 Gen 特征 (遍历 Submission)
            # 必须确保 Gen 和 GT 的数量和顺序大体一致 (FVD 是分布距离，不需要严格 pair-wise，但最好样本集一致)
            # 这里我们只计算 Loader 中存在的 ID
            all_ids = []
            for batch in dataloader: all_ids.extend(batch['video_id'])
            
            for vid_id in tqdm(all_ids, desc="FVD: Extracting Gen Features"):
                gen_video = submission.get_generated_video(vid_id)
                if gen_video is not None:
                    gen_video = gen_video.to(self.device)
                    feat = extract_i3d_features(gen_video, evaluator.model)
                    feats_gen.append(feat)
                else:
                    logger.error(f'fail to load {vid_id} mapping generated video as tensor.')

            if not feats_gen or not feats_gt:
                return 0.0

            feats_gen = np.concatenate(feats_gen, axis=0)
            feats_gt = np.concatenate(feats_gt, axis=0)
            return FrechetVideoDistance(feats_gen, feats_gt)

        except Exception as e:
            logger.error(f"FVD Error: {e}")
            return 0.0
        finally:
            if 'evaluator' in locals(): evaluator.clear()

    def _clear_cache_for_metric(self, metric_name: str) -> None:
        """Clear metric-specific keys from the global cache.

        Args:
            metric_name: Full name of the metric just computed.
        """
        if metric_name in ['TemporalFlickering', 'MotionSmoothness', 'OpticalFlowCorrelation']:
            self.global_cache = {k:v for k,v in self.global_cache.items() if 'flow_' not in k}
        elif metric_name in ['CameraCenteringError', 'ViewpointValidity', 'AppearanceConsistency', 'BackgroundSemanticConsistency', 'TrajectoryAlignment']:
            self.global_cache = {k:v for k,v in self.global_cache.items() if 'detection_' not in k}