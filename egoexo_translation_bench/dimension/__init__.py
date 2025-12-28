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
    'FrechetVideoDistance', 'SourceControlConditionRecall', # [新增] SCCR
    'AestheticQuality', 'ImagingQuality', 
    'TemporalFlickering', 'MotionSmoothness', 'DynamicDegree', 
    'CameraCenteringError', 'AppearanceConsistency', 'ViewpointValidity', 
    'BackgroundSemanticConsistency', 'HumanActionAlignment', 
    'OpticalFlowCorrelation', 'TrajectoryAlignment'
]

DIMENSION_NAMES_IN_SHORT = [
    'fvd', 'sccr', # [新增] sccr
    'aq', 'iq',
    'tf', 'ms', 'dd',
    'cce', 'ac', 'vv',
    'bsc', 'haa',
    'ofc', 'ta'
]
DIMENSION_MODULE_MAP = dict(zip(DIMENSION_NAMES, DIMENSION_NAMES_IN_SHORT))
SHORT_TO_FULL_MAP = {v: k for k, v in DIMENSION_MODULE_MAP.items()}

class DimensionEvaluator():
    """Base class for all metric evaluators.

    Subclasses should implement :meth:`compute` and may override :meth:`prepare`.
    """
    def __init__(self, device: str):
        self.device: str = device
        self.model: Optional[Any] = None
        
    def prepare(self):
        """Prepare internal models or state for evaluation."""
        logger.info(f"{self.__class__.__name__} prepared!")
    
    def compute(self, **kwargs):
        """Compute the metric for a single case.

        Returns:
            A numeric score for the provided inputs.
        """
        raise NotImplementedError
    
    def clear(self):
        """Release any heavy resources (models, GPU memory)."""
        if self.model is not None:
            del self.model
            self.model = None
        gpu.clear_gpu_memory()
        logger.info(f"{self.__class__.__name__} VRAM cleared!")

class BenchRouter:
    """Router to manage evaluators and compute metrics."""

    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.evaluators = {}
        self.global_cache = {}

    def get_evaluator(self, metric_name: str) -> DimensionEvaluator:
        """Lazy load and return the evaluator instance."""
        if metric_name not in DIMENSION_NAMES:
            raise ValueError(f"Unknown metric: {metric_name}")
        
        if metric_name not in self.evaluators:
            module_name = DIMENSION_MODULE_MAP[metric_name]
            module = importlib.import_module(f".{module_name}", package=__package__)
            class_name = f"{metric_name}Evaluator"
            evaluator_cls = getattr(module, class_name)
            self.evaluators[metric_name] = evaluator_cls(device=self.device)
            
        return self.evaluators[metric_name]
    
    def compute_metric_with_loader(
        self, 
        metric_name: str, 
        submission: Submission, 
        dataloader: Any
    ) -> Union[Dict[str, float], float]:
        """
        通用计算入口：遍历 DataLoader 计算指标。
        统一处理 Case-level (如 IQ, TF) 和 Dataset-level (如 FVD, SCCR) 指标。
        """
        full_name = SHORT_TO_FULL_MAP.get(metric_name, metric_name)
        evaluator = self.get_evaluator(full_name)
        
        results = {}
        
        try:
            evaluator.prepare()
            
            # 1. 主循环：遍历数据集 (Feature Extraction / Single Case Compute)
            for batch in tqdm(dataloader, desc=f"Evaluating {full_name}"):
                ids = batch['video_id']
                
                # 预处理数据移至 GPU
                ego_videos = batch.get('ego_video')
                gt_videos = batch.get('exo_video')
                ref_images = batch.get('ref_image')

                if ego_videos is not None: ego_videos = ego_videos.to(self.device)
                if gt_videos is not None: gt_videos = gt_videos.to(self.device)

                for i, vid_id in enumerate(ids):
                    # 获取生成视频
                    gen_video = submission.get_generated_video(vid_id)
                    if gen_video is None:
                        # 对于生成失败的样本，Case-level 记 0，Global-level 忽略
                        if not hasattr(evaluator, 'finalize_metric'):
                            results[vid_id] = 0.0
                        continue
                    
                    tensor_gen = gen_video.to(self.device)
                    tensor_gt = gt_videos[i] if gt_videos is not None else None
                    tensor_ego = ego_videos[i] if ego_videos is not None else None
                    
                    # Ref 图片处理 (用于 AC/BSC)
                    pillow_ref = None
                    if ref_images is not None:
                        # 反归一化 [-1, 1] -> [0, 1] -> PIL
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
                        'global_cache': self.global_cache, # 关键：传入缓存
                        'metrics_to_compute': {DIMENSION_MODULE_MAP.get(full_name, full_name)}
                    }
                    
                    # 执行计算或缓存特征
                    score = evaluator.compute(**compute_kwargs)
                    
                    # 仅记录 Case-level 的分数
                    if not hasattr(evaluator, 'finalize_metric'):
                        results[vid_id] = score
                    
                    # 及时释放显存
                    del tensor_gen

            # 2. 结算阶段：处理 Global Metrics (FVD, SCCR)
            if hasattr(evaluator, 'finalize_metric'):
                return self.calculate_global_metric(full_name, self.global_cache)
            
            return results

        except Exception as e:
            logger.error(f"Error computing {full_name}: {e}")
            import traceback; traceback.print_exc()
            return 0.0 if hasattr(evaluator, 'finalize_metric') else {}
        finally:
            evaluator.clear()
            self._clear_cache_for_metric(full_name)
            gc.collect()
            
    def calculate_global_metric(self, metric_name: str, global_cache: dict):
        """
        通用全局指标计算入口。
        不再需要传入 dataset 和 submission 重新遍历，
        而是直接利用主循环中填充好的 global_cache 进行结算。
        """
        evaluator = self.get_evaluator(metric_name)
        
        # 检查是否支持 finalize_metric 接口
        if hasattr(evaluator, 'finalize_metric'):
            logger.info(f"Finalizing global metric: {metric_name}...")
            return evaluator.finalize_metric(global_cache)
        else:
            logger.warning(f"{metric_name} does not support global finalization.")
            return 0.0
        
    def _clear_cache_for_metric(self, metric_name: str):
        """Clear metric-specific keys from the global cache.

        Args:
            metric_name: Full name of the metric just computed.
        """
        if metric_name in ['TemporalFlickering', 'MotionSmoothness', 'OpticalFlowCorrelation']:
            self.global_cache = {k:v for k,v in self.global_cache.items() if 'flow_' not in k}
        elif metric_name in ['CameraCenteringError', 'ViewpointValidity', 'AppearanceConsistency', 'StructuralFidelity', 'BackgroundSemanticConsistency']:
            # 清理检测结果缓存
            self.global_cache = {k:v for k,v in self.global_cache.items() if 'detection_' not in k}
        elif metric_name in ['FrechetVideoDistance']:
            self.global_cache = {k:v for k,v in self.global_cache.items() if 'i3d_feat_' not in k}
        # [新增] SCCR 缓存清理 (如果使用了 global_cache 模式)
        elif metric_name in ['SourceControlConditionRecall']:
            self.global_cache = {k:v for k,v in self.global_cache.items() if 'videomae_feat_' not in k}
            
        gc.collect()
        torch.cuda.empty_cache()