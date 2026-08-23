"""EgoExo Translation Benchmark - Main benchmark engine and evaluation pipeline.

This module provides the core Bench class that orchestrates the evaluation
pipeline for ego-to-exocentric video translation models.
"""

from pathlib import Path
from typing import Union
from tqdm import tqdm
import logging
DETAILED_FORMAT = (
    '%(asctime)s | '
    '%(levelname)-s | '
    '%(name)s | '
    '%(filename)s:%(lineno)d | '
    '%(funcName)s() | ' 
    # 'PID:%(process)d | TID:%(thread)d | '
    '%(message)s'
)
logging.basicConfig(
    level=logging.INFO,
    format=DETAILED_FORMAT,
    datefmt='%Y-%m-%d %H:%M:%S',
    force=True
)
logger = logging.getLogger()

import torch
from torchvision.transforms.functional import to_pil_image
import torch.nn.functional as F

from .configs import BaseEnvConfig        
from .dataflow.submission import Submission
from .dataflow.option import Options


class Bench:
    """EgoExo Translation Benchmark Engine.
    
    This class orchestrates the complete evaluation pipeline:
    Submission -> DataLoader -> BenchRouter -> Recorder
    
    The benchmark evaluates ego-to-exocentric video translation models across
    multiple dimensions including visual quality, motion consistency, human
    action alignment, and temporal coherence.
    
    Attributes:
        device: Computing device ('cuda' or 'cpu')
        assets_root: Path to benchmark assets and test data
        router: BenchRouter instance for metric computation
    """
    
    def __init__(self, cfg: BaseEnvConfig):
        """Initialize the benchmark engine.
        
        Args:
            device: Computing device to use ('cuda' or 'cpu')
            assets_root: Path to the assets directory containing test data
        """
        self.cfg = cfg
        self.device = cfg.device
        self.assets_root = Path(cfg.assets.path)
        from .dimension import BenchRouter
        self.router = BenchRouter(self.device)
        
        from .dataflow.option import Options
        self.opt = Options(assets=cfg.assets.path, height=cfg.rules.height, width=cfg.rules.width, phase=cfg.meta.split)

    def run(self, cfg: BaseEnvConfig, submission: Submission):
        """Execute the benchmark evaluation pipeline.
        
        Optimized flow:
        1. Initialize all Evaluators.
        2. Iterate through the DataLoader ONCE.
        3. For each batch, compute ALL per-case metrics (sharing detection/flow cache).
        4. After the loop, finalize ALL global metrics (FVD, SCCR).
        
        Args:
            opt: Configuration options for the evaluation
            submission: Submission object containing generated videos
        """
        output_dir = Path(cfg.output.path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Prepare DataLoader
        try:
            from .dataflow.loader import BenchmarkDataLoader
            loader_wrapper = BenchmarkDataLoader(self.opt)
            dataloader = loader_wrapper.dataloader
        except Exception as e:
            logger.error(f"Failed to create DataLoader: {e}")
            raise e

        # 2. Initialize Recorder
        from .record.recoder import Recorder
        recorder = Recorder(cfg.meta, output_dir)
        
        # 3. Prepare Evaluators
        metrics_list = cfg.metrics
        active_evaluators = {}
        
        logger.info("Preparing evaluators...")
        for metric_name in metrics_list:
            try:
                evaluator = self.router.get_evaluator(metric_name)
                evaluator.prepare()
                active_evaluators[metric_name] = evaluator
            except Exception as e:
                logger.error(f"Failed to initialize evaluator for {metric_name}: {e}")

        # 4. Main Evaluation Loop (One Pass)
        logger.info(f"Starting Evaluation on {len(dataloader)} batches...")
        
        # Clear router cache before starting
        self.router.global_cache.clear()
        valid_case_ids = set(submission.mapping.keys())
        logger.info(f"Starting Evaluation. Only {len(valid_case_ids)} cases from submission will be evaluated.")
        for batch in tqdm(dataloader, desc="Evaluating"):
            ids = batch['video_id']
            for i, vid_id in enumerate(ids):
                gen_video = submission.get_generated_video(vid_id)
                if gen_video is None:
                    logger.warning(f"Missing generation for {vid_id}, skipping.")
                    continue
                
                # Move data to device
                ego_videos = batch.get('ego_video')
                gt_videos = batch.get('exo_video')
                ref_images = batch.get('ref_image')
                
                if ego_videos is not None: ego_videos = ego_videos.to(self.device)
                if gt_videos is not None: gt_videos = gt_videos.to(self.device)
                tensor_gen = gen_video.to(self.device)
                tensor_gt = gt_videos[i] if gt_videos is not None else None
                tensor_ego = ego_videos[i] if ego_videos is not None else None
                
                # Handle Reference Image (Normalize -> PIL)
                pillow_ref = None
                if ref_images is not None:
                    # Assuming dataset normalized to [-1, 1], convert to [0, 1] then PIL
                    ref_tensor = ref_images[i].clone() * 0.5 + 0.5
                    ref_tensor = torch.clamp(ref_tensor, 0, 1)
                    pillow_ref = to_pil_image(ref_tensor)

                # Shared Context for this video
                # global_cache is shared across metrics for this video to reuse detections/flows
                tensor_gen, tensor_gt = self._preprocess_video_pair(tensor_gen, tensor_gt)
                compute_kwargs = {
                    'tensor_gen': tensor_gen,
                    'tensor_gt': tensor_gt,
                    'tensor_ego': tensor_ego,
                    'pillow_ref': pillow_ref,
                    'video_id': vid_id,
                    'global_cache': self.router.global_cache
                }

                # Compute ALL metrics for this video
                for name, evaluator in active_evaluators.items():
                    try:
                        score = evaluator.compute(**compute_kwargs)
                        
                        # Only record score if it's a Case-level metric
                        # (Dataset-level metrics like FVD return dummy 0.0 and cache features)
                        if not hasattr(evaluator, 'finalize_metric'):
                            recorder.update(vid_id, {name: score})
                            
                    except Exception as e:
                        logger.error(f"Error computing {name} for {vid_id}: {e}")

                # [Optimization] Immediate Cleanup for Per-Video Cache
                # Clear detection/flow results specific to this video ID to save VRAM.
                # Do NOT clear list-based accumulators (used by FVD/SCCR).
                keys_to_remove = [
                    k for k in self.router.global_cache.keys() 
                    if str(vid_id) in k and 'list' not in k
                ]
                for k in keys_to_remove:
                    del self.router.global_cache[k]
                
                # Release Gen Tensor
                del tensor_gen

        # 5. Finalize Global Metrics (FVD, SCCR)
        logger.info("Finalizing global metrics...")
        for name, evaluator in active_evaluators.items():
            if hasattr(evaluator, 'finalize_metric'):
                try:
                    # Use router helper to calculate final score from accumulated cache
                    global_score = self.router.calculate_global_metric(name, self.router.global_cache)
                    
                    # Support both dict (SCCR) and float (FVD) returns
                    if isinstance(global_score, dict):
                        recorder.update("Dataset_Global", global_score)
                    else:
                        recorder.update("Dataset_Global", {name: global_score})
                        
                except Exception as e:
                    logger.error(f"Error finalizing {name}: {e}")
            
            # Cleanup evaluator resources
            evaluator.clear()

        # 6. Save Report
        recorder.save_report()
        self.router.global_cache.clear()
        logger.info(f"Evaluation complete. Results saved to {output_dir}")
        
    def _align_video_tensor(self, tensor: torch.Tensor, target_frames: int, target_h: int, target_w: int) -> torch.Tensor:
        """
        对单段视频张量进行时空强制对齐。
        假设输入 tensor 形状为: [T, C, H, W]
        """
        T_in, C, H_in, W_in = tensor.shape

        # ==========================================
        # 1. 时间轴对齐: 均匀抽帧 / 插值补帧
        # ==========================================
        if T_in != target_frames:
            # torch.linspace 生成从 0 到 T_in-1 的 target_frames 个均匀点
            # .round().long() 将这些点映射到最近的真实帧索引
            # 效果:
            # - 如果 T_in > target_frames: 均匀跳帧抽样
            # - 如果 T_in < target_frames: 均匀复制某些帧以补齐 (完美防残影)
            indices = torch.linspace(0, T_in - 1, steps=target_frames).round().long()
            tensor = tensor[indices]

        # ==========================================
        # 2. 空间轴对齐: 分辨率缩放
        # ==========================================
        if H_in != target_h or W_in != target_w:
            # 此时 tensor 形状为 [target_frames, C, H_in, W_in]
            # F.interpolate 期望输入 [N, C, H, W]，刚好把 target_frames 作为 N 传入
            tensor = F.interpolate(
                tensor, 
                size=(target_h, target_w), 
                mode='bilinear', 
                align_corners=False
            )

        return tensor
    
    def _preprocess_video_pair(self, tensor_gen: torch.Tensor, tensor_gt: torch.Tensor):
        """
        在吐出 Batch 前，统一处理生成视频和原视频
        """
        target_frames = self.cfg.rules.num_frames 
        target_h = self.cfg.rules.height
        target_w = self.cfg.rules.width 
        tensor_gen = self._align_video_tensor(tensor_gen, target_frames, target_h, target_w)
        tensor_gt = self._align_video_tensor(tensor_gt, target_frames, target_h, target_w)

        return tensor_gen, tensor_gt