"""EgoExo Translation Benchmark - Main benchmark engine and evaluation pipeline.

This module provides the core Bench class that orchestrates the evaluation
pipeline for ego-to-exocentric video translation models.
"""

from pathlib import Path
from typing import Union
from tqdm import tqdm
import logging
logger = logging.getLogger()

import torch
from torchvision.transforms.functional import to_pil_image

from .dimension import BenchRouter, DIMENSION_NAMES
from .record.recoder import Recorder
from .dataflow.submission import Submission
from .dataflow.option import Options
from .dataflow.loader import BenchmarkDataLoader
from .configs import CONFIG


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
    
    def __init__(self, device: str, assets_root: Union[str, Path] = 'assets/') -> None:
        """Initialize the benchmark engine.
        
        Args:
            device: Computing device to use ('cuda' or 'cpu')
            assets_root: Path to the assets directory containing test data
        """
        self.device = device
        self.assets_root = Path(assets_root)
        self.router = BenchRouter(device, self.assets_root)
        
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("EgoExoBench")

    def run(self, opt: Options, submission: Submission):
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
        output_dir = Path(opt.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Prepare DataLoader
        try:
            loader_wrapper = BenchmarkDataLoader(opt)
            dataloader = loader_wrapper.dataloader
            meta = loader_wrapper.dataset.metadata
        except Exception as e:
            self.logger.error(f"Failed to create DataLoader: {e}")
            raise e

        # 2. Initialize Recorder
        recorder = Recorder(meta, output_dir)
        
        # 3. Prepare Evaluators
        metrics_list = opt.metrics
        active_evaluators = {}
        
        self.logger.info("Preparing evaluators...")
        for metric_name in metrics_list:
            try:
                evaluator = self.router.get_evaluator(metric_name)
                evaluator.prepare()
                active_evaluators[metric_name] = evaluator
            except Exception as e:
                self.logger.error(f"Failed to initialize evaluator for {metric_name}: {e}")

        # 4. Main Evaluation Loop (One Pass)
        self.logger.info(f"Starting Evaluation on {len(dataloader)} batches...")
        
        # Clear router cache before starting
        self.router.global_cache.clear()

        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move data to device
            ego_videos = batch.get('ego_video')
            gt_videos = batch.get('exo_video')
            ref_images = batch.get('ref_image')
            
            if ego_videos is not None: ego_videos = ego_videos.to(self.device)
            if gt_videos is not None: gt_videos = gt_videos.to(self.device)
            
            ids = batch['video_id']

            for i, vid_id in enumerate(ids):
                # Retrieve Generated Video
                gen_video = submission.get_generated_video(vid_id)
                if gen_video is None:
                    self.logger.warning(f"Missing generation for {vid_id}, skipping.")
                    continue
                
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
                        self.logger.error(f"Error computing {name} for {vid_id}: {e}")

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
        self.logger.info("Finalizing global metrics...")
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
                    self.logger.error(f"Error finalizing {name}: {e}")
            
            # Cleanup evaluator resources
            evaluator.clear()

        # 6. Save Report
        recorder.save_report()
        self.router.global_cache.clear()
        self.logger.info(f"Evaluation complete. Results saved to {output_dir}")