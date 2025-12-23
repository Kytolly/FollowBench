"""EgoExo Translation Benchmark - Main benchmark engine and evaluation pipeline.

This module provides the core Bench class that orchestrates the evaluation
pipeline for ego-to-exocentric video translation models.
"""

import logging
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
import torch

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
        self.router = BenchRouter(device, assets_root)

    def evaluate(self, 
                 submission: Submission, 
                 output_dir: Union[str, Path] = 'output/',
                 metrics_list: Optional[List[str]] = None,
                 batch_size: int = 1,
                 num_workers: int = 4,
                 *args: Any,
                 **kwargs: Any) -> None:
        """Run complete evaluation pipeline on a submission.
        
        This method processes the submission through all evaluation metrics,
        computes scores, and generates a comprehensive evaluation report.
        
        Args:
            submission: Submission object containing model results
            output_dir: Directory to save evaluation results
            metrics_list: List of metric names to compute. If None, computes all metrics
            batch_size: Batch size for data loading
            num_workers: Number of worker processes for data loading
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments including:
                - annotation_path: Custom path to annotation file
                
        Raises:
            Exception: If DataLoader creation fails or evaluation encounters errors
        """
        if metrics_list is None:
            metrics_list = DIMENSION_NAMES
        
        # Prepare dataflow options from submission metadata
        meta = submission.meta_info
        anno_path = kwargs.get('annotation_path', self.assets_root / 'test/annotation.json')
        
        opt = Options(
            assets=str(self.assets_root),
            phase='test',  # Force test mode for evaluation
            modal=meta.get('modal', 'vace_instruct'),
            mode=meta.get('mode', 'easy'),
            batch_size=batch_size,
            num_workers=num_workers,
            height=CONFIG['rules']['resolution_height'],
            width=CONFIG['rules']['resolution_width'],
            clip_len=300   # Default frame count
        )
        
        # Initialize DataLoader
        # This automatically loads GT and Ego videos without manual path passing
        try:
            loader_wrapper = BenchmarkDataLoader(opt)
            dataloader = loader_wrapper.dataloader
        except Exception as e:
            logging.error(f"Failed to create DataLoader: {e}")
            raise e

        # Initialize result recorder
        recorder = Recorder(meta, output_dir)
        
        # Execute evaluation loop
        logging.info("Starting Evaluation Pipeline...")
        for metric in metrics_list:
            logging.info(f"--- Computing {metric} ---")
            
            # Compute metric using router
            scores = self.router.compute_metric_with_loader(
                metric_name=metric, 
                submission=submission, 
                dataloader=dataloader
            )
            
            # Record results (BenchRouter returns {vid: score} or float)
            # Recorder.update expects (vid, dict), so we need to adapt
            if isinstance(scores, dict):
                # Case-level metrics
                for vid, score in scores.items():
                    recorder.update(vid, {metric: score})
            else:
                # Dataset-level metrics (e.g., FVD)
                # Record under a virtual ID for dataset-level metrics
                # TODO: Consider adding add_global_metric interface to Recorder
                recorder.update("Dataset_Global", {metric: scores})

        # Save evaluation report
        recorder.save_report()
        logging.info(f"Evaluation complete. Results saved to {output_dir}")
