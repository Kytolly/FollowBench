import logging
import os
from pathlib import Path
import torch

from .dimension import BenchRouter, DIMENSION_NAMES
from .record.recoder import Recorder
from .dataflow.submission import Submission
from .dataflow.option import Options
from .dataflow.loader import BenchmarkDataLoader
from .configs import CONFIG

class Bench():
    """
    Ego2Exo Benchmark Engine.
    Pipeline: Submission -> DataLoader -> BenchRouter -> Recorder
    """
    def __init__(self, device, assets_root='assets/'):
        self.device = device
        self.assets_root = Path(assets_root)
        self.router = BenchRouter(device, assets_root)

    def evaluate(self, 
                 submission: Submission, 
                 output_dir: str = 'output/',
                 metrics_list: list = None,
                 batch_size: int = 1,
                 num_workers: int = 4,
                 *args,
                 **kwargs):
        """
        Args:
            output_dir: 结果输出文件夹
            metrics_list: 需要计算的指标列表
        """
        if metrics_list is None:
            metrics_list = DIMENSION_NAMES
        
        # 准备 Dataflow Options
        meta = submission.meta_info
        anno_path = kwargs.get('annotation_path', self.assets_root / 'test/annotation.json')
        # caption_path = kwargs.get('caption_path', self.assets_root / 'train/annotation.json')
        opt = Options(
            assets=str(self.assets_root),
            # annotation=str(anno_path),
            phase='test', # 强制为 test 模式
            modal=meta.get('modal', 'vace_instruct'),
            mode=meta.get('mode', 'easy'),
            batch_size=batch_size,
            num_workers=num_workers,
            height=CONFIG['rules']['resolution_height'],
            width=CONFIG['rules']['resolution_width'],
            clip_len=300   # 默认帧数
        )
        
        # 初始化 DataLoader
        # 这将自动加载 GT 和 Ego 视频，无需手动传路径
        # logging.info(f"Initializing DataLoader with annotation: {opt.annotation}")
        try:
            loader_wrapper = BenchmarkDataLoader(opt)
            dataloader = loader_wrapper.dataloader
        except Exception as e:
            logging.error(f"Failed to create DataLoader: {e}")
            raise e

        # 4. 初始化 Recorder
        recorder = Recorder(meta, output_dir)
        
        # 5. 执行计算循环
        logging.info("Starting Evaluation Pipeline...")
        for metric in metrics_list:
            logging.info(f"--- Computing {metric} ---")
            
            # 将 loader 传给 router
            scores = self.router.compute_metric_with_loader(
                metric_name=metric, 
                submission=submission, 
                dataloader=dataloader
            )
            
            # 记录结果 (Dimensionrouter 返回 {vid: score} 或 float)
            # Recorder.update 需要 (vid, dict)，我们需要适配一下
            if isinstance(scores, dict):
                # Case-level metrics
                for vid, score in scores.items():
                    recorder.update(vid, {metric: score})
            else:
                # Dataset-level metrics (e.g., FVD)
                # Recorder 目前设计为 update(vid, metrics)，dataset level 可能需要特殊处理
                # 这里简单将其记录在一个虚拟 ID 下，或者 Recorder 需要增加 add_global_metric 接口
                # 暂时记录为 "Dataset_Global"
                recorder.update("Dataset_Global", {metric: scores})

        # 6. 保存报告
        recorder.save_report()
        logging.info(f"Evaluation complete. Results saved to {output_dir}")