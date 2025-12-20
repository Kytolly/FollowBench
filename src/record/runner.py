import os
import torch
import logging
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader

from src.dataflow.loader import BenchmarkDataLoader
from src.dataflow.submission import Submission
from src.record.recoder import Ego2ExoRecorder
from src import DIMENSION_MODULE_MAP
import importlib

class BenchmarkRunner:
    def __init__(self, opt, device='cuda'):
        self.opt = opt
        self.device = device
        self.logger = logging.getLogger("BenchmarkRunner")
        
        # 1. 初始化数据加载器 (Ego & GT)
        self.dataloader = BenchmarkDataLoader(opt).dataloader
        
        # 2. 初始化评估器工厂
        self.evaluators = {}
        self.global_cache = {} # 用于共享光流、检测等中间结果
        
    def _load_evaluators(self, metrics_list):
        """按需加载指标评估器"""
        for metric_name in metrics_list:
            # 映射 FVD -> fvd
            short_name = DIMENSION_MODULE_MAP.get(metric_name, metric_name).lower()
            module_name = f"src.dimension.{short_name}"
            class_name = f"{metric_name}Evaluator" if metric_name in DIMENSION_MODULE_MAP else f"{metric_name.upper()}Evaluator"
            
            try:
                module = importlib.import_module(module_name)
                # 尝试不同的命名规范匹配
                if hasattr(module, class_name):
                    cls = getattr(module, class_name)
                elif hasattr(module, f"{short_name.upper()}Evaluator"):
                    cls = getattr(module, f"{short_name.upper()}Evaluator")
                else:
                    # Fallback mapping
                    map_fix = {
                        'fvd': 'FrechetVideoDistanceEvaluator',
                        'cce': 'CameraCenteringErrorEvaluator',
                        'aq': 'AestheticQualityEvaluator',
                        # ... 其他映射
                    }
                    cls = getattr(module, map_fix.get(short_name, f"{short_name.upper()}Evaluator"))
                    
                self.evaluators[metric_name] = cls(self.device)
                self.evaluators[metric_name].prepare()
                self.logger.info(f"Loaded evaluator: {metric_name}")
            except Exception as e:
                self.logger.error(f"Failed to load evaluator {metric_name}: {e}")

    def evaluate(self, submission_path, model_name, metrics_to_run=None):
        """
        核心评估循环
        Args:
            submission_path: 提交的文件夹路径或 mapping json
            model_name: 模型名称 (用于 meta 记录)
        """
        if metrics_to_run is None:
            metrics_to_run = list(DIMENSION_MODULE_MAP.keys())

        # 加载 Submission
        # 如果传入的是文件夹，假设它是 source_path，submission_path 设为 None (需要 Submission 类支持自动扫描或外部构建)
        # 这里假设传入的是标准的 json 路径
        if submission_path.endswith('.json'):
             sub = Submission(source_path=os.path.dirname(submission_path), submission_path=submission_path)
        else:
             # 简易模式：直接扫文件夹，假设文件名就是 ID
             # 需根据实际 Submission 类逻辑调整
             pass

        # 初始化记录器
        recorder = Ego2ExoRecorder(
            team_name="AutoEval", 
            model_name=model_name, 
            output_dir=self.opt.output_dir
        )
        
        # 加载模型
        self._load_evaluators(metrics_to_run)
        
        self.logger.info(f"Starting evaluation for {model_name}...")
        
        # FVD 特殊处理：通常需要计算整个数据集的分布
        if 'FrechetVideoDistance' in metrics_to_run or 'FVD' in metrics_to_run:
            # 这里的 FVD 计算逻辑需要适配你的 FVD Evaluator 接口
            # 假设 FVD Evaluator 支持传入 tensor 列表或者 路径
            pass 

        for batch in tqdm(self.dataloader, desc="Evaluating Cases"):
            video_ids = batch['video_id']
            # 注意: DataLoader batch_size=1 时 video_ids 是 tuple ('case-1',)
            
            # 数据移动到 GPU
            ego_video = batch['ego_video'].to(self.device) # [B, T, C, H, W]
            exo_gt = batch['exo_video'].to(self.device)
            ref_image = batch['ref_image'] # PIL list or Tensor
            
            # 处理 batch (目前默认 B=1)
            for i, vid_id in enumerate(video_ids):
                # 获取生成视频
                gen_video = sub.get_generated_video(vid_id)
                if gen_video is None:
                    self.logger.warning(f"Missing generation for {vid_id}")
                    continue
                gen_video = gen_video.to(self.device).unsqueeze(0) # [1, T, C, H, W] if loader returns TCHW
                
                # 转换 Ref Image (根据 Evaluator 需求，有的要 PIL 有的要 Tensor)
                # 这里假设 dataloader 的 ref_image 已经是 Tensor [B, C, H, W]
                # 如果 Evaluator 需要 PIL，需在此转换
                curr_ref = ref_image[i] 
                # Tips: 如果 Evaluator 需要 PIL，可以用 transforms.ToPILImage()(curr_ref)

                case_scores = {}
                
                for metric_name, evaluator in self.evaluators.items():
                    # 跳过 FVD (Dataset level metric)
                    if metric_name in ['FrechetVideoDistance', 'FVD']: continue
                    
                    try:
                        # 构造计算参数
                        compute_kwargs = {
                            'tensor_gen': gen_video[0], # [T, C, H, W]
                            'tensor_gt': exo_gt[i],
                            'tensor_ego': ego_video[i],
                            'pillow_ref': None, # 需根据实际情况传入 PIL
                            'video_id': vid_id,
                            'global_cache': self.global_cache
                        }
                        
                        score = evaluator.compute(**compute_kwargs)
                        case_scores[metric_name] = float(score)
                    except Exception as e:
                        self.logger.error(f"Error computing {metric_name} for {vid_id}: {e}")
                        case_scores[metric_name] = 0.0
                
                # 记录该 Case 的所有分数
                recorder.add_case_score(vid_id, case_scores)
                
                # 清理显存缓存 (针对当前 Video ID)
                self._clear_cache(vid_id)

        # 保存报告
        report_path = recorder.save_report()
        return report_path

    def _clear_cache(self, video_id):
        keys = list(self.global_cache.keys())
        for k in keys:
            if str(video_id) in k:
                del self.global_cache[k]