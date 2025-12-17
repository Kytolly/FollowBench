import os
import argparse
import torch
from torch.utils.data import DataLoader

from data.benchmark_dataset import Ego2ExoBenchmarkDataset
from data.submission_loader import SubmissionLoader
from utils.recorder import Recorder

# 导入所有 Evaluators
from dimension.cce import CameraCenteringErrorEvaluator
from dimension.vv import ViewpointValidityEvaluator
from dimension.ta import TrajectoryAlignmentEvaluator
from dimension.ac import AppearanceConsistencyEvaluator
from dimension.bsc import BackgroundSemanticConsistencyEvaluator
from dimension.haa import HumanActionAlignmentEvaluator
from dimension.fvd import FrechetVideoDistanceEvaluator
# ... 导入其他 ...

def parse_args():
    parser = argparse.ArgumentParser()
    # Dataset Config
    parser.add_argument('--dataroot', type=str, default='./assets/Ego2ExoDataset')
    parser.add_argument('--hf_repo_id', type=str, default='Kytolly/Ego2ExoFollowShotBenchmark')
    parser.add_argument('--json_path', type=str, default='test/annotation.json')
    parser.add_argument('--phase', type=str, default='test')
    parser.add_argument('--prompt_mode', type=str, default='fullymodal')
    parser.add_argument('--clip_len', type=int, default=16)
    parser.add_argument('--load_size', type=int, default=256)
    
    # Submission Config
    parser.add_argument('--submission_path', type=str, required=True, help="Path to user generated videos (folder or json)")
    
    # Output
    parser.add_argument('--output_dir', type=str, default='./results')
    
    # System
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    return parser.parse_args()

def main():
    opt = parse_args()
    
    # 1. 初始化 Benchmark Dataset (Ground Truth Provider)
    print(">>> Initializing Benchmark Dataset...")
    dataset = Ego2ExoBenchmarkDataset(opt)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    
    # 2. 初始化 Submission Loader (Model Output Provider)
    print(f">>> Loading User Submission from {opt.submission_path}...")
    submission_loader = SubmissionLoader(opt.submission_path, load_size=opt.load_size, clip_len=opt.clip_len)
    
    # 3. 初始化 Recoder
    recorder = Recorder(opt.output_dir)
    
    # 4. 初始化 Evaluators
    # 这里是一个示例，您需要根据实际情况添加所有 Evaluators
    evaluators = {
        'CCE': CameraCenteringErrorEvaluator(opt.device),
        'HAA': HumanActionAlignmentEvaluator(opt.device),
        'FVD': FrechetVideoDistanceEvaluator(opt.device),
        # 'AC': AppearanceConsistencyEvaluator(opt.device),
        # ... Add others
    }
    
    # 预热/准备 Evaluators
    for name, evaluator in evaluators.items():
        print(f"Preparing {name}...")
        evaluator.prepare()
        
    global_cache = {} # 用于跨 Evaluator 共享检测结果
    
    # 5. 评测循环
    print(">>> Starting Evaluation Loop...")
    for i, data in enumerate(dataloader):
        video_id = data['video_id'][0] # Batch size=1
        print(f"Processing [{i+1}/{len(dataset)}] ID: {video_id}")
        
        # 获取 Benchmark 数据
        ego_video = data['ego_video'].to(opt.device) # [1, T, C, H, W] -> need [T, C, H, W] ? Evaluators usually expect Batch or T first
        exo_gt = data['exo_video'].to(opt.device)
        ref_img_path = data.get('ref_image_path') # 如果 Evaluator 需要路径
        
        # 获取用户生成视频
        # 注意: submission_loader 返回的是 [T, C, H, W]
        gen_video = submission_loader.get_generated_video(video_id)
        
        if gen_video is None:
            print(f"  [Warning] Submission missing for ID {video_id}. Skipping.")
            continue
            
        gen_video = gen_video.to(opt.device).unsqueeze(0) # [1, T, C, H, W] used by most evaluators? 
        # Check dimensions: Evaluators defined in previous steps mostly took [T, C, H, W] or [B, T, C, H, W]
        # 假设 evaluator.compute 接受带 batch 维度的输入
        
        # 运行评估
        metrics = {}
        for name, evaluator in evaluators.items():
            try:
                # 传递所有可能需要的参数
                score = evaluator.compute(
                    tensor_gen=gen_video[0], # 假设 Evaluator 内部处理单视频 [T, C, H, W]
                    tensor_gt=exo_gt[0],
                    tensor_ego=ego_video[0],
                    video_id=video_id,
                    global_cache=global_cache
                )
                metrics[name] = score
            except Exception as e:
                print(f"  [Error] {name} failed: {e}")
                metrics[name] = float('nan')
        
        print(f"  Scores: {metrics}")
        recorder.update(video_id, metrics)
        
    # 6. 保存报告
    recorder.save_report()
    
    # 7. 清理
    for evaluator in evaluators.values():
        evaluator.clear()

if __name__ == "__main__":
    main()