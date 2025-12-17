import os
import argparse
import torch
from torch.utils.data import DataLoader

from src.dataflow.loader import Ego2ExoBenchmarkDataset
from src.dataflow.submission import SubmissionLoader
from src.record.recoder import Recorder

# 导入所有 Evaluators
from src.dimension.cce import CameraCenteringErrorEvaluator
from src.dimension.vv import ViewpointValidityEvaluator
from src.dimension.ta import TrajectoryAlignmentEvaluator
from src.dimension.ac import AppearanceConsistencyEvaluator
from src.dimension.bsc import BackgroundSemanticConsistencyEvaluator
from src.dimension.tf import TemporalFlickeringEvaluator
from src.dimension.ms import MotionSmoothnessEvaluator
from src.dimension.dd import DynamicDegreeEvaluator
from src.dimension.haa import HumanActionAlignmentEvaluator
from src.dimension.aq import AestheticQualityEvaluator
from src.dimension.iq import ImagingQualityEvaluator
from src.dimension.fvd import FrechetVideoDistanceEvaluator
# 如有 OFC 也请导入

def parse_args():
    parser = argparse.ArgumentParser()
    # ... (保持原有参数定义) ...
    parser.add_argument('--dataroot', type=str, default='./assets/Ego2ExoDataset')
    parser.add_argument('--submission_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./results')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()

def main():
    opt = parse_args()
    
    print(">>> Initializing...")
    dataset = Ego2ExoBenchmarkDataset(opt) # 需自行适配 Dataset 参数
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    submission_loader = SubmissionLoader(opt.submission_path)
    recorder = Recorder(opt.output_dir)
    
    # 初始化所有 Evaluator
    evaluators = {
        'CCE': CameraCenteringErrorEvaluator(opt.device),
        'VV': ViewpointValidityEvaluator(opt.device),
        'AC': AppearanceConsistencyEvaluator(opt.device),
        'BSC': BackgroundSemanticConsistencyEvaluator(opt.device),
        'TF': TemporalFlickeringEvaluator(opt.device),
        'MS': MotionSmoothnessEvaluator(opt.device),
        'DD': DynamicDegreeEvaluator(opt.device),
        'HAA': HumanActionAlignmentEvaluator(opt.device),
        'AQ': AestheticQualityEvaluator(opt.device),
        'IQ': ImagingQualityEvaluator(opt.device),
        'FVD': FrechetVideoDistanceEvaluator(opt.device),
    }
    
    for name, e in evaluators.items():
        print(f"Preparing {name}...")
        e.prepare()
        
    global_cache = {} 
    
    print(">>> Starting Evaluation Loop...")
    for i, data in enumerate(dataloader):
        video_id = data['video_id'][0]
        print(f"Processing {video_id}...")
        
        # 准备数据 (假设 data 返回的结构)
        ego_video = data['ego_video'].to(opt.device)[0] # [T,C,H,W]
        exo_gt = data['exo_video'].to(opt.device)[0]
        # 注意: Dataset 需要返回 Pillow 对象给 ref_image，或者在这里转换
        # 这里假设 dataloader 里的 ref_image 已经是 Tensor，AC/BSC 可能需要 PIL
        # 建议在 Dataset 里处理好，或者在此处转换
        # pillow_ref = ... 
        
        gen_video = submission_loader.get_generated_video(video_id)
        if gen_video is None:
            continue
        gen_video = gen_video.to(opt.device)

        metrics = {}
        for name, evaluator in evaluators.items():
            try:
                score = evaluator.compute(
                    tensor_gen=gen_video,
                    tensor_gt=exo_gt,
                    tensor_ego=ego_video,
                    # pillow_ref=pillow_ref, 
                    video_id=video_id,
                    global_cache=global_cache
                )
                metrics[name] = score
            except Exception as e:
                print(f"  [Error] {name}: {e}")
                metrics[name] = 0.0
        
        recorder.update(video_id, metrics)
        
        # [Fix] 显存/内存优化：清除当前 video_id 相关的缓存
        # 遍历 cache keys，删除包含当前 video_id 的项
        keys_to_remove = [k for k in global_cache.keys() if str(video_id) in k]
        for k in keys_to_remove:
            del global_cache[k]
            
        # 可选：定期清空 CUDA 缓存
        if i % 10 == 0:
            torch.cuda.empty_cache()

    recorder.save_report()
    
    
# 1. 初始化 Loader
opt = TestOptions()
data_loader = Ego2ExoDataLoader(opt)
dataset = data_loader.load_data()

# 2. 遍历数据进行 Benchmark
for i, data in enumerate(dataset):
    print(f"Processing ID: {data['video_id']}")
    
    # 获取数据
    ego_video = data['ego_video'].cuda() # [B, T, C, H, W]
    ref_img = data['ref_image'].cuda()   # [B, C, H, W]
    prompt = data['pos_prompt']          # List of strings
    
    # 3. 运行模型 (假设 model 是你的生成模型)
    # generated_video = model(ego_video, ref_img, prompt)
    
    # 4. 计算指标 (使用之前设计的 Evaluators)
    # score = evaluator.compute(generated_video, ...)
if __name__ == "__main__":
    main()