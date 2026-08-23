import os
import glob
import torch
import numpy as np
from tqdm import tqdm
import logging

from follow_bench.dimension.dd import DynamicDegreeEvaluator

# 视频加载工具
try:
    from follow_bench.utils.video_kit import load_video_to_device
except ImportError:
    from torchvision.io import read_video
    def load_video_to_device(path, device):
        v, _, _ = read_video(path, output_format="TCHW")
        return v.float().div(255.0).to(device)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestDD")

def run_dd_test():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    GEN_ROOT = "assets/test_gen" 
    
    gen_files = sorted(glob.glob(os.path.join(GEN_ROOT, "*", "*.mp4")))
    if not gen_files:
        print(f"❌ 未在 {GEN_ROOT} 下找到 .mp4 文件。")
        return

    print(f"🚀 找到 {len(gen_files)} 个视频，开始 DD (Dynamic Degree) 测试...")
    print("ℹ️  原理: 计算视频全局运动的平均速度幅值 (Velocity Magnitude)。")
    print("ℹ️  分数: 0.0 (静止) -> 1.0 (高速运动)。Follow Camera 视频通常在 0.2~0.6 之间。")
    
    evaluator = DynamicDegreeEvaluator(device=device)
    evaluator.prepare()
    
    scores = []
    
    for gen_path in tqdm(gen_files, desc="Computing DD"):
        tensor_gen = None
        try:
            relative_path = os.path.relpath(gen_path, GEN_ROOT)
            case_folder_name = os.path.dirname(relative_path)
            if not case_folder_name:
                case_folder_name = os.path.splitext(os.path.basename(gen_path))[0]

            # 加载视频到 GPU
            tensor_gen = load_video_to_device(gen_path, device=device)
            if tensor_gen is None: continue

            # 计算
            score = evaluator.compute(
                tensor_gen=tensor_gen,
                video_id=case_folder_name
            )
            
            scores.append(score)
            print(f"[{case_folder_name}] DD Score: {score:.4f}")

        except Exception as e:
            logger.error(f"处理 {case_folder_name} 时出错: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if tensor_gen is not None:
                del tensor_gen

    if scores:
        print(f"\n✅ 测试完成！")
        print(f"   平均 DD 分数 (Motion Magnitude): {np.mean(scores):.4f}")
    else:
        print("\n❌ 测试失败，无结果。")

if __name__ == "__main__":
    run_dd_test()