# tests/dimension/test_bsc.py

import os
import glob
import torch
import numpy as np
from tqdm import tqdm
import logging

from follow_bench.dimension.bsc import BackgroundSemanticConsistencyEvaluator

# 视频加载工具兼容性处理
try:
    from follow_bench.utils.video_kit import load_video_to_device
except ImportError:
    from torchvision.io import read_video
    def load_video_to_device(path, device):
        v, _, _ = read_video(path, output_format="TCHW")
        return v.float().div(255.0).to(device)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestBSC")

def run_bsc_test():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 只需定义视频目录
    GEN_ROOT = "assets/test_gen" 
    
    # 查找视频
    gen_files = sorted(glob.glob(os.path.join(GEN_ROOT, "*", "*.mp4")))
    
    if not gen_files:
        print(f"❌ 未在 {GEN_ROOT} 下找到任何 .mp4 文件。")
        return

    print(f"🚀 找到 {len(gen_files)} 个视频，开始 BSC (Continuity) 测试...")
    print("ℹ️  注意：新版 BSC 衡量背景的【时间连续性】，不需要参考图。")

    evaluator = BackgroundSemanticConsistencyEvaluator(device=device)
    print("⏳ 正在加载模型...")
    evaluator.prepare()
    print("✅ 模型加载完成")
    
    scores = []
    global_cache = {} 
    
    for gen_path in tqdm(gen_files, desc="Computing BSC"):
        try:
            # 1. 获取 Video ID
            relative_path = os.path.relpath(gen_path, GEN_ROOT)
            case_folder_name = os.path.dirname(relative_path)
            if not case_folder_name: # 处理根目录下的视频
                case_folder_name = os.path.splitext(os.path.basename(gen_path))[0]

            # 2. 加载视频
            tensor_gen = load_video_to_device(gen_path, device=device)
            if tensor_gen is None: continue

            # 3. 计算 (不再传入 pillow_ref)
            score = evaluator.compute(
                tensor_gen=tensor_gen,
                video_id=case_folder_name,
                global_cache=global_cache
            )
            
            scores.append(score)
            print(f"[{case_folder_name}] BSC: {score:.4f}")
            
            # 显存清理
            del tensor_gen

        except Exception as e:
            logger.error(f"Error processing {case_folder_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    if scores:
        print(f"\n✅ 测试完成！")
        print(f"   平均 BSC (Continuity) 分数: {np.mean(scores):.4f}")
    else:
        print("\n❌ 没有成功计算任何样本。")

if __name__ == "__main__":
    run_bsc_test()