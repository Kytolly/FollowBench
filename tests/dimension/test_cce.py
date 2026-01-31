import os
import glob
import torch
import numpy as np
from tqdm import tqdm
import logging
import sys

# 确保能导入项目模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from follow_bench.dimension.cce import CameraCenteringErrorEvaluator
# 视频加载工具
try:
    from follow_bench.utils.video_kit import load_video_to_device
except ImportError:
    from torchvision.io import read_video
    def load_video_to_device(path, device):
        v, _, _ = read_video(path, output_format="TCHW")
        return v.float().div(255.0).to(device)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestCCE")

def run_cce_test():
    # ================= 配置区域 =================
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 生成视频的根目录
    # 请根据你的实际路径修改，例如 "assets/test_gen" 或 "assets/gen"
    GEN_ROOT = "assets/test_gen" 
    
    # 查找所有 .mp4 文件
    gen_files = sorted(glob.glob(os.path.join(GEN_ROOT, "*", "*.mp4")))
    
    if not gen_files:
        print(f"❌ 未在 {GEN_ROOT} 下找到任何 .mp4 文件，请检查路径。")
        return

    print(f"🚀 找到 {len(gen_files)} 个视频，开始 CCE (Camera Centering Error) 测试...")
    print(f"ℹ️  CCE 越低越好 (0.0 = 完美居中, 1.0 = 丢失目标或极度偏离)")
    
    # ================= 初始化评估器 =================
    evaluator = CameraCenteringErrorEvaluator(device=device)
    evaluator.prepare()
    
    scores = []
    global_cache = {} 
    
    # ================= 循环评测 =================
    progress = tqdm(gen_files, desc="Computing CCE")
    for gen_path in progress:
        try:
            case_id = os.path.basename(os.path.dirname(gen_path))
            
            # 1. 加载视频 [T, C, H, W]
            tensor_gen = load_video_to_device(gen_path, device=device)
            if tensor_gen is None:
                logger.warning(f"无法加载视频: {gen_path}")
                continue

            # 2. 计算 CCE
            # CCE 不需要 GT 或 Ref，只需要生成视频
            score = evaluator.compute(
                tensor_gen=tensor_gen,
                video_id=case_id,
                global_cache=global_cache
            )
            
            scores.append(score)
            progress.set_postfix({"Last Score": f"{score:.4f}"})
            
            # 显存清理
            del tensor_gen

        except Exception as e:
            logger.error(f"处理 {case_id} 时出错: {e}")
            continue

    # ================= 结果汇总 =================
    if scores:
        avg_score = np.mean(scores)
        print(f"\n✅ 测试完成！")
        print(f"   样本数: {len(scores)}")
        print(f"   平均 CCE 分数: {avg_score:.4f} (Lower is Better)")
    else:
        print("\n❌ 没有成功计算任何样本。")

    evaluator.clear()

if __name__ == "__main__":
    run_cce_test()