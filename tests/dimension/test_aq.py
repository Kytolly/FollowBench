import os
import glob
import torch
import numpy as np
from tqdm import tqdm
import logging
import sys

# 确保能导入项目模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 尝试导入 Evaluator
try:
    from follow_bench.dimension.aq import AestheticQualityEvaluator
except ImportError:
    print("❌ 无法导入 AestheticQualityEvaluator，请检查 follow_bench/dimension/aq.py 是否存在")
    sys.exit(1)

# 视频加载工具
try:
    from follow_bench.utils.video_kit import load_video_to_device
except ImportError:
    from torchvision.io import read_video
    def load_video_to_device(path, device):
        if not os.path.exists(path): return None
        v, _, _ = read_video(path, output_format="TCHW")
        return v.float().div(255.0).to(device)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestAQ")

def run_aq_test():
    # ================= 配置区域 =================
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 生成视频目录
    GEN_ROOT = "assets/test_gen" 
    
    # 查找所有 .mp4 文件
    gen_files = sorted(glob.glob(os.path.join(GEN_ROOT, "*", "*.mp4")))
    
    if not gen_files:
        print(f"❌ 未在 {GEN_ROOT} 下找到任何 .mp4 文件，请检查路径。")
        return

    print(f"🚀 找到 {len(gen_files)} 个视频，开始 AQ (Aesthetic Quality) 测试...")
    print(f"ℹ️  AQ 分数通常基于 LAION-Aesthetics 模型 (范围通常 4.0 - 7.0，越高越好)")
    
    # ================= 初始化评估器 =================
    evaluator = AestheticQualityEvaluator(device=device)
    evaluator.prepare()
    
    scores = []
    global_cache = {} 
    
    # ================= 循环评测 =================
    progress = tqdm(gen_files, desc="Computing AQ")
    for gen_path in progress:
        try:
            case_id = os.path.basename(os.path.dirname(gen_path))
            
            # 1. 加载视频
            tensor_gen = load_video_to_device(gen_path, device=device)
            if tensor_gen is None:
                continue

            # 2. 计算 AQ
            # AQ 是无参考指标，只需要 tensor_gen
            score = evaluator.compute(
                tensor_gen=tensor_gen,
                video_id=case_id,
                global_cache=global_cache
            )
            
            scores.append(score)
            print(f"[{case_id}] AQ: {score:.4f}")
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
        print(f"   有效样本数: {len(scores)}")
        print(f"   平均 AQ 分数: {avg_score:.4f} (Higher is Better)")
    else:
        print("\n❌ 没有成功计算任何样本。")

    evaluator.clear()

if __name__ == "__main__":
    run_aq_test()