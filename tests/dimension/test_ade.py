import os
import glob
import torch
import numpy as np
from tqdm import tqdm
import logging
import sys

# 确保能导入项目模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from follow_bench.dimension.ade import AverageDisplacementErrorEvaluator
from follow_bench.utils.video_kit import load_video_to_device

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestADE")

def run_ade_test():
    # ================= 配置区域 =================
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    GEN_ROOT = "assets/test_gen"
    GT_ROOT = "assets/dataset/test_unseen"
    
    # 查找所有生成视频
    gen_files = sorted(glob.glob(os.path.join(GEN_ROOT, "*", "*.mp4")))
    
    if not gen_files:
        print(f"❌ 未在 {GEN_ROOT} 下找到任何 .mp4 文件，请检查路径。")
        return

    print(f"🚀 找到 {len(gen_files)} 个视频，开始 ADE (Average Displacement Error) 测试...")
    print(f"ℹ️  ADE 衡量轨迹误差 (0.0 = 完美重合, Lower is Better)")
    
    # ================= 初始化评估器 =================
    evaluator = AverageDisplacementErrorEvaluator(device=device)
    evaluator.prepare()
    
    scores = []
    global_cache = {} 
    
    # ================= 循环评测 =================
    progress = tqdm(gen_files, desc="Computing ADE")
    for gen_path in progress:
        try:
            # 解析 Case ID (假设结构: .../case_id/exo.mp4)
            rel_path = os.path.relpath(gen_path, GEN_ROOT)
            case_id = os.path.dirname(rel_path)
            
            # 构造 GT 路径
            gt_path = os.path.join(GT_ROOT, rel_path)
            
            if not os.path.exists(gt_path):
                # logger.warning(f"缺少 GT: {gt_path}")
                continue

            # 1. 加载视频
            tensor_gen = load_video_to_device(gen_path, device=device)
            tensor_gt = load_video_to_device(gt_path, device=device)
            
            if tensor_gen is None or tensor_gt is None:
                continue

            # 2. 计算 ADE
            # 需要同时传入 gen 和 gt
            score = evaluator.compute(
                tensor_gen=tensor_gen,
                tensor_gt=tensor_gt,
                video_id=case_id,
                global_cache=global_cache
            )
            
            scores.append(score)
            print(f"[{case_id}] ADE: {score:.4f}")
            progress.set_postfix({"Last Score": f"{score:.4f}"})
            
            # 显存清理
            del tensor_gen
            del tensor_gt

        except Exception as e:
            logger.error(f"处理 {case_id} 时出错: {e}")
            continue

    # ================= 结果汇总 =================
    if scores:
        avg_score = np.mean(scores)
        print(f"\n✅ 测试完成！")
        print(f"   有效样本数: {len(scores)}")
        print(f"   平均 ADE 分数: {avg_score:.4f}")
    else:
        print("\n❌ 没有成功计算任何样本，请检查 GT 路径是否匹配。")

    evaluator.clear()

if __name__ == "__main__":
    run_ade_test()