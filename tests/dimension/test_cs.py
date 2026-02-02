import os
import glob
import torch
import numpy as np
from tqdm import tqdm
import logging

# [Import] 导入 CTE 评估器
from follow_bench.dimension.cs import CameraStabilityEvaluator

# 视频加载工具兼容性处理
try:
    from follow_bench.utils.video_kit import load_video_to_device
except ImportError:
    # 回退方案：使用 torchvision 读取
    from torchvision.io import read_video
    def load_video_to_device(path, device):
        # read_video 返回 [T, H, W, C] (0-255, uint8)
        # 我们需要转为 [T, C, H, W] (0.0-1.0, float) 并移至 GPU
        v, _, _ = read_video(path, output_format="TCHW")
        return v.float().div(255.0).to(device)

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestCTE")

def run_cs_test():
    # ================= 配置区域 =================
    # 强制使用 CUDA，因为核心算法依赖 torch.fft 加速
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if device == 'cpu':
        logger.warning("⚠️ 未检测到 CUDA！CTE 计算将回退到 CPU，速度可能会变慢且无法利用 FFT 加速优势。")

    # 定义生成视频目录
    # CTE 是无参考指标 (Blind Metric)，不需要 Reference Image 或 Ground Truth Video
    GEN_ROOT = "assets/test_gen" 
    
    # 查找所有生成视频
    gen_files = sorted(glob.glob(os.path.join(GEN_ROOT, "*", "*.mp4")))
    
    if not gen_files:
        print(f"❌ 未在 {GEN_ROOT} 下找到任何 .mp4 文件，请检查路径。")
        return

    print(f"🚀 找到 {len(gen_files)} 个生成视频，开始 CTE (Camera Trajectory Error) 测试...")
    print("ℹ️  原理: 基于 GPU 相位相关法 (Phase Correlation) 计算全局背景移动的平滑度。")
    print("ℹ️  指标: Jitter Score (0.0 = 完美平滑/匀速, 数值越高表示抖动越剧烈)。")
    
    # ================= 初始化评估器 =================
    try:
        evaluator = CameraStabilityEvaluator(device=device)
        # 本算法不需要预加载权重，prepare 为空操作，但为了规范性保留调用
        evaluator.prepare() 
        print("✅ 评估器初始化完成")
    except Exception as e:
        print(f"❌ 评估器初始化失败: {e}")
        return
    
    scores = []
    
    # ================= 循环评测 =================
    for gen_path in tqdm(gen_files, desc="Computing CTE"):
        tensor_gen = None
        try:
            # 1. 解析 Case ID
            relative_path = os.path.relpath(gen_path, GEN_ROOT)
            case_folder_name = os.path.dirname(relative_path)
            if not case_folder_name:
                case_folder_name = os.path.splitext(os.path.basename(gen_path))[0]

            # 2. 加载视频 (直接加载到 GPU)
            tensor_gen = load_video_to_device(gen_path, device=device)
            if tensor_gen is None or len(tensor_gen) < 2: 
                logger.warning(f"视频无效或太短: {gen_path}")
                continue

            # 3. 运行 CTE 计算
            # CTE 只需要 tensor_gen
            score = evaluator.compute(
                tensor_gen=tensor_gen,
                video_id=case_folder_name
            )
            
            scores.append(score)
            
            # Debug 输出 (可选)
            # print(f"[{case_folder_name}] Jitter: {score:.4f}")

        except Exception as e:
            logger.error(f"处理 {case_folder_name} 时出错: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # 4. 显存清理
            if tensor_gen is not None:
                del tensor_gen
            # 本算法显存占用极小，通常不需要频繁 empty_cache

    # ================= 结果汇总 =================
    if scores:
        avg_score = np.mean(scores)
        print(f"\n✅ 测试完成！")
        print(f"   有效样本数: {len(scores)}")
        print(f"   平均 CTE 分数 (Jitter): {avg_score:.4f}")
        print(f"   (注: 对于高质量生成的 Follow Camera 视频，分数通常应 < 0.2)")
    else:
        print("\n❌ 没有成功计算任何样本。")

if __name__ == "__main__":
    run_cs_test()