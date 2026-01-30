import os
import glob
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import logging

# 导入你的模块
from follow_bench.dimension.ac import AppearanceConsistencyEvaluator
# 如果 video_kit 不可用，可以使用 torchvision.io.read_video 替代
try:
    from follow_bench.utils.video_kit import load_video_to_device
except ImportError:
    # 简单的回退实现
    from torchvision.io import read_video
    def load_video_to_device(path, device):
        v, _, _ = read_video(path, output_format="TCHW")
        return v.float().div(255.0).to(device)

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestAC")

def run_ref_based_test():
    # ================= 配置区域 =================
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 定义目录
    # 1. 生成视频目录 (假设结构: assets/test_gen/test_unseen_case_xxxx/exo.mp4)
    GEN_ROOT = "assets/test_gen" 
    
    # 2. 参考图根目录
    # 结构: assets/dataset/test_unseen/test_unseen_case_xxxx/ref_img.jpg
    REF_ROOT = "assets/dataset/test_unseen"
    
    # 查找所有生成视频
    # 假设每个 case 是一个文件夹
    gen_files = sorted(glob.glob(os.path.join(GEN_ROOT, "*", "*.mp4")))
    
    if not gen_files:
        print(f"❌ 未在 {GEN_ROOT} 下找到任何 .mp4 文件，请检查路径。")
        return

    print(f"🚀 找到 {len(gen_files)} 个生成视频，开始基于参考图 (ref_img.jpg) 的 AC 测试...")
    
    # ================= 初始化评估器 =================
    evaluator = AppearanceConsistencyEvaluator(device=device)
    evaluator.prepare()
    
    scores = []
    global_cache = {} 
    
    # ================= 循环评测 =================
    for gen_path in tqdm(gen_files, desc="Computing AC"):
        try:
            # 1. 解析 Case ID
            # gen_path: assets/gen/test_unseen_case_1001/exo.mp4
            # relative_path: test_unseen_case_1001/exo.mp4
            # case_folder: test_unseen_case_1001
            relative_path = os.path.relpath(gen_path, GEN_ROOT)
            case_folder_name = os.path.dirname(relative_path)
            
            # 2. 构造参考图路径
            # 目标: assets/test_unseen/test_unseen_case_1001/ref_img.jpg
            ref_img_path = os.path.join(REF_ROOT, case_folder_name, "ref_img.jpg")
            
            if not os.path.exists(ref_img_path):
                # 尝试其他后缀 (.png)
                ref_img_path_png = ref_img_path.replace(".jpg", ".png")
                if os.path.exists(ref_img_path_png):
                    ref_img_path = ref_img_path_png
                else:
                    logger.warning(f"缺少参考图: {ref_img_path} (Case: {case_folder_name})")
                    continue

            # 3. 加载数据
            # 加载视频
            tensor_gen = load_video_to_device(gen_path, device=device)
            if tensor_gen is None: continue
            
            # 加载参考图 (PIL Image)
            try:
                pillow_ref = Image.open(ref_img_path).convert("RGB")
            except Exception as e:
                logger.error(f"无法读取参考图 {ref_img_path}: {e}")
                continue

            # 4. 运行 AC 计算 (传入 pillow_ref)
            score = evaluator.compute(
                tensor_gen=tensor_gen,
                pillow_ref=pillow_ref,   # 传入参考图
                video_id=case_folder_name,
                global_cache=global_cache
            )
            
            scores.append(score)
            
            # Debug 输出 (可选)
            # print(f"AC Score: {case_folder_name} = {score:.4f}")

            # 5. 显存清理
            del tensor_gen
            # 如果显存紧张，可以每 N 轮调用 evaluator.clear() 或者 torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"处理 {case_folder_name} 时出错: {e}")
            import traceback
            traceback.print_exc()
            continue

    # ================= 结果汇总 =================
    if scores:
        avg_ac = np.mean(scores)
        print(f"\n✅ 测试完成！")
        print(f"   有效样本数: {len(scores)}")
        print(f"   平均 AC 分数: {avg_ac:.4f}")
    else:
        print("\n❌ 没有成功计算任何样本。")

if __name__ == "__main__":
    run_ref_based_test()