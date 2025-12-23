import os
import json
from pathlib import Path

# ================= 配置区域 =================
# 您的真实数据根目录
ASSETS_ROOT = r"assets/test"

# 输出文件路径 (根据您之前的上下文，放在 cache/gen 下)
OUTPUT_FILE = r"templates/submission_example.json"

# 元数据配置
META_INFO = {
    "team_name": "MyTeam",
    "model_name": "EgoGen-V1",
    # "modal": "vace_instruct",
    # "mode": "easy",
    "contact": "email@example.com"
}
# ===========================================

def build_submission():
    print(f"Scanning assets from: {ASSETS_ROOT}")
    
    if not os.path.exists(ASSETS_ROOT):
        print(f"Error: Directory not found: {ASSETS_ROOT}")
        return

    results = {}
    
    # 获取 assets/Test 下的所有子文件夹 (即 test-case-xxx)
    try:
        case_dirs = [d for d in os.listdir(ASSETS_ROOT) if os.path.isdir(os.path.join(ASSETS_ROOT, d))]
        # 简单排序，保证 JSON 有序
        case_dirs.sort()
        
        print(f"Found {len(case_dirs)} cases.")

        for case_id in case_dirs:
            # 构造绝对路径: D:\...\test-case-xxx\exo.mp4
            case_path = os.path.join(ASSETS_ROOT, case_id)
            exo_video_path = os.path.join(case_path, "exo.mp4")
            
            # (可选) 检查该文件是否存在，如果不存在打印警告
            if not os.path.exists(exo_video_path):
                print(f"[Warning] GT video not found for {case_id}: {exo_video_path}")
            
            # 按照要求的结构填入
            results[case_id] = {
                "generated video": f"{case_id}/exo.mp4",
                # "prompt": "[EGO2EXO] [REF] a woman [EGO] walking [TARGET-VIEW] Third-person medium shot, urban road"
            }
            
    except Exception as e:
        print(f"Error during scanning: {e}")
        return

    # 构建完整的 JSON 数据
    submission_data = {
        "meta": META_INFO,
        "results": results
    }

    # 确保输出目录存在
    output_dir = os.path.dirname(OUTPUT_FILE)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # 写入文件
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(submission_data, f, indent=4, ensure_ascii=False)
        
    print(f"\n✅ Submission file generated successfully!")
    print(f"   Path: {OUTPUT_FILE}")
    print(f"   Total entries: {len(results)}")

if __name__ == "__main__":
    build_submission()