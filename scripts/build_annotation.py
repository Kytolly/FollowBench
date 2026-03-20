import os
import json
import logging
from pathlib import Path

# ================= 配置区域 =================
# 数据集根目录 (请根据实际情况修改，这里指向 assets/Test)
# 注意：生成的 json 中路径将是相对于此目录的相对路径 (e.g., "test-case-000/ego.mp4")
ASSETS_ROOT = r"assets/followbench/test_unseen"

# 输出文件路径
OUTPUT_FILE = r"templates/index.json"

# 默认 Prompt 模板 (因为脚本无法自动理解视频内容，这里使用占位符，您后续可以手动或用其他脚本填充)
# DEFAULT_PROMPT = "[EGO2EXO] [REF] a person [EGO] doing something [TARGET-VIEW] Third-person medium shot, environment"
# NEGATIVE_PROMPT = "shaking, blurry, distorted, low quality, pixelated, artifacts, text, watermark, signature"

# ===========================================

def build_annotation():
    root_path = Path(ASSETS_ROOT)
    if not root_path.exists():
        print(f"Error: Assets directory not found: {root_path}")
        return

    print(f"Scanning assets from: {root_path}")
    
    annotation_data = {}
    
    # 获取所有子目录并排序
    # 假设目录名为 test-case-xxx
    case_dirs = sorted([d for d in root_path.iterdir() if d.is_dir()])
    
    print(f"Found {len(case_dirs)} cases.")

    for case_dir in case_dirs:
        case_id = case_dir.name
        
        # 1. 构建文件相对路径
        # 注意：这里使用正斜杠 / 以保证跨平台兼容性和 JSON 格式标准
        ego_path = f"{case_id}/ego.mp4"
        exo_path = f"{case_id}/exo.mp4"
        ref_path = f"{case_id}/ref_img.jpg"

        # 3. 构造 Prompt 字典 (对齐 caption.json 格式)
        # 如果您有 csv 或其他元数据文件包含真实 prompt，可以在这里读取并替换 DEFAULT_PROMPT
        # prompts = {
        #     "t2v_generic": DEFAULT_PROMPT,
        #     "i2v_generic": DEFAULT_PROMPT,
        #     "vace_instruct": DEFAULT_PROMPT,
        #     "ours_lora": DEFAULT_PROMPT
        # }

        # 4. 组装 Entry
        annotation_data[case_id] = {
            "ego video path": ego_path,
            "exo video path": exo_path,
            "reference image path": ref_path,
            # "prompts": prompts,
            # "negative_prompt": NEGATIVE_PROMPT
        }

    # 确保输出目录存在
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    # 写入 JSON
    try:
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(annotation_data, f, indent=4, ensure_ascii=False)
        print(f"\n✅ Index file built successfully!")
        print(f"   Path: {OUTPUT_FILE}")
        print(f"   Total cases: {len(annotation_data)}")
        
        # 打印一个示例供检查
        if case_dirs:
            first_key = case_dirs[0].name
            print("\n[Example Entry]:")
            print(json.dumps({first_key: annotation_data[first_key]}, indent=4))
            
    except Exception as e:
        print(f"Failed to write output file: {e}")

if __name__ == "__main__":
    build_annotation()