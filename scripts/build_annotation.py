import os
import json
import random
import argparse
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Build annotation.json for Ego2Exo Benchmark")
    parser.add_argument('--data_root', type=str, required=True, help="Root directory containing First_Video, Third_Video, Reference_Image")
    parser.add_argument('--output_dir', type=str, default='./assets/Ego2ExoDataset', help="Output directory for formatted dataset")
    parser.add_argument('--split_ratio', type=float, default=0.9, help="Train/Test split ratio (default 0.9 for train)")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    return parser.parse_args()

def scan_files(data_root):
    """
    扫描目录并配对文件。
    假设文件名格式一致，例如:
    Ego: {id}_part_{seq}.mp4
    Exo: {id}_part_{seq}.mp4 (或者通过某种规则匹配)
    Ref: {id}_part_{seq}.png
    
    这里根据您的 pipeline.py 逻辑:
    FPV: {i}-1_slice_{j}.mp4 (假设) -> 您的 pipeline 输出是 {i}-1
    TPV: {i}-3_slice_{j}.mp4
    Ref: {i}-3_slice_{j}.png
    我们需要一种通用的 ID 提取方式。
    """
    ego_dir = Path(data_root) / 'First_Video'
    exo_dir = Path(data_root) / 'Third_Video'
    ref_dir = Path(data_root) / 'Reference_Image'
    
    samples = []
    
    # 遍历 Exo 视频 (通常作为 GT)
    for exo_path in exo_dir.glob('*.mp4'):
        # 假设文件名类似: 1-3_part_001.mp4
        # 对应的 Ego 应该是: 1-1_part_001.mp4 (根据您的业务逻辑调整)
        fname = exo_path.name
        
        # [逻辑适配] 这是一个示例 ID 转换，请根据实际文件名修改
        # 假设 exo: "X-3_part_Y.mp4" -> ego: "X-1_part_Y.mp4"
        if "-3_" in fname:
            ego_fname = fname.replace("-3_", "-1_")
        else:
            # Fallback
            ego_fname = fname 
            
        ego_path = ego_dir / ego_fname
        ref_path = ref_dir / (fname.replace('.mp4', '.png'))
        
        if ego_path.exists() and ref_path.exists():
            samples.append({
                'id': fname.replace('.mp4', ''),
                'ego_path': str(ego_path.relative_to(data_root)), # 存储相对路径
                'exo_path': str(exo_path.relative_to(data_root)),
                'ref_path': str(ref_path.relative_to(data_root))
            })
        else:
            print(f"[Warning] Missing pair for {fname}: Ego={ego_path.exists()}, Ref={ref_path.exists()}")
            
    return samples

def generate_prompts(sample_id, phase='train'):
    """生成 Prompt (示例)"""
    # 这里可以读取外部txt，或者使用模板
    neg_prompt = "deformed, distorted, disfigured, doll, poorly drawn, bad anatomy, wrong anatomy"
    
    if phase == 'train':
        return {
            "positive": "a person dancing in the room, high quality, 4k",
            "negative": neg_prompt
        }
    else:
        # Test: 多级 Prompt
        return {
            "positive": {
                "for text only model": "a person dancing in the room",
                "for_text_image model": "a person matching the reference image dancing",
                "for fullymodal model": "transform the first person view to third person view, keep character consistent"
            },
            "negative": neg_prompt
        }

def main():
    args = parse_args()
    random.seed(args.seed)
    
    # 1. 扫描数据
    print(f"Scanning data from {args.data_root}...")
    samples = scan_files(args.data_root)
    print(f"Found {len(samples)} valid triplets.")
    
    # 2. 划分数据集
    random.shuffle(samples)
    split_idx = int(len(samples) * args.split_ratio)
    train_samples = samples[:split_idx]
    test_samples = samples[split_idx:]
    
    # 3. 构建 JSON 结构
    datasets = {'train': train_samples, 'test': test_samples}
    
    for phase, data_list in datasets.items():
        json_content = {}
        for item in data_list:
            json_content[item['id']] = {
                "the first view": item['ego_path'],
                "the third view": item['exo_path'],
                "reference image": item['ref_path'],
                "prompt": generate_prompts(item['id'], phase)
            }
            
        # 4. 保存
        out_dir = Path(args.output_dir) / phase
        out_dir.mkdir(parents=True, exist_ok=True)
        
        out_json = out_dir / 'annotation.json'
        with open(out_json, 'w', encoding='utf-8') as f:
            json.dump(json_content, f, indent=4)
            
        print(f"Saved {phase} annotations to {out_json} ({len(data_list)} samples)")

if __name__ == "__main__":
    main()