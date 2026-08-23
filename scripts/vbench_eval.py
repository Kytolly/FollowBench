"""
独立脚本: 用于计算指定视频文件夹的 VBench 综合评分。

使用方法:
1. 确保在 VBench 的专属环境 (conda activate vbench) 下运行此脚本
2. 运行命令示例:
   python scripts/eval_vbench.py \
       --video_folder data/closed_source_gen \
       --output_dir results/vbench_output \
       --dimensions subject_consistency motion_smoothness
"""

import os
import argparse
import sys
import json
from datetime import datetime

from vbench import VBench


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate video folder using VBench")
    
    parser.add_argument(
        "--video_folder", 
        type=str, 
        required=True, 
        help="Path to the directory containing generated videos (e.g., .mp4 files)."
    )
    
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="results/vbench_output", 
        help="Directory to save the VBench evaluation reports."
    )
    
    parser.add_argument(
        "--model_name", 
        type=str, 
        default="FollowBench_Gen_Model", 
        help="Name of the model being evaluated (used for report naming)."
    )

    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda", 
        choices=["cuda", "cpu"],
        help="Device to run VBench models on."
    )

    # 默认选出视频生成中最重要的 5 个维度
    default_dims = [
        "subject_consistency", 
        "background_consistency", 
        "motion_smoothness", 
        "temporal_flickering", 
        "aesthetic_quality"
    ]
    
    parser.add_argument(
        "--dimensions", 
        nargs="+", 
        default=default_dims,
        help="List of VBench dimensions to evaluate. Use 'all' for full evaluation."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 1. 检查输入路径
    if not os.path.exists(args.video_folder):
        print(f"❌ 错误: 视频文件夹 {args.video_folder} 不存在!")
        sys.exit(1)
        
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("🚀 开始执行 VBench 评测")
    print(f"📂 视频目录: {args.video_folder}")
    print(f"🎯 评测维度: {', '.join(args.dimensions)}")
    print(f"🤖 评测设备: {args.device}")
    print("=" * 60)

    # 2. 如果输入了 'all'，则让 VBench 跑全部维度
    dims_to_eval = args.dimensions
    if len(dims_to_eval) == 1 and dims_to_eval[0].lower() == 'all':
        # VBench 内部默认支持不传维度的全量评测，或者显式获取全部列表
        # 这里为了安全，我们如果不传具体的 list，就利用 VBench 自身的默认行为
        dims_to_eval = None 
        print("💡 提示: 将进行 VBench 全维度综合评测 (耗时较长)。")

    try:
        # 3. 初始化 VBench
        print("\n⏳ 正在初始化 VBench 核心组件 (首次运行可能需要下载模型权重)...")
        my_VBench = VBench(device=args.device, output_path=args.output_dir)
        # VBench(device, "vbench/VBench_full_info.json", "evaluation_results")
        # 4. 执行评估 (Custom Folder 模式)
        print("\n⏳ 开始逐视频提取特征并打分...")
        my_VBench.evaluate(
            videos_path=args.video_folder,
            name=args.model_name,
            dimension_list=dims_to_eval,
            mode='custom_folder' # 关键：告诉 VBench 这是一个纯视频文件夹
        )
        
        print("\n" + "=" * 60)
        print(f"✅ VBench 评测完成! 报告已保存至: {args.output_dir}")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ VBench 评测过程中发生错误:")
        print(e)
        sys.exit(1)

if __name__ == "__main__":
    '''
    export HTTP_PROXY="http://127.0.0.1:7890"
    export HTTPS_PROXY="https://127.0.0.1:7890"
    python scripts/vbench_eval.py \
    --video_folder assets/flat_eval/kling_flat \
    --model_name "Kling" \
    --output_dir ./results/kling_vbench_report
    '''
    
    main()