import os
import argparse
import glob

def parse_args():
    parser = argparse.ArgumentParser(description="Flatten video directory structure using symlinks.")
    parser.add_argument(
        "--source_dir", 
        type=str, 
        required=True, 
        help="The root directory containing video files (can have subdirectories)."
    )
    parser.add_argument(
        "--target_dir", 
        type=str, 
        required=True, 
        help="The target directory where flat symlinks will be created."
    )
    parser.add_argument(
        "--ext", 
        type=str, 
        default=".mp4", 
        help="The video file extension to search for (default: .mp4)."
    )
    return parser.parse_args()

def flatten_directory(source_dir, target_dir, ext):
    # 确保源目录是绝对路径
    source_dir = os.path.abspath(source_dir)
    target_dir = os.path.abspath(target_dir)

    if not os.path.exists(source_dir):
        print(f"❌ 错误: 源目录 {source_dir} 不存在。")
        return

    # 创建目标目录
    os.makedirs(target_dir, exist_ok=True)

    # 构建搜索模式 (递归查找所有扩展名为 ext 的文件)
    search_pattern = os.path.join(source_dir, f"**/*{ext}")
    video_files = glob.glob(search_pattern, recursive=True)

    if not video_files:
        print(f"⚠️ 警告: 在 {source_dir} 中没有找到任何 {ext} 文件。")
        return

    print(f"🔍 找到 {len(video_files)} 个 {ext} 视频文件。准备创建软链接...")

    success_count = 0
    skip_count = 0

    for file_path in video_files:
        # 1. 获取相对于 source_dir 的相对路径
        # 例如: action_1/0001.mp4
        rel_path = os.path.relpath(file_path, source_dir)
        
        # 2. 将相对路径的目录分隔符替换为下划线，生成新的文件名
        # action_1/0001.mp4 -> action_1_0001.mp4
        new_filename = rel_path.replace(os.sep, "_")
        
        # 3. 构造目标软链接的完整路径
        target_link_path = os.path.join(target_dir, new_filename)
        
        # 4. 创建软链接
        try:
            # 如果目标链接已经存在，先删除它 (防止因为重复运行脚本导致报错)
            if os.path.lexists(target_link_path):
                os.remove(target_link_path)
                
            os.symlink(file_path, target_link_path)
            success_count += 1
        except Exception as e:
            print(f"❌ 创建 {new_filename} 的软链接失败: {e}")
            skip_count += 1

    print("=" * 50)
    print(f"✅ 展平完成!")
    print(f"📂 目标目录: {target_dir}")
    print(f"🔗 成功创建链接: {success_count} 个")
    if skip_count > 0:
        print(f"⚠️ 失败跳过: {skip_count} 个")
    print("=" * 50)


if __name__ == "__main__":
    '''
    python scripts/flat.py \
    --source_dir assets/kling \
    --target_dir assets/flat_eval/kling_flat
    
    python scripts/flat.py \
    --source_dir assets/runway \
    --target_dir assets/flat_eval/runway_flat
    
    python scripts/flat.py \
    --source_dir assets/seedance \
    --target_dir assets/flat_eval/seedance_flat
    
    python scripts/flat.py \
    --source_dir assets/worldwander/test_seen \
    --target_dir assets/flat_eval/worldwander/test_seen
    
    python scripts/flat.py \
    --source_dir assets/worldwander/test_unseen \
    --target_dir assets/flat_eval/worldwander/test_unseen
    '''
    
    args = parse_args()
    ext = args.ext if args.ext.startswith('.') else f".{args.ext}"
    
    flatten_directory(args.source_dir, args.target_dir, ext)