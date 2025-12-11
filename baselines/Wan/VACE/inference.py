import os
import logging
import torch
import torch.distributed as dist
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
import torch
import yaml
import sys
import subprocess
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.append(project_dir)

from diffusers import DiffusionPipeline
from diffusers.utils import load_image, export_to_video

from wan.configs import WAN_CONFIGS, SIZE_CONFIGS
from wan.vace import WanVace
from wan.utils.utils import cache_video

from utils.get_free_gpus import get_free_gpu_ids

logging.basicConfig(level=logging.INFO)

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

logging.info("Loading configuration from config.yml...")
config = load_config('baselines/Wan/VACE/config.yml')
TASK = config['TASK']
CKPT_DIR = config['CKPT_DIR']
REF_IMG_PATH = config['REF_IMG_PATH']
EGO_VIDEO_PATH = config['EGO_VIDEO_PATH']   
OUTPUT_PATH = config['OUTPUT_PATH']
SIZE_KEY = config['SIZE_KEY']
TARGET_W, TARGET_H = map(int, SIZE_KEY.split('*'))
FRAME_NUM = config['FRAME_NUM']
PROMPT = config['PROMPT']

def generate_full_mask(video_path, size):
    """
    生成全屏 Mask 视频 (全白 MP4)，用于指示模型重新生成整个画面。
    必须生成视频文件，因为 WanVace 的底层 decord 不支持读取单张图片。
    """
    width, height = size
    
    # 1. 读取原视频的属性 (FPS 和 帧数)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    # 2. 设置 Mask 视频路径 (改为 .mp4)
    mask_dir = os.path.dirname(video_path)
    mask_path = os.path.join(mask_dir, "temp_full_mask.mp4")
    
    # 3. 创建视频写入器
    # 使用 mp4v 编码器生成 mp4 文件
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(mask_path, fourcc, fps, (width, height), isColor=True)
    
    # 4. 生成全白帧并写入
    # 255 代表全白 (Reactive区域，即生成区域)
    # 注意：WanVace 内部会读取 RGB，所以我们需要 3 通道的白色
    white_frame = np.full((height, width, 3), 255, dtype=np.uint8)
    
    print(f"正在生成 Mask 视频 ({total_frames} 帧)...")
    for _ in range(total_frames):
        out.write(white_frame)
        
    out.release()
    
    print(f"Mask 视频已生成: {mask_path}")
    return mask_path

def process_ref_image_center_crop(image_path, crop_ratio=1/3):
    """
    读取参考图，并只保留中间 1/3 区域 (以此作为人物特征)，其余部分变黑或裁剪。
    这里我们选择裁剪并Resize回原比例，或者填充黑色背景只留中间。
    策略：创建一个新图，只把原图中间部分贴进去，其余部分为黑色（或透明）。
    这能有效去除参考图背景对生成的干扰。
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Reference image not found: {image_path}")
    
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    
    # 计算中间区域
    center_w = w * crop_ratio
    left = (w - center_w) // 2
    right = left + center_w
    
    # 裁剪中间部分
    crop = img.crop((left, 0, right, h))
    
    # 策略：为了保持长宽比输入，我们将裁剪部分贴在一个黑色背景中心
    # 或者直接使用裁剪后的图（CLIP 会自动 Resize，但可能会变形）
    # 推荐：创建一个与原图等宽高的黑底图，将人物贴在中间
    new_img = Image.new("RGB", (w, h), (0, 0, 0))
    paste_x = int((w - crop.size[0]) // 2)
    new_img.paste(crop, (paste_x, 0))
    
    # 保存临时处理后的图片用于调试 (可选)
    processed_path = image_path.replace(".png", "_processed.png").replace(".jpg", "_processed.jpg")
    new_img.save(processed_path)
    logging.info(f"Processed Reference Image (Center 1/3) saved to: {processed_path}")
    return processed_path

def inference_worker():
    # 1. 初始化进程组
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    
    if global_rank == 0:
        logging.basicConfig(level=logging.INFO)
        logging.info(f"启动分布式推理 (World Size: {world_size})")
    else:
        logging.basicConfig(level=logging.ERROR)
    
    # 2. 加载模型
    wan_vace = WanVace(
        config=WAN_CONFIGS[TASK],
        checkpoint_dir=CKPT_DIR,
        device_id=local_rank,
        rank=global_rank,
        t5_fsdp=True,
        dit_fsdp=True,
        use_usp=False,
        t5_cpu=False
    )
    
    # 数据准备
    mask_path = os.path.join(os.path.dirname(EGO_VIDEO_PATH), "temp_dist_mask.mp4")
    if global_rank == 0:
        logging.info("主进程正在生成 Mask...")
        cap = cv2.VideoCapture(EGO_VIDEO_PATH)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(mask_path, fourcc, fps, (TARGET_W, TARGET_H), isColor=True)
        white_frame = np.full((TARGET_H, TARGET_W, 3), 255, dtype=np.uint8)
        for _ in range(total_frames):
            out.write(white_frame)
        out.release()
    dist.barrier()
    
    # 4. 预处理输入
    # prepare_source 内部会自动将数据移动到当前 device
    src_video, src_mask, src_ref = wan_vace.prepare_source(
        src_video=[EGO_VIDEO_PATH],
        src_mask=[mask_path],
        src_ref_images=[[REF_IMG_PATH]],
        num_frames=FRAME_NUM,
        image_size=(TARGET_H, TARGET_W), # H, W
        device=device
    )
    
    # 5. 执行推理生成任务
    if global_rank == 0: logging.info("开始 FSDP 并行推理...")
    video = wan_vace.generate(
        input_prompt=PROMPT,
        input_frames=src_video,
        input_masks=src_mask,
        input_ref_images=src_ref,
        size=(TARGET_W, TARGET_H),
        frame_num=FRAME_NUM,
        shift=5.0,
        sample_solver='unipc',
        sampling_steps=50,
        guide_scale=5.0,
        seed=42,
        offload_model=False 
    )
    
    # 6. 保存结果
    if global_rank == 0:
        if video is not None:
            os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
            cache_video(
                tensor=video[None], 
                save_file=OUTPUT_PATH,
                fps=16,
                nrow=1,
                normalize=True,
                value_range=(-1, 1)
            )
            print(f"推理成功！视频已保存至: {OUTPUT_PATH}")
            if os.path.exists(mask_path): os.remove(mask_path)
        else:
            print("推理失败 (Result is None)")
    dist.destroy_process_group()
    
def launch_dist_inference():
    """
    主控逻辑：检测 GPU -> 构建命令 -> 启动子进程
    """
    # 1. 自动检测空闲 GPU
    free_gpus = get_free_gpu_ids()
    if not free_gpus:
        print("没有检测到空闲 GPU")
        return
    gpu_list = free_gpus.split(',')
    num_gpus = len(gpu_list)
    print(f"检测到 {num_gpus} 张空闲显卡: {free_gpus}")

    # 2. 构建 torchrun 命令 我们将当前脚本作为 worker 重新运行，但带有 LOCAL_RANK 环境变量
    cmd = [
        "torchrun",
        f"--nproc_per_node={num_gpus}",
        "--master_port=4444",
        sys.argv[0],  # 重新运行自己
        "--worker"    # 标记为 worker 模式
    ]

    # 3. 设置环境变量，只让 PyTorch 看到选定的卡
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = free_gpus
    
    print(f"启动命令: {' '.join(cmd)}")
    subprocess.run(cmd, env=env)
    
if __name__ == "__main__":
    # 如果带有 --worker 参数，说明是被 torchrun 启动的子进程，执行推理逻辑
    if "--worker" in sys.argv:
        inference_worker()
    else:
        # 否则是主启动脚本，负责检测 GPU 并启动 torchrun
        launch_dist_inference()