import os
import logging
import torch
import torch.distributed as dist
import numpy as np
import sys
import subprocess
import argparse
from PIL import Image
import yaml
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.append(project_dir)

from wan.configs import WAN_CONFIGS, MAX_AREA_CONFIGS
from wan.image2video import WanI2V
from wan.utils.utils import cache_video

from utils.get_free_gpus import get_free_gpu_ids

logging.basicConfig(level=logging.INFO)

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

logging.info("Loading configuration from config.yml...")
config = load_config('baselines/Wan/I2V/config.yml')
TASK = config['TASK']
CKPT_DIR = config['CKPT_DIR']
REF_IMG_PATH = config['REF_IMG_PATH']
OUTPUT_PATH = config['OUTPUT_PATH']
SIZE_KEY = config['SIZE_KEY']
TARGET_W, TARGET_H = map(int, SIZE_KEY.split('*'))
FRAME_NUM = config['FRAME_NUM']
PROMPT = config['PROMPT']

def broadcast_image(img_path, device, rank):
    """
    负责将图片从 Rank 0 广播到所有其他 Rank，确保输入一致。
    """
    if rank == 0:
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Input image not found: {img_path}")
        pil_img = Image.open(img_path).convert("RGB")
        
        # 【关键修改】在 permute 后必须加 .contiguous()
        # 否则该张量在内存中不连续，dist.broadcast 会报错
        img_tensor = torch.tensor(np.array(pil_img)).permute(2, 0, 1).float().contiguous() 
        
        # 发送尺寸信息 (H, W)
        size_tensor = torch.tensor([img_tensor.shape[1], img_tensor.shape[2]], dtype=torch.long, device=device)
    else:
        size_tensor = torch.zeros(2, dtype=torch.long, device=device)
    
    # 1. 广播尺寸
    dist.broadcast(size_tensor, src=0)
    h, w = size_tensor.tolist()
    
    # 2. 准备图片数据容器
    if rank != 0:
        img_tensor = torch.zeros((3, h, w), dtype=torch.float32, device=device)
    else:
        img_tensor = img_tensor.to(device)
        
    # 3. 广播图片数据
    dist.broadcast(img_tensor, src=0)
    
    # 转回 PIL Image
    img_np = img_tensor.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    return Image.fromarray(img_np)

def inference_worker():
    """
    实际执行推理的工作进程
    """
    # 1. 初始化进程组
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    
    # 仅在主进程打印日志
    if global_rank == 0:
        logging.basicConfig(level=logging.INFO)
        logging.info(f"启动 Wan-I2V 分布式推理 (World Size: {world_size})")
        logging.info(f"Task: {TASK}, Input: {REF_IMG_PATH}")
    else:
        logging.basicConfig(level=logging.ERROR)
    
    # 2. 加载配置
    if TASK not in WAN_CONFIGS:
        raise ValueError(f"Task {TASK} not found in WAN_CONFIGS")
    cfg = WAN_CONFIGS[TASK]
    
    # 3. 初始化 WanI2V 模型
    wan_i2v = WanI2V(
        config=cfg,
        checkpoint_dir=CKPT_DIR,
        device_id=local_rank,
        rank=global_rank,
        t5_fsdp=True, 
        dit_fsdp=True,
        use_usp=False, 
        t5_cpu=False, 
    )
    
    # 4. 数据准备 (主进程读取图片，但所有进程都需要参与 generate)
    img = broadcast_image(REF_IMG_PATH, device, global_rank)

    # 5. 执行推理
    if global_rank == 0: logging.info("开始生成视频...")
    max_area = MAX_AREA_CONFIGS.get(SIZE_KEY, 1280 * 720)
    video = wan_i2v.generate(
        input_prompt=PROMPT,
        img=img,                    # PIL Image
        max_area=max_area,          # I2V 使用面积控制
        frame_num=FRAME_NUM,
        shift=5.0,                  # 官方推荐: 720P用5.0, 480P用3.0
        sample_solver='unipc',
        sampling_steps=50,
        guide_scale=5.0,
        seed=42,
        offload_model=False         # 多卡并行时通常不需要频繁卸载
    )
    
    # 6. 保存结果 (仅主进程)
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

    # 2. 构建 torchrun 命令
    cmd = [
        "torchrun",
        f"--nproc_per_node={num_gpus}",
        "--master_port=29500", # 端口号可以修改
        sys.argv[0],           # 重新运行自己
        "--worker"             # 标记为 worker 模式
    ]

    # 3. 设置环境变量
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = free_gpus
    
    print(f"启动命令: {' '.join(cmd)}")
    subprocess.run(cmd, env=env)

if __name__ == "__main__":
    # 如果带有 --worker 参数，说明是被 torchrun 启动的子进程
    if "--worker" in sys.argv:
        inference_worker()
    else:
        # 否则是主启动脚本
        launch_dist_inference()