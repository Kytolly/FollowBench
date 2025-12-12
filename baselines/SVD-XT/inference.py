import torch
import os
import numpy as np
from PIL import Image
import yaml
import logging

from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import export_to_video

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

logging.info("Loading configuration from config.yml...")
config = load_config('baselines/SVD-XT/config.yml')
CKPT_DIR = config['CKPT_DIR']
REF_IMG_PATH = config['REF_IMG_PATH']
OUTPUT_PATH = config['OUTPUT_PATH']
SEED = config['SEED']
HEIGHT = config['HEIGHT']
WIDTH = config['WIDTH']
DECODE_CHUNK_SIZE = config['DECODE_CHUNK_SIZE']
FPS = config['FPS']

def main():
    pipe = StableVideoDiffusionPipeline.from_pretrained(
        CKPT_DIR, 
        torch_dtype=torch.float16, 
        variant="fp16"
    )
    # 自动将不用的组件卸载到 CPU，极大节省显存
    pipe.enable_model_cpu_offload()
    
    if not os.path.exists(REF_IMG_PATH):
        raise FileNotFoundError(f"参考图未找到: {REF_IMG_PATH}")
    image = Image.open(REF_IMG_PATH).convert("RGB")
    image = image.resize((WIDTH, HEIGHT), Image.LANCZOS)
    
    generator = torch.manual_seed(SEED)
    

    # motion_bucket_id: 控制运动幅度 (1-255)。默认 127。
    #   - 数值越大，动作幅度越大，但画面越容易崩坏。
    #   - 数值越小，画面越稳定，但趋于静止。
    # noise_aug_strength: 对参考图的加噪程度。微调保真度。  
    frames = pipe(
        image, 
        decode_chunk_size=DECODE_CHUNK_SIZE,
        generator=generator,
        motion_bucket_id=127,  # 标准动态
        noise_aug_strength=0.1,
        num_inference_steps=25
    ).frames[0]

    # 5. 保存结果
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    
    # SVD-XT 生成 25 帧。
    export_to_video(frames, OUTPUT_PATH, fps=FPS)

if __name__ == "__main__":
    main()