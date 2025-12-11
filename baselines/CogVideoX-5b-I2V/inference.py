import torch
from diffusers import CogVideoXImageToVideoPipeline
from diffusers.utils import export_to_video, load_image
import argparse
import os
import yaml
import logging

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config
logging.info("Loading configuration from config.yml...")
config = load_config('baselines/CogVideoX-5b-I2V/config.yml')
REF_IMG_PATH = config['REF_IMG_PATH']
OUTPUT_PATH = config['OUTPUT_PATH']
CKPT_PATH = config['CKPT_PATH']
PROMPT = config['PROMPT']
NEGATIVE_PROMPT = config['NEGATIVE_PROMPT']

SEED = config['SEED']
GUIDANCE_SCALE = config['GUIDANCE_SCALE']
NUM_INFERENCE_STEPS = config['NUM_INFERENCE_STEPS']
USE_DYNAMIC_CFG = True


def main():
    print(f"初始化 CogVideoX-I2V Pipeline...")
    
    # 1. 加载模型
    # CogVideoX 使用 bfloat16 精度以节省显存并保持质量
    pipe = CogVideoXImageToVideoPipeline.from_pretrained(
        CKPT_PATH,
        torch_dtype=torch.bfloat16
    ).to("cuda")

    # 2. 加载并预处理图片
    if not os.path.exists(REF_IMG_PATH):
        raise FileNotFoundError(f"参考图未找到: {REF_IMG_PATH}")
    
    print(f"加载参考图: {REF_IMG_PATH}")
    image = load_image(REF_IMG_PATH)

    # 4. 执行生成
    print(f"开始生成视频...")
    print(f"   Prompt: {PROMPT}")
    generator = torch.Generator(device="cuda").manual_seed(SEED)
    video = pipe(
        prompt=PROMPT,
        image=image, # 输入参考图
        negative_prompt=NEGATIVE_PROMPT,
        num_videos_per_prompt=1,
        num_inference_steps=NUM_INFERENCE_STEPS,
        num_frames=49, # CogVideoX-5b 通常生成 49 帧 (约 6 秒 @ 8fps)
        guidance_scale=GUIDANCE_SCALE,
        use_dynamic_cfg=USE_DYNAMIC_CFG,
        generator=generator,
    ).frames[0]

    # 5. 保存视频
    print(f"保存视频至: {OUTPUT_PATH}")
    export_to_video(video, OUTPUT_PATH, fps=8)
    print("完成！")

if __name__ == "__main__":
    main()