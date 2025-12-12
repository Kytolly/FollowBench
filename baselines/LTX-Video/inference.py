import os
import torch
import yaml
import logging  

from diffusers import LTXPipeline
from diffusers.utils import export_to_video

from captioner import VisualCaptioner

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config
logging.info("Loading configuration from config.yml...")
config = load_config('baselines/LTX-Video/config.yml')
REF_IMG_PATH = config['REF_IMG_PATH']
EGO_VIDEO_PATH = config['EGO_VIDEO_PATH']
OUTPUT_PATH = config['OUTPUT_PATH']
MVLM_CKPT_PATH = config['MVLM_CKPT_PATH']
LTXV_CKPT_PATH = config['LTXV_CKPT_PATH']
VLM_DEVICE = config.get('VLM_DEVICE', "cuda:0")
GEN_DEVICE = config.get('GEN_DEVICE', "cuda:1")
SEED = config['SEED']
FRAME_NUM = config['FRAME_NUM']
HEIGHT = config['HEIGHT']
WIDTH = config['WIDTH']

def main():
    # 1. 实例化 Captioner 
    captioner = VisualCaptioner(MVLM_CKPT_PATH, device=VLM_DEVICE)
    
    # 2. 生成角色描述
    print("分析参考图...")
    char_desc = captioner.describe_character(REF_IMG_PATH)
    print(f"角色描述: {char_desc}")
    
    # 3. 生成环境动作描述
    print("分析 Ego 视频...")
    env_desc = captioner.describe_action_scene(EGO_VIDEO_PATH)
    print(f"环境动作: {env_desc}")
    
    # 4. 释放 VLM 显存
    del captioner
    torch.cuda.empty_cache()

    # 5. 组合 Prompt
    final_prompt = (
        f"Third-person view. "
        f"{char_desc}. "
        f"{env_desc}. "
        f"Cinematic lighting, 4k, high quality."
    )
    print(f"最终 Prompt: \n{final_prompt}")
    if len(final_prompt) > 350:
        logging.warning(f"Prompt 过长 ({len(final_prompt)} chars)，正在截断...")
        final_prompt = final_prompt[:350]
        
    logging.info(f"最终 Prompt: \n{final_prompt}")
    
    # 6. LTX-Video 推理
    logging.info(f"正在 GPU {GEN_DEVICE} 上加载 LTX-Video 并生成...")
    pipe = LTXPipeline.from_pretrained(
        LTXV_CKPT_PATH, torch_dtype=torch.bfloat16
    ).to(GEN_DEVICE)

    video = pipe(
        prompt=final_prompt,
        negative_prompt="worst quality, blurry, distorted", # 简化负面提示
        width=WIDTH,
        height=HEIGHT,
        num_frames=FRAME_NUM,
        num_inference_steps=50,
        guidance_scale=3.0,
        generator=torch.manual_seed(SEED),
        max_sequence_length=128 
    ).frames[0]
    
    # 7. 保存
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    export_to_video(video, OUTPUT_PATH, fps=24)
    print(f"完成！视频已保存: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()