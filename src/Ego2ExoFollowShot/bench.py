import numpy as np
from PIL import Image
from scipy.stats import pearsonr
import yaml
import logging

import torch
from transformers import CLIPProcessor, CLIPModel
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn, 
    FasterRCNN_ResNet50_FPN_Weights,
    keypointrcnn_resnet50_fpn, 
    KeypointRCNN_ResNet50_FPN_Weights
)

from Ego2ExoFollowShot.dimension.metrics import *
from utils import *

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config
logging.info("Loading configuration from config.yml...")
config = load_config('benchmark/config.yml')
baseline_names = ['CogVideo', 'LTX', 'SVDXT', 'WanI2V', 'WanVACE']
baseline_outputs = {
    'CogVideo': config['BASELINE_OUTPUTS']['COGVIDEO'],
    'LTX': config['BASELINE_OUTPUTS']['LTX'],
    'SVDXT': config['BASELINE_OUTPUTS']['SVDXT'],
    'WanI2V': config['BASELINE_OUTPUTS']['WANI2V'],
    'WanVACE': config['BASELINE_OUTPUTS']['WANVACE'],
}
assets = {
    'Ego': config['ASSETS_PATH']['EGO'],
    'Exo': config['ASSETS_PATH']['EXO'],
    'Refexo': config['ASSETS_PATH']['REF'],
}
device = config['DEVICE']
print(f"Using device: {device}")

results = {
    'FVD': {},
    'AQ': {},
    'IQ': {},
    'TF': {},
    'MS': {},
    'DD': {},
    'CCE': {},
    'AC': {},
    'VV': {},
    'BSC': {},
    'OFC': {},
    'HAA': {},
    'TA': {},
}
# 预加载参考图和特征
ref_image = Image.open(assets['REF'])

# 预加载 Ego Video Flow
ego_frames = get_video_frames(assets['EGO'])
ego_flow_mags = calculate_optical_flow_magnitude(ego_frames)

# 预加载 CLIP
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
if clip_model:
    with torch.no_grad():
        inputs = clip_processor(images=ref_image, return_tensors="pt").to(device)
        ref_embed = clip_model.get_image_features(**inputs)
        ref_embed /= ref_embed.norm(dim=-1, keepdim=True)
clear_gpu_memory(clip_model)

# 预加载 EXO Video Features
exo_frames = get_video_frames(assets['EXO'])
gt_embeds = []
if clip_model:
    with torch.no_grad():
        for frame in exo_frames:
            inputs = clip_processor(images=frame, return_tensors="pt").to(device)
            emb = clip_model.get_image_features(**inputs)
            emb /= emb.norm(dim=-1, keepdim=True)
            gt_embeds.append(emb)
    if gt_embeds:
        gt_embeds_tensor = torch.cat(gt_embeds)
        gt_mean_embed = torch.mean(gt_embeds_tensor, dim=0, keepdim=True)

# 预加载 baselines frames tensors
gen_video = {}
gen_video_tensors = {}
for name in baseline_names:
    frames = get_video_frames(baseline_outputs[name])
    gen_video[name] = frames
    if not frames:
        print(f"Could not read video of baseline {[name]}")
        continue
    frames_tensor = torch.stack([torch.from_numpy(f) for f in frames])
    frames_tensor = frames_tensor.permute(0, 3, 1, 2).float() / 255.0
    frames_tensor = frames_tensor.to(device)
    gen_video_tensors[name] = frames_tensor

# 通用：加载模型 -> 遍历所有视频 -> 计算平均分 -> 清理显存
def calculate_metric_for_all_assets(
    metric_loader_func, 
    metric_name,
    device):
    metric_model = metric_loader_func(device)
    if metric_model is None: return {}
    
    scores = {}
    print(f"--- Starting {metric_name} Evaluation ---")
    for name in baseline_names:
        frames_tensor = gen_video_tensors[name]
        with torch.no_grad():
            val = metric_model(frames_tensor).mean().item()
        scores[name] = val
        print(f"[{name}] {metric_name}: {val:.4f}")
    print(f"Cleaning up {metric_name} model...")
    clear_gpu_memory(metric_model)
    return scores

def main():
    # --- temporal_consistency(Temporal Flickering & Motion Smoothness & Dynamic Degree) calculation ---
    for name in baseline_names:
        frames = gen_video[name]
        flicker, smooth, dynamic = calculate_temporal_consistency(frames)
        results['TF'][name] = flicker
        results['MS'][name] = smooth
        results['DD'][name] = dynamic
        
    # --- Optical Flow Correlation Calculation ---
    for name in baseline_names:
        gen_frames = gen_video[name]
        ofc_score = OpticalFlowCorrelation(gen_frames, exo_frames)
        results['OFC'][name] = ofc_score
    
    # --- Human Action Alignment Calculation ---
    weights = KeypointRCNN_ResNet50_FPN_Weights.DEFAULT
    model = keypointrcnn_resnet50_fpn(weights=weights).to(device)
    model.eval()
    for name in baseline_names:
        frames = gen_video[name]
        haa_score = HumanActionAlignment(frames, exo_frames, device)
        results['HAA'][name] = haa_score
    clear_gpu_memory(model)
    
    # --- Appearance Consistency calculation ---
    dinov2_model, dino_transform = load_dinov2(device)
    detection_model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT).to(device)
    detection_model.eval()
    ref_emb = prepare_ref_embedding(dinov2_model, dino_transform, ref_image)
    for name in baseline_names:
        frames = get_video_frames(baseline_outputs[name])
        app_score, center_err, validity = calculate_subject_metrics(
            dinov2_model, 
            dino_transform, 
            detection_model, 
            ref_emb, 
            frames
        )
        results['AC'][name] = app_score
        results['CCE'][name] = center_err
        results['VV'][name] = validity
    clear_gpu_memory(dinov2_model, dino_transform, detection_model, ref_emb)
    
    # --- Background Semantic Consistency Calculation ---
    for name in baseline_names:
        frames_tensor = gen_video_tensors[name]
        bg_score = BackgroundSemanticConsistency(
            clip_model=clip_model,
            clip_processor=clip_processor,
            detection_model=detection_model,
            ref_image=ref_image,
            video_tensor=frames_tensor
        )
        results['BSC'][name] = bg_score
    
    # --- Aesrhetic Quality calculation ---
    aes_results = calculate_metric_for_all_assets(
        load_aesthetic_metric, "Aesthetic Quality", device
    )
    results['AQ'] = aes_results
    
    # --- Imaging Quality calculation ---
    imgq_results = calculate_metric_for_all_assets(
        load_imaging_quality_metric, "Imaging Quality", device
    )
    results['IQ'] = imgq_results
    
    # --- FVD calculation ---
    # TODO: 确保 assets 为视频所在目录
    for name in baseline_names:
        fvd_val = FrechetVideoDistance(
            repo_path=config['FVD']['STYLEGANV_REPO_PATH'],
            real_videos_dir=assets['Exo'], 
            gen_videos_dir=assets[name],
            mirror=config['FVD']['MIRROR'],
            gpus=config['FVD']['GPUS'],
            resolution=config['FVD']['RESOLUTION'],
            metrics=config['FVD']['metrics'],
        )
        results['FVD'][name] = fvd_val
    torch.cuda.empty_cache()
    # FVD 结束后，subprocess 会自动释放它的显存。
    
    save_results(results, 'results.json')