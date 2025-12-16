import subprocess
import sys
import os
import logging
import re

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

def FrechetVideoDistance(
    repo_path,
    real_videos_dir, 
    gen_videos_dir,
    mirror,
    gpus,
    resolution,
    metrics,):
    '''计算 FVD 需要视频矩阵'''
    if not repo_path or not os.path.exists(repo_path):
        logging.error(f"StyleGAN-V repo path not found: {repo_path}")
        return None
    script_path = os.path.join(repo_path, 'src', 'scripts', 'calc_metrics_for_dataset.py')
    if not os.path.exists(script_path):
        logging.error(f"Script not found at: {script_path}")
        return None
    
    cmd = [
        sys.executable, script_path,
        '--real_data_path', real_videos_dir,
        '--fake_data_path', gen_videos_dir,
        '--mirror', mirror,
        '--gpus', gpus,
        '--resolution', resolution,
        '--metrics', metrics,
    ]
    logging.info(f"Executing FVD calculation: {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd, 
            cwd=repo_path, 
            capture_output=True, 
            text=True, 
            env=os.environ.copy(),
            check=True
        )
        output_lines = result.stdout.split('\n')
        fvd_score = None
        for line in output_lines:
            if metrics in line:
                parts = line.split()
                matches = re.findall(r"[-+]?\d*\.\d+|\d+", line)
                if matches:
                    fvd_score = float(matches[-1])
        
        logging.info(f"FVD Calculation Success. Output: {fvd_score}")
        return fvd_score

    except subprocess.CalledProcessError as e:
        logging.error(f"FVD Calculation Failed with error:\n{e.stderr}")
        return None

def HumanActionAlignment(gen_frames, ref_frames, model, device):
    '''
    计算人物动作对齐 (Human Action Alignment)
    逻辑：提取骨骼关键点 -> 构建肢体向量 -> 计算向量余弦相似度
    '''
    min_len = min(len(gen_frames), len(ref_frames))
    scores = []
    for i in range(0, min_len, 2): # 步长为2，加速计算
        vec_gen = get_pose_vectors(gen_frames[i], model)
        vec_ref = get_pose_vectors(ref_frames[i], model)
        
        if vec_gen is not None and vec_ref is not None:
            # 计算余弦相似度 (Cosine Similarity)
            cos_sim = (vec_gen * vec_ref).sum(dim=1) # [Num_Limbs]
            
            # 只统计非零向量 (即双方都检测到的肢体)
            valid_limbs = (cos_sim != 0)
            if valid_limbs.any():
                frame_score = cos_sim[valid_limbs].mean().item()
                scores.append(frame_score)
    if not scores:
        return 0.0
    return sum(scores) / len(scores)

