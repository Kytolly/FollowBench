import subprocess
import sys
import os
import logging
import re

import torch
from torch import Tensor
import torch.nn.functional as F
from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
import torchvision.transforms.functional as TF
from torchvision.ops import roi_align

from utils import *
import metric

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
    
def calculate_temporal_consistency(frames, flow_model=None, device=None):
    """
    基于 Tensor 计算时序一致性指标
    Returns:
        flickering (float): 闪烁度 (Warp Error)
        smoothness (float): 平滑度 (Flow Acceleration)
        dynamic_degree (float): 动态程度 (Avg Flow Magnitude)
    """
    if frames.dim() != 4:
        raise ValueError(f"Input frames must be [T, C, H, W], got {frames.shape}")
        
    if len(frames) < 3:
        return 0.0, 0.0, 0.0

    if device is None:
        device = frames.device

    # 1. 准备光流模型
    if flow_model is None:
        # 如果未提供模型，则临时加载 (会影响性能，建议外部传入)
        weights = Raft_Small_Weights.DEFAULT
        flow_model = raft_small(weights=weights, progress=False).to(device)
        flow_model.eval()
        
    # 2. 预处理
    # RAFT 训练时期望输入范围是 [-1, 1] 或 [0, 255] 归一化? 
    # Torchvision RAFT transforms 通常做 (img - 0.5) * 2 或直接输入 [0, 1] * 255
    # 这里我们将 [0, 1] -> [-1, 1] 以获得最佳效果
    frames_norm = (frames * 2.0) - 1.0
    
    T, C, H, W = frames.shape
    img1 = frames_norm[:-1] # [0, 1, ..., T-2]
    img2 = frames_norm[1:]  # [1, 2, ..., T-1]
    
    # 3. 批量计算光流
    # 为了防止显存爆炸，可以分 batch 处理，这里假设显存足够
    # RAFT 输出 list of flow predictions，取最后一个(refined)
    with torch.no_grad():
        list_of_flows = flow_model(img1, img2)
        flows = list_of_flows[-1] # [T-1, 2, H, W]

    # --- Dynamic Degree ---
    # 计算光流模长: sqrt(dx^2 + dy^2)
    flow_mags = torch.norm(flows, p=2, dim=1) # [T-1, H, W]
    dynamic_degree = flow_mags.mean().item()

    # --- Temporal Flickering (Warp Error) ---
    # 构造网格 grid: [T-1, H, W, 2]
    grid_y, grid_x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    grid = torch.stack((grid_x, grid_y), dim=0).float() # [2, H, W]
    grid = grid.unsqueeze(0).expand(T-1, -1, -1, -1) # [T-1, 2, H, W]
    
    # 加上光流 (Original logic: map_x = grid_x + flow_x)
    # 注意：grid_sample 需要归一化到 [-1, 1]
    # flow 单位是像素，加到 grid 上后得到采样坐标
    sampling_grid = grid + flows
    
    # 归一化采样坐标到 [-1, 1]
    # x_norm = 2 * (x / (W - 1)) - 1
    # y_norm = 2 * (y / (H - 1)) - 1
    sampling_grid[:, 0, :, :] = 2.0 * sampling_grid[:, 0, :, :] / (W - 1.0) - 1.0
    sampling_grid[:, 1, :, :] = 2.0 * sampling_grid[:, 1, :, :] / (H - 1.0) - 1.0
    
    # [T-1, 2, H, W] -> [T-1, H, W, 2] for grid_sample
    sampling_grid = sampling_grid.permute(0, 2, 3, 1)
    
    # Warp prev frame (img1) to curr frame (img2)
    # 注意：grid_sample 默认 input 是 [0, 1] 还是 [-1, 1] 取决于 frames 的原始值
    # 我们这里用原始 frames (0-1) 进行 warp，方便计算 error
    frames_orig1 = frames[:-1]
    frames_orig2 = frames[1:]
    
    warped_prev = F.grid_sample(frames_orig1, sampling_grid, mode='bilinear', padding_mode='border', align_corners=True)
    
    # 计算误差 L1 Loss
    diff = torch.abs(frames_orig2 - warped_prev)
    
    # 边缘 Mask (Original: border=5)
    border = 5
    if H > 2*border and W > 2*border:
        mask = torch.zeros_like(diff)
        mask[..., border:-border, border:-border] = 1.0
        flickering = (diff * mask).sum() / mask.sum()
    else:
        flickering = diff.mean()
    
    flickering = flickering.item()

    # --- Motion Smoothness ---
    # E_smooth = || F_t - F_{t-1} ||^2 (加速度)
    # flows: [0->1, 1->2, 2->3 ...]
    # flow_diff: flows[1:] - flows[:-1]
    if T > 2:
        flow_diffs = flows[1:] - flows[:-1] # [T-2, 2, H, W]
        # L2 norm over (dx, dy) -> mean over spatial and temporal
        smoothness = torch.norm(flow_diffs, p=2, dim=1).mean().item()
    else:
        smoothness = 0.0

    return flickering, smoothness, dynamic_degree

def AppearanceConsistency(dinov2_model, dino_transform, detection_model, ref_emb, video_tensor):
    '''计算外观一致性 (Re-ID Score)'''
    device = dinov2_model.device
    if ref_emb is None: return 0.0
    scores = []
    step = 5 # 采样间隔
    detection_model.eval()
    
    for i in range(0, len(video_tensor), step):
        frame_tensor = video_tensor[i] # [C, H, W]
        det_input = frame_tensor.unsqueeze(0) # [1, C, H, W]
        with torch.no_grad():
            prediction = detection_model(det_input)[0]
        
        # 筛选置信度最高的人
        best_box = None
        max_score = 0
        # 找到所有 label==1 (person) 且 score > 0.7 的索引
        valid_mask = (prediction['labels'] == 1) & (prediction['scores'] > 0.7)
        if valid_mask.any():
            # 找到最高分的索引
            valid_scores = prediction['scores'][valid_mask]
            valid_boxes = prediction['boxes'][valid_mask]
            
            best_idx = torch.argmax(valid_scores)
            best_box = valid_boxes[best_idx] # [x1, y1, x2, y2]
            
            # 构建 RoI 格式
            batch_index = torch.zeros((1, 1), device=device)
            roi_box = torch.cat([batch_index, best_box.view(1, 4)], dim=1) # [1, 5]
            
            # 在 GPU 上做 Crop + Resize
            person_crop = roi_align(
                input=det_input, # [1, C, H, W]
                boxes=roi_box,
                output_size=(224, 224),
                spatial_scale=1.0, # 因为我们在原图上操作
                aligned=True # 提高精度
            ) # [1, C, 224, 224]
            
            # Normalize + DINOv2 推理
            person_crop = dino_transform(person_crop)
            with torch.no_grad():
                frame_emb = dinov2_model(person_crop)
            
            # 计算相似度
            sim = torch.nn.functional.cosine_similarity(ref_emb, frame_emb).item()
            scores.append(sim)

    if not scores:
        return 0.0
        
    return sum(scores) / len(scores)

def CameraCenteringError(best_box, img_h, img_w):
    """计算 Camera Centering Error"""
    cx = (best_box[0] + best_box[2]) / 2.0; cy = (best_box[1] + best_box[3]) / 2.0 # Box 中心
    img_cx = img_w / 2.0; img_cy = img_h / 2.0 # 画面中心
    dist = torch.sqrt((cx - img_cx)**2 + (cy - img_cy)**2) # 计算欧氏距离
    max_dist = torch.sqrt(torch.tensor(img_cx**2 + img_cy**2, device=best_box.device)) # 归一化 (除以中心到角落的距离)
    error = dist / max_dist
    return error.item()

def calculate_subject_metrics(dinov2_model, dino_transform, detection_model, ref_emb, video_tensor):
    device = dinov2_model.device
    if ref_emb is None: return 0.0
    app_scores = []      # 外观一致性分数列表
    center_errors = []   # 中心误差列表
    detected_count = 0   # 检测到的帧数
    total_samples = 0    # 总采样帧数
    step = 5 # 采样间隔
    detection_model.eval()
    _, _, H, W = video_tensor.shape
    
    for i in range(0, len(video_tensor), step):
        total_samples += 1
        frame_tensor = video_tensor[i] # [C, H, W]
        det_input = frame_tensor.unsqueeze(0) # [1, C, H, W]
        with torch.no_grad():
            prediction = detection_model(det_input)[0]
        
        # 筛选置信度最高的人
        best_box = None
        max_score = 0
        # 找到所有 label==1 (person) 且 score > 0.7 的索引
        valid_mask = (prediction['labels'] == 1) & (prediction['scores'] > 0.7)
        if valid_mask.any():
            detected_count += 1
            # 找到最高分的索引
            valid_scores = prediction['scores'][valid_mask]
            valid_boxes = prediction['boxes'][valid_mask]
            
            best_idx = torch.argmax(valid_scores)
            best_box = valid_boxes[best_idx] # [x1, y1, x2, y2]
            
            # --- Camera Centering Error ---
            c_err = CameraCenteringError(best_box, H, W)
            center_errors.append(c_err)
            
            # --- Appearance Consistency ---
            # 构建 RoI 格式
            batch_index = torch.zeros((1, 1), device=device)
            roi_box = torch.cat([batch_index, best_box.view(1, 4)], dim=1) # [1, 5]
            
            # 在 GPU 上做 Crop + Resize
            person_crop = roi_align(
                input=det_input, # [1, C, H, W]
                boxes=roi_box,
                output_size=(224, 224),
                spatial_scale=1.0, # 因为我们在原图上操作
                aligned=True # 提高精度
            ) # [1, C, 224, 224]
            
            # Normalize + DINOv2 推理
            person_crop = dino_transform(person_crop)
            with torch.no_grad():
                frame_emb = dinov2_model(person_crop)
            
            # 计算相似度
            sim = torch.nn.functional.cosine_similarity(ref_emb, frame_emb).item()
            app_scores.append(sim)
        else: # 如果没检测到人：
            app_scores.append(0.0)
            center_errors.append(1.0)
            

    avg_app_score = sum(app_scores) / len(app_scores) if app_scores else 0.0
    avg_center_error = sum(center_errors) / len(center_errors) if center_errors else 1.0
    validity_score = detected_count / total_samples if total_samples > 0 else 0.0
    return avg_app_score, avg_center_error, validity_score

def BackgroundSemanticConsistency(
    clip_model, 
    clip_processor, 
    detection_model, 
    ref_image, 
    video_tensor):
    '''
    计算 Background Semantic Consistency
    逻辑：检测人 -> Mask掉人物区域 -> 计算剩余背景与参考图背景的CLIP特征相似度
    '''
    device = clip_model.device
    detection_model.eval()
    clip_model.eval()
    
    scores = []
    step = 5  # 采样间隔，为了提高计算速度
    ref_tensor = transforms.ToTensor()(ref_image).to(device).unsqueeze(0) # [1, C, H, W]
    
    with torch.no_grad():
        ref_pred = detection_model(ref_tensor)[0]
        ref_mask = torch.ones_like(ref_tensor)
        
        # 寻找参考图中的人并将其 Mask 掉 (置为黑)
        valid_mask = (ref_pred['labels'] == 1) & (ref_pred['scores'] > 0.7)
        if valid_mask.any():
            best_idx = torch.argmax(ref_pred['scores'][valid_mask])
            box = ref_pred['boxes'][valid_mask][best_idx].int()
            
            # 边界保护
            H, W = ref_tensor.shape[2], ref_tensor.shape[3]
            x1, y1, x2, y2 = max(0, box[0]), max(0, box[1]), min(W, box[2]), min(H, box[3])
            ref_mask[:, :, y1:y2, x1:x2] = 0.0
        
        masked_ref = ref_tensor * ref_mask
        
        # 转换回 CLIP Processor 接受的格式 (PIL) 并提取特征
        inputs_ref = clip_processor(images=transforms.ToPILImage()(masked_ref.squeeze().cpu()), return_tensors="pt").to(device)
        ref_bg_emb = clip_model.get_image_features(**inputs_ref)
        ref_bg_emb /= ref_bg_emb.norm(dim=-1, keepdim=True)

    # 遍历生成视频的帧
    for i in range(0, len(video_tensor), step):
        frame_tensor = video_tensor[i].unsqueeze(0) # [1, C, H, W]
        
        with torch.no_grad():
            prediction = detection_model(frame_tensor)[0]
            frame_mask = torch.ones_like(frame_tensor)
            valid_mask = (prediction['labels'] == 1) & (prediction['scores'] > 0.7)
            if valid_mask.any():
                best_idx = torch.argmax(prediction['scores'][valid_mask])
                box = prediction['boxes'][valid_mask][best_idx].int()
                H, W = frame_tensor.shape[2], frame_tensor.shape[3]
                x1, y1, x2, y2 = max(0, box[0]), max(0, box[1]), min(W, box[2]), min(H, box[3])
                frame_mask[:, :, y1:y2, x1:x2] = 0.0 # 将检测到的人物区域置为 0 (黑色)
            
            masked_frame = frame_tensor * frame_mask
            
            # CLIP 编码当前帧背景
            inputs_frame = clip_processor(images=transforms.ToPILImage()(masked_frame.squeeze().cpu()), return_tensors="pt").to(device)
            frame_bg_emb = clip_model.get_image_features(**inputs_frame)
            frame_bg_emb /= frame_bg_emb.norm(dim=-1, keepdim=True)
            
            # 计算 Cosine Similarity
            sim = torch.nn.functional.cosine_similarity(ref_bg_emb, frame_bg_emb).item()
            scores.append(sim)
    if not scores:
        return 0.0
    return sum(scores) / len(scores)

def OpticalFlowCorrelation(gen_frames, ref_frames):
    '''
    计算光流相关性 (Optical Flow Correlation)
    衡量生成视频的运动趋势(主要是相机运动)是否与参考视频同步。
    '''
    min_len = min(len(gen_frames), len(ref_frames))
    if min_len < 2:
        return 0.0
    gen_f = gen_frames[:min_len]
    ref_f = ref_frames[:min_len]

    # 分别提取生成视频和参考视频的运动序列
    gen_dx, gen_dy = get_motion_series(gen_f)
    ref_dx, ref_dy = get_motion_series(ref_f)
    
    # 计算皮尔逊相关系数 平滑化防止除 0 报错
    if np.std(gen_dx) < 1e-6 or np.std(ref_dx) < 1e-6:
        corr_x = 0.0
    else:
        corr_x, _ = pearsonr(gen_dx, ref_dx)
        
    if np.std(gen_dy) < 1e-6 or np.std(ref_dy) < 1e-6:
        corr_y = 0.0
    else:
        corr_y, _ = pearsonr(gen_dy, ref_dy)

    return (corr_x + corr_y) / 2.0

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

def TrajectoryAlignment(frames, device):
    pass

def calculate_all_flow_metrics(
    gen_frames: Tensor, 
    gt_frames=None, 
    flow_model=None, 
    device=None):
    """基于预加载的 RAFT 模型计算所有基于光流的指标 TF, MS, DD, OFC
    Args:
        gen_frames: 生成视频 Tensor [T, C, H, W] (0-1)
        gt_frames:  GT视频 Tensor [T, C, H, W] (0-1)。
        如果提供，则计算 OFC。
    """    
    # 预处理 (0~1 -> -1~1)
    if device is None: device = gen_frames.device
    gen_norm = (gen_frames * 2.0) - 1.0
    
    # 准备模型
    if flow_model == None: 
        flow_model=raft_small(
                    weights=Raft_Small_Weights.DEFAULT, 
                    progress=False)
        flow_model.to(device).eval()

    # 计算生成视频光流
    gen_flows = compute_flow(gen_norm, flow_model) # [T-1, 2, H, W]
    if gen_flows is None:
        return {'tf': 0.0, 'ms': 0.0, 'dd': 0.0, 'ofc': 0.0}
    
    # --- Dynamic Degree ---
    dd = metric.DynamicDegree(gen_flows)
    
    # --- Motion Smoothness ---
    ms = metric.MotionSmoothness(gen_flows)
        
    # --- Temporal Flickering ---
    tf = metric.TemporalFlickering(gen_frames, gen_flows, device)

    # --- Optical Flow Correlation ---
    ofc = None
    if gt_frames is not None and len(gt_frames) >= 2:
        gt_norm = (gt_frames * 2.0) - 1.0
        gt_flows = compute_flow(gt_norm, flow_model)
        ofc = metric.OpticalFlowCorrelation(gen_flows, gt_flows, device)

    return {'tf': tf, 'ms': ms, 'dd': dd, 'ofc': ofc}