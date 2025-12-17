import numpy as np
from scipy.linalg import sqrtm

import torch
from torch import Tensor
import torch.nn.functional as F
from torchvision.ops import roi_align
from torchvision.transforms.functional import to_pil_image

from utils.math import p_corr
from utils.video_kit import get_traj, compute_flow
from . import metric

def AestheticQuality(video_gen: Tensor, aq_model, batch_size=8):
    """
    计算美学质量 (LAION-Aesthetics) 
    Args:
        video_gen: [T, 3, H, W] tensor
        aq_model: pyiqa model
    """
    scores = []
    with torch.no_grad():
        for i in range(0, len(video_gen), batch_size):
            batch = video_gen[i : i + batch_size]
            res = aq_model(batch) # [B, 1]
            scores.append(res.view(-1)) # 展平并收集 Tensor
    if not scores:
        return 0.0
    all_scores = torch.cat(scores)
    return all_scores.mean().item()

def AppearanceConsistency(
    ref_emb,
    video_gen: Tensor,
    dinov2, 
    dino_transform, 
    detection_results,
    device):
    """
    计算外观一致性 (AC)
    Args:
        dinov2: Loaded DINOv2 model
        dino_transform: Preprocessing transform for DINOv2
        detection_results: List of (box, score) from get_detection_results
        ref_emb: Embedding of reference image [1, D]
        video_gen: Generated video tensor [T, 3, H, W] (0-1 float)
    """
    scores = []
    T = video_gen.shape[0]
    H, W = video_gen.shape[2], video_gen.shape[3]

    for i in range(T):
        res = detection_results[i]
        if res is None:
            scores.append(0.0) # 惩罚没有检测到人的结果
            continue
        
        box, _ = res
        x1, y1, x2, y2 = map(int, box.tolist())
        x1, y1 = max(0, x1), max(0, y1); x2, y2 = min(W, x2), min(H, y2) # 边界保护
        if x2 - x1 < 10 or y2 - y1 < 10:
            scores.append(0.0) # 框太小则忽略
            continue

        # Crop 人物区域提取特征
        person_crop = video_gen[i, :, y1:y2, x1:x2] # [3, h, w]
        img_pil = to_pil_image(person_crop.cpu()) # 转 PIL -> Transform -> Tensor -> GPU
        input_tensor = dino_transform(img_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            curr_emb = dinov2(input_tensor) # [1, D]

        sim = F.cosine_similarity(curr_emb, ref_emb) # 计算余弦相似度
        scores.append(sim.item())

    return sum(scores) / len(scores) if scores else 0.0

def BackgroundSemanticConsistency(
    ref_img_pil,
    video_gen,
    clip_model,
    clip_proc,
    detection_results,
    device):
    """
    计算背景语义一致性
    Args:
        clip_model: Loaded CLIP model
        clip_proc: CLIP Processor
        detection_results: List of (box, score)
        ref_img_pil: Reference image (PIL)
        video_gen: Generated video tensor [T, 3, H, W]
    """
    scores = []
    T = video_gen.shape[0]
    H, W = video_gen.shape[2], video_gen.shape[3]
    
    # 预计算参考图特征
    inputs_ref = clip_proc(images=ref_img_pil, return_tensors="pt").to(device)
    with torch.no_grad():
        ref_emb = clip_model.get_image_features(**inputs_ref)
    
    for i in range(T):
        frame_tensor = video_gen[i].clone() # [3, H, W]
        
        # Mask 掉人物
        res = detection_results[i]
        if res is not None:
            box, _ = res
            x1, y1, x2, y2 = map(int, box.tolist())
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)
            frame_tensor[:, y1:y2, x1:x2] = 0.0 # Black out
            
        # 提取特征
        img_pil = to_pil_image(frame_tensor.cpu())
        inputs = clip_proc(images=img_pil, return_tensors="pt").to(device)
        
        with torch.no_grad():
            curr_emb = clip_model.get_image_features(**inputs)
            
        # 计算相似度
        sim = F.cosine_similarity(curr_emb, ref_emb)
        scores.append(sim.item())
        
    return sum(scores) / len(scores) if scores else 0.0

def CameraCenteringError(detection_results, H, W):
    """计算相机中心误差 衡量生成的主角是否位于画面中心 符合第三人称跟随视角的构图习惯"""
    errors = []
    center_frame = torch.tensor([W / 2.0, H / 2.0]) 
    max_dist = torch.sqrt((center_frame[0])**2 + (center_frame[1])**2)
    
    for res in detection_results:
        if res is None:
            errors.append(1.0)
            continue
        
        box, _ = res
        center_box = torch.tensor([(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0])
        dist = torch.dist(center_box, center_frame)
        errors.append((dist / max_dist).item())
        
    return sum(errors) / len(errors) if errors else 1.0

def DynamicDegree(gen_flows: Tensor):
    '''计算视频的动态程度 所有像素移动距离的平均值 数值越大 视频里的东西动得越剧烈'''
    # gen_flows: [T-1, 2, H, W]
    flow_mags = torch.norm(gen_flows, p=2, dim=1) # [T-1, H, W]
    dd = flow_mags.mean().item()
    return dd

def FrechetVideoDistance(feat_gen: np.ndarray, feat_gt: np.ndarray):
    """
    计算 FVD (Fréchet Video Distance)
    使用 Scipy 进行矩阵运算以保证数值稳定性
    """
    if feat_gen.shape[0] == 0 or feat_gt.shape[0] == 0:
        return 0.0
    mu1, sigma1 = np.mean(feat_gen, axis=0), np.cov(feat_gen, rowvar=False)
    mu2, sigma2 = np.mean(feat_gt, axis=0), np.cov(feat_gt, rowvar=False)
    if feat_gen.shape[0] == 1: sigma1 = 0.0
    if feat_gt.shape[0] == 1: sigma2 = 0.0
    
    diff = mu1 - mu2
    if np.isscalar(sigma1) and np.isscalar(sigma2):
        covmean = np.sqrt(sigma1 * sigma2)
        trace_term = sigma1 + sigma2 - 2.0 * covmean
        fvd = diff.dot(diff) + trace_term
    else:
        covmean = sqrtm(sigma1.dot(sigma2))
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        fvd = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2.0 * covmean)
        
    return float(fvd)

def HumanActionAlignment(gen_results, gt_results, H, W):
    """
    计算人体动作对齐度 (HAA)
    使用归一化后的平均关键点位置误差 (MPJPE)。
    
    Args:
        gen_results: List of (keypoints, scores)
        gt_results: List of (keypoints, scores)
        H (int): 图像高度
        W (int): 图像宽度
    """
    min_len = min(len(gen_results), len(gt_results))
    if min_len < 1: return 1.0

    frame_errors = []
    img_diag = np.sqrt(H**2 + W**2) + 1e-6
    
    for i in range(min_len):
        kp_gen, _ = gen_results[i] if gen_results[i] is not None else (None, None)
        kp_gt, _ = gt_results[i] if gt_results[i] is not None else (None, None)
        
        if kp_gen is None or kp_gt is None:
            frame_errors.append(1.0) # 惩罚
            continue
            
        if kp_gen.shape != kp_gt.shape:
            frame_errors.append(1.0)
            continue
            
        pos_gen = kp_gen[:, :2] 
        pos_gt = kp_gt[:, :2] 

        # 计算欧氏距离
        joint_dists = torch.norm(pos_gen - pos_gt, dim=1) 
        
        # 仅考虑有效关节 (Confidence > 0.5)
        conf_gen = kp_gen[:, 2] > 0.5
        conf_gt = kp_gt[:, 2] > 0.5
        valid_joints = conf_gen & conf_gt

        if valid_joints.sum() == 0:
            frame_errors.append(1.0)
            continue

        # 计算该帧平均误差
        mean_error_px = joint_dists[valid_joints].mean().item()
        
        # 归一化误差
        normalized_error = min(mean_error_px / img_diag, 1.0)
        frame_errors.append(normalized_error)
        
    return np.mean(frame_errors) if frame_errors else 1.0

def ImagingQuality(video_gen: Tensor, iq_model, batch_size=4):
    """
    计算图像质量 (MUSIQ) 
    Args:
        video_gen: [T, 3, H, W] tensor
        iq_model: pyiqa model
    """
    scores = []
    with torch.no_grad():
        for i in range(0, len(video_gen), batch_size):
            batch = video_gen[i : i + batch_size]
            res = iq_model(batch)
            scores.append(res.view(-1))
    if not scores:
        return 0.0
    all_scores = torch.cat(scores)
    return all_scores.mean().item()

def MotionSmoothness(gen_flows: Tensor):
    '''计算运动平滑性 光流场在时间上的变化率'''
    # gen_flows: [T-1, 2, H, W]
    if len(gen_flows) > 1:
        flow_acc = gen_flows[1:] - gen_flows[:-1]
        ms = torch.norm(flow_acc, p=2, dim=1).mean().item()
    else:
        ms = 0.0
    return ms

def OpticalFlowCorrelation(gen_flows: Tensor, gt_flows: Tensor, device):
    '''视频的抖动程度 越低视频约接近电影级运镜 在某个范围内认为是接近 ground truth 需要 tradeoff'''
    min_len = min(len(gen_flows), len(gt_flows))
    g_flow = gen_flows[:min_len]
    t_flow = gt_flows[:min_len]

    g_motion = g_flow.mean(dim=[2, 3]) # [t, 2]
    t_motion = t_flow.mean(dim=[2, 3]) # [t, 2]
    
    corr_x = p_corr(g_motion[:, 0], t_motion[:, 0], device)
    corr_y = p_corr(g_motion[:, 1], t_motion[:, 1], device)
    ofc = ((corr_x + corr_y) / 2.0).item()
    return ofc

def TemporalFlickering(gen_frames: Tensor, gen_flows: Tensor, device):
    '''计算时间闪烁 从 t-1 到 t 的光流模长'''
    T, C, H, W = gen_frames.shape
    grid_y, grid_x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    grid = torch.stack((grid_x, grid_y), dim=0).float().unsqueeze(0).expand(T-1, -1, -1, -1) # [T-1, 2, H, W]

    sampling_grid = grid + gen_flows
    sampling_grid[:, 0] = 2.0 * sampling_grid[:, 0] / (W - 1.0) - 1.0
    sampling_grid[:, 1] = 2.0 * sampling_grid[:, 1] / (H - 1.0) - 1.0
    sampling_grid = sampling_grid.permute(0, 2, 3, 1) # [T-1, H, W, 2]

    frames_orig1 = gen_frames[:-1]
    frames_orig2 = gen_frames[1:]
    warped_prev = F.grid_sample(frames_orig1, sampling_grid, mode='bilinear', padding_mode='border', align_corners=True)

    diff = torch.abs(frames_orig2 - warped_prev)
    border = 5
    if H > 2 * border and W > 2 * border:
        mask = torch.zeros_like(diff)
        mask[..., border:-border, border:-border] = 1.0
        tf = (diff * mask).sum() / mask.sum()
    else:
        tf = diff.mean()
    tf = tf.item()
    return tf

def TrajectoryAlignment(gen_results, gt_results, H, W):
    '''计算生成视频和ground truth的人物在画面中移动轨迹的对齐程度'''
    traj_gen = get_traj(gen_results)
    traj_gt = get_traj(gt_results)
    
    min_len = min(len(traj_gen), len(traj_gt))
    if min_len < 2: return 1.0 # 无法计算
    
    dists = []
    diag = np.sqrt(H**2 + W**2)
    for i in range(min_len):
        p_gen = traj_gen[i]
        p_gt = traj_gt[i]
        if p_gen is not None and p_gt is not None:
            d = np.linalg.norm(p_gen - p_gt) / diag
            dists.append(d)
            
    return np.mean(dists) if dists else 1.0

def ViewpointValidity(detection_results):
    """计算视角的人物检测率 衡量生成模型是否崩坏，是否生成了无法被识别为人的扭曲图像"""
    if not detection_results: return 0.0
    detected = sum(1 for res in detection_results if res is not None)
    return detected / len(detection_results)

def calculate_metrics_based_flow_model(
    gen_frames: Tensor,
    gt_frames=None,
    metrics_to_compute=None,
    flow_model=None,
    device=None):
    """基于预加载的 RAFT 模型计算所有基于光流的指标 TF, MS, DD, OFC
    Args:
        gen_frames: 生成视频 Tensor [T, C, H, W] (0-1)
        gt_frames:  GT视频 Tensor [T, C, H, W] (0-1)
        如果提供，则计算 OFC。
    """
    if device is None: device = gen_frames.device
    assert flow_model is not None 

    # 计算生成视频光流
    gen_norm = (gen_frames * 2.0) - 1.0
    gen_flows = compute_flow(gen_norm, flow_model) # [T-1, 2, H, W]
    if gen_flows is None:
        return {'tf': 0.0, 'ms': 0.0, 'dd': 0.0, 'ofc': 0.0}
    
    results = {}
    # --- Dynamic Degree (DD) ---
    if 'dd' in metrics_to_compute:
        results['dd'] = metric.DynamicDegree(gen_flows)
    
    # --- Motion Smoothness (MS) ---
    if 'ms' in metrics_to_compute:
        results['ms'] = metric.MotionSmoothness(gen_flows)
        
    # --- Temporal Flickering (TF) ---
    if 'tf' in metrics_to_compute:
        results['tf'] = metric.TemporalFlickering(gen_frames, gen_flows, device)

    # --- Optical Flow Correlation --- 
    if 'ofc' in metrics_to_compute and gt_frames is not None:
        if len(gt_frames) >= 2:
            gt_norm = (gt_frames * 2.0) - 1.0
            gt_flows = compute_flow(gt_norm, flow_model)
            results['ofc'] = metric.OpticalFlowCorrelation(gen_flows, gt_flows, device)
        else:
            results['ofc'] = 0.0

    return results