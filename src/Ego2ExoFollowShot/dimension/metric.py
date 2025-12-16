import numpy as np

import torch
from torch import Tensor
import torch.nn.functional as F
from torchvision.transforms.functional import to_pil_image

from utils.math import p_corr
from utils.video_kit import get_traj

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