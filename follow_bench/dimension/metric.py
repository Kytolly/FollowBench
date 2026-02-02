import numpy as np
from scipy.linalg import sqrtm
import logging
logger = logging.getLogger(__name__)

import torch
from torch import Tensor
import torch.nn.functional as F
from torchvision.transforms.functional import to_pil_image
from typing import Any, Optional, Set

from ..utils.math import(
    axis_angle_to_matrix,
    wrap_to_pi
)
from ..utils.video_kit import get_traj, compute_flow
from . import metric


def AestheticQuality(video_gen: Tensor, aq_model: Any, batch_size: int = 8):  # noqa: ANN201
    """Compute aesthetics score (LAION-Aesthetics) for a video.

    The function applies `aq_model` to video frames in batches and returns the
    mean per-frame aesthetics score.

    Args:
        video_gen: Tensor of shape [T, 3, H, W], frames normalized as model expects.
        aq_model: callable model that accepts a batch tensor and returns a score
            tensor of shape [B, 1] or [B].
        batch_size: batch size to use when evaluating the model.

    Returns:
        Mean aesthetic score as a float (0.0 if no frames).
    """
    assert aq_model is not None
    scores = []
    with torch.no_grad():
        for i in range(0, len(video_gen), batch_size):
            batch = video_gen[i:i + batch_size]
            res = aq_model(batch)  # [B, 1]
            scores.append(res.view(-1))  # 展平并收集 Tensor
    if not scores:
        return 0.0
    all_scores = torch.cat(scores)
    return all_scores.mean().item()


def AppearanceConsistency(embeddings_gen: torch.Tensor, embedding_ref: torch.Tensor):
    """
    计算外观一致性分数 (Cosine Similarity)。
    
    Args:
        embeddings_gen: [T, D] 生成视频每一帧的人物特征向量
        embedding_ref:  [1, D] 参考图的人物特征向量
        
    Returns:
        float: 平均余弦相似度
    """
    # 1. 归一化特征向量 (L2 Norm)
    embeddings_gen = F.normalize(embeddings_gen, p=2, dim=-1)
    embedding_ref = F.normalize(embedding_ref, p=2, dim=-1)
    
    # 2. 计算余弦相似度 (Dot Product)
    # [T, D] * [D, 1] -> [T, 1]
    sim_scores = torch.mm(embeddings_gen, embedding_ref.transpose(0, 1))
    
    # 3. 计算均值
    return float(sim_scores.mean().item())


def AverageDisplacementError(gen_results, gt_results, H, W):  # noqa: ANN201
    """Measure alignment between generated and ground-truth trajectories.

    Computes normalized per-frame distances between trajectories (center points
    per frame) and returns the mean normalized distance.

    Args:
        gen_results: Detection/keypoint results for generated video frames.
        gt_results: Detection/keypoint results for ground-truth frames.
        H: Image height.
        W: Image width.

    Returns:
        Mean normalized trajectory alignment error in [0.0, 1.0] (1.0 worst).
    """
    traj_gen = get_traj(gen_results)
    traj_gt = get_traj(gt_results)
    
    min_len = min(len(traj_gen), len(traj_gt))
    if min_len < 2:
        return 1.0  # 无法计算
    
    dists = []
    diag = np.sqrt(H**2 + W**2)
    for i in range(min_len):
        p_gen = traj_gen[i]
        p_gt = traj_gt[i]
        if p_gen is not None and p_gt is not None:
            d = np.linalg.norm(p_gen - p_gt) / diag
            dists.append(d)
            
    return np.mean(dists) if dists else 1.0


def BackgroundSemanticContinuity(
    video_gen: Tensor,
    clip_model: Any,
    detection_results: Any,
    device: Any,
    batch_size: int = 64
):
    """
    计算背景语义连续性 (Background Semantic Continuity)。
    
    衡量视频每一帧的背景与下一帧背景的语义相似度，反映背景的稳定性。
    该指标不需要参考图。
    
    Args:
        video_gen: [T, 3, H, W] 生成视频张量
        clip_model: CLIP 模型
        detection_results: 检测结果列表 (用于 Mask 掉人物)
        device: 计算设备
        batch_size: 推理时的 Batch Size
        
    Returns:
        float: 相邻帧背景特征的平均余弦相似度 (0.0 ~ 1.0)
    """
    T, C, H, W = video_gen.shape
    if T < 2:
        return 0.0 # 视频太短无法计算连续性
    
    # ================= 阶段 1: GPU Masking =================
    # 在 GPU 上直接将人物区域涂黑，只保留背景
    # 使用 clone 防止修改原始视频数据
    masked_video = video_gen.clone() 
    
    for i in range(T):
        res = detection_results[i]
        if res is not None:
            # 解析 Bbox: 假设格式为 [x1, y1, x2, y2]
            # 兼容 list, tuple 或 tensor
            box = res[0] if isinstance(res, (list, tuple)) else res
            
            x1 = int(max(0, box[0].item()))
            y1 = int(max(0, box[1].item()))
            x2 = int(min(W, box[2].item()))
            y2 = int(min(H, box[3].item()))
            
            # 将人物区域填黑 (0.0)
            masked_video[i, :, y1:y2, x1:x2] = 0.0

    # ================= 阶段 2: GPU Resize & Normalize =================
    # 一次性处理所有帧 (Resize -> 224x224)
    # CLIP 必须使用 bicubic 插值以保证特征准确性
    frames_resized = F.interpolate(
        masked_video, 
        size=(224, 224), 
        mode='bicubic', 
        align_corners=False,
        antialias=True
    )
    
    # CLIP 标准化 (OpenAI Mean/Std)
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)
    
    frames_input = (frames_resized - mean) / std

    # ================= 阶段 3: Batch Inference =================
    # 提取所有帧的 Embedding
    all_embeddings = []
    
    with torch.no_grad():
        for i in range(0, T, batch_size):
            batch = frames_input[i : i + batch_size]
            
            # 提取特征
            batch_emb = clip_model.get_image_features(pixel_values=batch)
            # 归一化 (计算 Cosine Similarity 前必须做)
            batch_emb = F.normalize(batch_emb, p=2, dim=-1)
            
            all_embeddings.append(batch_emb)

    # 拼接所有帧特征: [T, D]
    if not all_embeddings:
        return 0.0
    embeddings = torch.cat(all_embeddings, dim=0)

    # ================= 阶段 4: 计算相邻帧相似度 =================
    # Frame[0] vs Frame[1]
    # Frame[1] vs Frame[2]
    # ...
    # Frame[T-1] vs Frame[T]
    
    emb_t = embeddings[:-1]   # [0, 1, ..., T-2]
    emb_t_plus_1 = embeddings[1:] # [1, 2, ..., T-1]
    
    # 计算点积 (因为已经归一化了，点积等于余弦相似度)
    # dim=-1 表示在特征维度求和
    similarities = (emb_t * emb_t_plus_1).sum(dim=-1)
    
    return float(similarities.mean().item())


def CameraCenteringError(detection_results: Any, H: int, W: int):  # noqa: ANN201
    """Compute camera centering error measuring how close the main person is to frame center.

    The metric normalizes the distance from detection box center to the image
    center by the image diagonal. Missing detections are penalized (1.0).

    Args:
        detection_results: List of detection results per frame or None.
        H: Image height.
        W: Image width.

    Returns:
        Mean normalized centering error in [0.0, 1.0] (1.0 worst).
    """
    if not detection_results:
        return 1.0

    # 确定计算使用的设备 (优先 GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 定义画面中心点
    center_frame = torch.tensor([W / 2.0, H / 2.0], device=device)
    # 使用半对角线长度作为归一化基准 (最大可能距离)
    max_dist = torch.sqrt(center_frame[0]**2 + center_frame[1]**2)
    
    errors = []
    for res in detection_results:
        # 2. 惩罚机制：如果该帧未检测到人，误差计为最大值 1.0
        if res is None:
            errors.append(torch.tensor(1.0, device=device))
            continue
        
        # 提取 Bbox Tensor (假设格式为 [x1, y1, x2, y2])
        box = res[0] if isinstance(res, (list, tuple)) else res
        box = box.to(device)
        
        # 3. 计算 Box 中心点
        center_box = torch.tensor([
            (box[0] + box[2]) / 2.0, 
            (box[1] + box[3]) / 2.0
        ], device=device)
        
        # 4. 计算欧式距离并归一化
        dist = torch.dist(center_box, center_frame)
        errors.append(torch.clamp(dist / max_dist, max=1.0))
    
    if not errors:
        return 1.0
        
    # 5. 返回全视频平均误差
    return float(torch.stack(errors).mean().item())


def CameraSubjectHeadingAlignment(
    cam_pos: torch.Tensor,
    subj_pos: torch.Tensor,
    subj_orient_aa: torch.Tensor,
    forward_axis: str = '+z'
):
    """
    CSHA 核心计算逻辑。
    
    Args:
        cam_pos: [T, 3] 相机世界坐标
        subj_pos: [T, 3] 主角 Root 世界坐标
        subj_orient_aa: [T, 3] 主角 Root 旋转 (Axis-Angle)
        forward_axis: SMPL 默认 T-pose 前向通常是 +z
        
    Returns:
        dict: {
            "stability_error": float, (圆周标准差)
            "locking_error": float, (与正后方的偏差)
            "azimuth_seq": Tensor (角度序列，用于可视化)
        }
    """
    T = cam_pos.shape[0]
    
    # 1. 计算主角朝向向量 (Subject Forward Vector)
    # R_subj: [T, 3, 3]
    R_subj = axis_angle_to_matrix(subj_orient_aa)
    
    # 定义局部坐标系下的前向向量 [0, 0, 1]
    local_forward = torch.tensor([0.0, 0.0, 1.0], device=cam_pos.device).view(1, 3, 1)
    if forward_axis == '-z':
        local_forward = -local_forward
        
    # 变换到世界坐标系: v_subj = R * v_local
    # [T, 3, 3] @ [T, 3, 1] -> [T, 3, 1] -> [T, 3]
    v_subj_3d = torch.matmul(R_subj, local_forward.repeat(T, 1, 1)).squeeze(-1)
    
    # 投影到 XZ 平面 (Ground Plane)
    v_subj_2d = v_subj_3d[:, [0, 2]] # [x, z]
    
    # 2. 计算相机方位向量 (Camera Position Vector relative to Subject)
    # D = P_cam - P_subj
    D_3d = cam_pos - subj_pos
    v_cam_2d = D_3d[:, [0, 2]] # [x, z]
    
    # 3. 计算角度 (Atan2)
    # atan2(z, x) 注意参数顺序 (y, x) -> (z, x)
    theta_subj = torch.atan2(v_subj_2d[:, 1], v_subj_2d[:, 0])
    theta_cam = torch.atan2(v_cam_2d[:, 1], v_cam_2d[:, 0])
    
    # 4. 相对方位角 (Relative Azimuth)
    # phi = wrap(theta_cam - theta_subj)
    phi = wrap_to_pi(theta_cam - theta_subj)
    
    # 5. 计算指标
    
    # A. 稳定性误差 (Circular Standard Deviation)
    # R_bar = (1/T) * sum( exp(i * phi) )
    cos_sum = torch.sum(torch.cos(phi))
    sin_sum = torch.sum(torch.sin(phi))
    R_bar_len = torch.sqrt(cos_sum**2 + sin_sum**2) / T
    
    # 防止 log(0)
    R_bar_len = torch.clamp(R_bar_len, min=1e-6, max=1.0 - 1e-6)
    csha_stability = torch.sqrt(-2 * torch.log(R_bar_len))
    
    # B. 锁定误差 (Locking Error to pi/180 degrees)
    # 理想情况下 phi 应该接近 pi 或 -pi
    csha_lock = torch.mean(torch.abs(wrap_to_pi(phi - torch.pi)))
    
    return {
        "stability_score": csha_stability.item(),
        "locking_score": csha_lock.item(),
        "azimuth_series": phi
    }


def CameraStability(
    video_gen: Tensor,
    target_size: int = 512
) -> float:
    """
    计算相机稳定性。
    使用 GPU 加速的相位相关法 (Phase Correlation) 估算帧间全局运动，
    并计算运动加速度的平滑度。这个指标不是在看“相机走得对不对”，而是在看相机拿得稳不稳

    Args:
        video_gen: [T, 3, H, W] Normalized video tensor (0.0 ~ 1.0)
        target_size: 计算 FFT 时的缩放尺寸，默认 512 以平衡速度和精度

    Returns:
        float: 抖动分数 (Jitter Score). 0.0 表示完美平滑/匀速。
    """
    # 1. 检查输入并确保在 GPU
    if video_gen is None:
        return 0.0
    
    device = video_gen.device
    if not video_gen.is_cuda and torch.cuda.is_available():
        # 如果还在 CPU，尝试移动到 GPU (但通常 Loader 已经做好了)
        video_gen = video_gen.to('cuda')
        device = video_gen.device

    T, C, H, W = video_gen.shape
    if T < 2:
        return 0.0

    # 2. 预处理: RGB -> Grayscale -> Resize
    # 权重: RGB -> Grayscale (Standard Rec. 601)
    weights = torch.tensor([0.299, 0.587, 0.114], device=device).view(1, 3, 1, 1)
    gray = F.conv2d(video_gen, weights) # [T, 1, H, W]

    # Resize 到固定大小 (例如 512x512) 以保证 FFT 效率
    if H > target_size or W > target_size:
        gray_resized = F.interpolate(
            gray, 
            size=(target_size, target_size), 
            mode='bilinear', 
            align_corners=False
        ).squeeze(1) # [T, 512, 512]
    else:
        gray_resized = gray.squeeze(1)
        target_size = H # 简化处理，假设方形或接近

    # 3. 加窗 (Hanning Window) 防止 FFT 边缘泄漏
    hann = torch.hann_window(target_size, device=device)
    window = hann.view(-1, 1) * hann.view(1, -1) # [Sz, Sz]
    frames_windowed = gray_resized * window.unsqueeze(0)

    # 4. 批量 FFT 计算 (PyTorch FFT 极快)
    ffts = torch.fft.rfft2(frames_windowed)

    # 5. 相位相关 (Phase Correlation)
    # 计算相邻帧之间的互功率谱
    f1 = ffts[:-1] # Frame t
    f2 = ffts[1:]  # Frame t+1
    
    cross_prod = f1 * torch.conj(f2)
    eps = 1e-8
    cross_power = cross_prod / (torch.abs(cross_prod) + eps)
    
    # 逆变换得到响应图 (Impulse Response)
    response = torch.fft.irfft2(cross_power) # [T-1, Sz, Sz]

    # 6. 寻找峰值位置 (Peak Finding -> Shift Estimation)
    B_res, H_res, W_res = response.shape
    response_flat = response.view(B_res, -1)
    argmax = torch.argmax(response_flat, dim=1)
    
    dy = argmax // W_res
    dx = argmax % W_res
    
    # 处理循环位移 (Negative shifts appear at the end)
    dy = torch.where(dy > H_res // 2, dy - H_res, dy)
    dx = torch.where(dx > W_res // 2, dx - W_res, dx)
    
    velocities = torch.stack([dx, dy], dim=1).float() # [T-1, 2]

    # 7. 计算抖动 (Jitter = Magnitude of Acceleration)
    if velocities.shape[0] < 2:
        return 0.0
        
    # 加速度 = 速度差分
    acceleration = velocities[1:] - velocities[:-1]
    
    # 计算加速度的 L2 范数并求平均
    jitter_magnitude = torch.norm(acceleration, dim=1).mean()
    
    # 归一化: 经验值 2.0 像素/帧^2 的抖动已经很大了
    score = torch.tanh(jitter_magnitude / 2.0).item()
    
    return float(score)


def DynamicDegree(gen_flows: Tensor):  # noqa: ANN201
    """Compute Dynamic Degree (DD) as the mean optical-flow magnitude.

    Args:
        gen_flows: Optical flows of shape [T-1, 2, H, W].

    Returns:
        Mean flow magnitude (float) representing overall motion intensity.
    """
    # gen_flows: [T-1, 2, H, W]
    flow_mags = torch.norm(gen_flows, p=2, dim=1)  # [T-1, H, W]
    dd = flow_mags.mean().item()
    return dd


def FrechetVideoDistance(feat_gen: np.ndarray, feat_gt: np.ndarray):  # noqa: ANN201
    """
    计算 FVD (Fréchet Video Distance)
    使用 Scipy 进行矩阵运算以保证数值稳定性
    """
    if feat_gen.shape[0] < 2 or feat_gt.shape[0] < 2:
        return 0.0
        
    mu1, sigma1 = np.mean(feat_gen, axis=0), np.cov(feat_gen, rowvar=False)
    mu2, sigma2 = np.mean(feat_gt, axis=0), np.cov(feat_gt, rowvar=False)
    
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


def HumanActionAlignment(gen_results: Any, gt_results: Any, H: int, W: int):  # noqa: ANN201
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
    if min_len < 1:
        return 1.0

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


def ImagingQuality(video_gen: Tensor, iq_model: Any, batch_size: int = 4):  # noqa: ANN201
    """
    计算图像质量 (MUSIQ)
    Args:
        video_gen: [T, 3, H, W] tensor
        iq_model: pyiqa model
    """
    scores = []
    with torch.no_grad():
        for i in range(0, len(video_gen), batch_size):
            batch = video_gen[i:i + batch_size]
            res = iq_model(batch)
            scores.append(res.view(-1))
    if not scores:
        return 0.0
    all_scores = torch.cat(scores)
    return all_scores.mean().item()


def MotionSmoothness(gen_flows: Tensor):  # noqa: ANN201
    """Compute motion smoothness as mean temporal acceleration magnitude of flow.

    The metric measures how much the optical flow changes over time; lower
    values indicate smoother motion.

    Args:
        gen_flows: Optical flows of shape [T-1, 2, H, W].

    Returns:
        Mean temporal change magnitude (float).
    """
    # gen_flows: [T-1, 2, H, W]
    if len(gen_flows) > 1:
        flow_acc = gen_flows[1:] - gen_flows[:-1]
        ms = torch.norm(flow_acc, p=2, dim=1).mean().item()
    else:
        ms = 0.0
    return ms


def OpticalFlowCorrelation(motion_gen: Tensor, motion_gt: Tensor):  # noqa: ANN201
    """Compute correlation between generated and ground-truth optical flow motions.

    The function averages per-frame displacement vectors and computes Pearson
    correlation on x and y channels, returning their mean.

    Args:
        gen_flows: Generated optical flows [T-1, 2, H, W].
        gt_flows: Ground-truth optical flows [T-1, 2, H, W].
        device: Torch device used by p_corr helper.

    Returns:
        Mean correlation value (float) between -1 and 1.
    """
    # 长度对齐
    min_len = min(motion_gen.shape[0], motion_gt.shape[0])
    if min_len < 2:
        return 0.0
    
    m_gen = motion_gen[:min_len].flatten() # [ (T-1)*2 ]
    m_gt = motion_gt[:min_len].flatten()   # [ (T-1)*2 ]
    
    # 计算余弦相似度
    score = F.cosine_similarity(m_gen.unsqueeze(0), m_gt.unsqueeze(0)).item()
    return score


def SideBySideDepthConsistency(
    depth_pred: torch.Tensor, 
    depth_gt: torch.Tensor, 
    eps: float = 1e-6
):
    """
    Core algorithm for SSDC: Affine-Invariant Depth Error.
    
    Args:
        depth_pred: Predicted depth map sequence [T, H, W] or flattened [N]
        depth_gt: Ground truth depth map sequence [T, H, W] or flattened [N]
        eps: Epsilon for numerical stability
        
    Returns:
        RMSE score after least-squares alignment (Lower is better)
    """
    # 1. Flatten
    d_pred = depth_pred.flatten().float()
    d_gt = depth_gt.flatten().float()
    
    if d_pred.numel() != d_gt.numel():
        # Handle mismatch by cropping to min length if necessary, 
        # though usually handled in Evaluator
        min_len = min(d_pred.numel(), d_gt.numel())
        d_pred = d_pred[:min_len]
        d_gt = d_gt[:min_len]

    # 2. Statistics
    mu_pred = torch.mean(d_pred)
    mu_gt = torch.mean(d_gt)

    d_pred_centered = d_pred - mu_pred
    d_gt_centered = d_gt - mu_gt

    # 3. Least Squares Alignment (Closed-form)
    # s = Cov(pred, gt) / Var(pred)
    numerator = torch.sum(d_pred_centered * d_gt_centered)
    denominator = torch.sum(d_pred_centered ** 2)

    if denominator < eps:
        s = torch.tensor(0.0, device=d_pred.device)
    else:
        s = numerator / denominator

    # t = mean_gt - s * mean_pred
    t = mu_gt - s * mu_pred

    # 4. Rectification & Error
    d_pred_aligned = s * d_pred + t
    
    mse = torch.mean((d_pred_aligned - d_gt) ** 2)
    rmse = torch.sqrt(mse)

    return rmse.item()


def SourceControlConditionRecall(
    gen_feats: torch.Tensor, 
    gt_feats: torch.Tensor, 
    top_k: tuple = (1, 5)
):
    """
    SCCR 核心计算：基于检索的后验验证。
    
    Args:
        gen_feats: [N, D] 生成视频特征矩阵 (Query)
        gt_feats:  [N, D] 真值视频特征矩阵 (Gallery/Distractors)
    """
    N = gen_feats.shape[0]
    if N == 0:
        return {f'SCCR@{k}': 0.0 for k in top_k}
    
    # 1. 归一化 (L2 Norm)
    gen_norm = torch.nn.functional.normalize(gen_feats, p=2, dim=-1)
    gt_norm = torch.nn.functional.normalize(gt_feats, p=2, dim=-1)
    
    # 2. 计算相似度矩阵 S [N, N]
    # S[i, j] = Gen[i] 与 GT[j] 的相似度
    sim_matrix = torch.mm(gen_norm, gt_norm.transpose(0, 1))
    
    # 3. 排序 (Ranking)
    # 对每一行降序排列，看 GT 对应的索引排在哪里
    # 理想情况：S[i, i] 应该是该行的最大值
    _, sorted_indices = torch.sort(sim_matrix, dim=1, descending=True)
    
    # 4. 找到 Ground Truth 所在的 Rank
    # 每一行的目标索引就是行号本身 (0, 1, ..., N-1)
    targets = torch.arange(N, device=sim_matrix.device).view(N, 1)
    
    # rank (0-based)
    rank_matrix = (sorted_indices == targets).nonzero(as_tuple=True)[1]
    
    # 转为 1-based rank 以便计算 @K
    ranks = rank_matrix + 1
    
    # 5. 计算 Recall@K
    results = {}
    for k in top_k:
        acc = (ranks <= k).float().mean().item()
        results[f'SCCR@{k}'] = acc
        
    return results


def SubjectCameraDistanceError(
    depth_gen: torch.Tensor, 
    depth_gt: torch.Tensor,
    eps: float = 1e-6
):
    """
    计算 SCDE (Subject-Camera Distance Error) 的数学核心。
    
    对应公式:
        1. 寻找最佳缩放 s: s_hat = sum(z_gen * z_gt) / sum(z_gen^2)
        2. 校正: z_gen_hat = s_hat * z_gen
        3. 误差: RMSE(z_gen_hat - z_gt)
    
    Args:
        depth_gen: 生成视频的深度/距离序列 [T] (z_gen)
        depth_gt:  真值视频的深度/距离序列 [T] (z_gt)
    
    Returns:
        float: Scale-Aligned RMSE
    """
    # 1. 确保输入为一维向量
    z_gen = depth_gen.flatten().float()
    z_gt = depth_gt.flatten().float()
    
    if z_gen.numel() != z_gt.numel():
        min_len = min(z_gen.numel(), z_gt.numel())
        z_gen = z_gen[:min_len]
        z_gt = z_gt[:min_len]

    # 2. 最小二乘法计算尺度因子 s (Scale Alignment)
    # Numerator: sum(z_gen * z_gt)
    numerator = torch.sum(z_gen * z_gt)
    # Denominator: sum(z_gen^2)
    denominator = torch.sum(z_gen ** 2)
    
    if denominator < eps:
        s_hat = torch.tensor(1.0, device=z_gen.device)
    else:
        s_hat = numerator / denominator

    # 3. 校正生成序列 (Rectification)
    # 此时 z_gen_hat 和 z_gt 在同一个量纲下
    z_gen_aligned = s_hat * z_gen
    
    # 4. 计算 RMSE (Scale-Aligned RMSE)
    mse = torch.mean((z_gen_aligned - z_gt) ** 2)
    scde = torch.sqrt(mse)
    
    return scde.item()
    

def SubjectDetectionRate(detection_results: Any):  # noqa: ANN201
    """Compute viewpoint validity as the fraction of frames with human detections.

    Args:
        detection_results: List of detection results per frame.

    Returns:
        Fraction in [0.0, 1.0] of frames where a person was detected.
    """
    if not detection_results:
        return 0.0
    detected = sum(1 for res in detection_results if res is not None)
    return detected / len(detection_results)


def StructuralFidelity(
    ref_emb: Tensor,
    video_gen: Tensor,
    dinov2: Any,
    dino_transform: Any,
    detection_results: Any,
    device: Any,
):  # noqa: ANN201
    """
    """
    scores = []
    T = video_gen.shape[0]
    H, W = video_gen.shape[2], video_gen.shape[3]

    for i in range(T):
        res = detection_results[i]
        if res is None:
            scores.append(0.0)  # 惩罚没有检测到人的结果
            continue
        
        box, _ = res
        x1, y1, x2, y2 = map(int, box.tolist())
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)  # 边界保护
        if x2 - x1 < 10 or y2 - y1 < 10:
            scores.append(0.0)  # 框太小则忽略
            continue

        # Crop 人物区域提取特征
        person_crop = video_gen[i, :, y1:y2, x1:x2]  # [3, h, w]
        img_pil = to_pil_image(person_crop.cpu())  # 转 PIL -> Transform -> Tensor -> GPU
        input_tensor = dino_transform(img_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            curr_emb = dinov2(input_tensor)  # [1, D]

        sim = F.cosine_similarity(curr_emb, ref_emb)  # 计算余弦相似度
        scores.append(sim.item())

    return sum(scores) / len(scores) if scores else 0.0


def TemporalAttentionAlignment(
    gen_seq: torch.Tensor, 
    gt_seq: torch.Tensor, 
    temperature: float = None
) -> float:
    """
    计算 TAA (Temporal Attention Alignment) 指标。
    
    Args:
        gen_seq: [T, D] 生成视频的时序特征序列 (Query)
        gt_seq:  [T, D] 真值视频的时序特征序列 (Key)
        temperature:用于调节 Attention 锐度的系数，默认 sqrt(D)
        
    Returns:
        float: 对角线迹的均值 (Trace Mean), 范围 [0, 1]
    """
    T, D = gen_seq.shape
    if temperature is None:
        temperature = D ** 0.5
        
    # 1. 计算 QK^T (Logits) -> [T, T]
    # 行(Row)代表 Gen 的时间步，列(Col)代表 GT 的时间步
    scores = torch.matmul(gen_seq, gt_seq.transpose(0, 1)) / temperature
    
    # 2. Softmax (Row-wise)
    # 归一化每一行，看生成视频的第 t 秒主要关注 GT 的哪一秒
    attn_map = torch.softmax(scores, dim=-1) # [T, T]
    
    # 3. 计算对角线 Trace
    # diag() 提取对角线元素 [A_11, A_22, ..., A_TT]
    trace = torch.trace(attn_map)
    
    # 4. 均值化
    taa_score = trace / T
    
    return float(taa_score.item())


def TemporalFlickering(gen_frames: Tensor, gen_flows: Tensor, device):  # noqa: ANN201
    """Measure temporal flickering using warping consistency with optical flow.

    The function warps frame t-1 using the estimated flow and compares to frame t;
    the mean absolute difference (optionally masked to avoid borders) is used as
    the flickering score.

    Args:
        gen_frames: Generated video frames tensor [T, C, H, W].
        gen_flows: Optical flows for frames [T-1, 2, H, W].
        device: Torch device for computation.

    Returns:
        Mean absolute photometric difference (float).
    """
    T, C, H, W = gen_frames.shape
    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing='ij',
    )
    grid = torch.stack((grid_x, grid_y), dim=0).float().unsqueeze(0)
    grid = grid.expand(T - 1, -1, -1, -1)  # [T-1, 2, H, W]

    sampling_grid = grid + gen_flows
    sampling_grid[:, 0] = 2.0 * sampling_grid[:, 0] / (W - 1.0) - 1.0
    sampling_grid[:, 1] = 2.0 * sampling_grid[:, 1] / (H - 1.0) - 1.0
    sampling_grid = sampling_grid.permute(0, 2, 3, 1)  # [T-1, H, W, 2]

    frames_orig1 = gen_frames[:-1]
    frames_orig2 = gen_frames[1:]
    warped_prev = F.grid_sample(
        frames_orig1,
        sampling_grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )

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


def TrajectorySmoothness(global_motion: Tensor):
    """
    基于全局运动向量计算轨迹平滑度。
    Input: [T-1, 2] velocity vectors (dx, dy)
    Metric: Mean magnitude of acceleration.
    """
    if global_motion.shape[0] < 2:
        return 0.0
    
    # 1. 计算加速度 (二阶差分): a_t = v_t - v_{t-1}
    # global_motion 本身已经是 v_t
    acc = global_motion[1:] - global_motion[:-1] # [T-2, 2]
    
    # 2. 计算模长并取平均
    score = torch.mean(torch.norm(acc, p=2, dim=1))
    return score.item()


def calculate_metrics_based_flow_model(  # noqa: ANN201
    gen_frames: Tensor,
    gt_frames: Tensor = None,
    metrics_to_compute: Optional[Set[str]] = None,
    flow_model: torch.nn.Module = None,
    device: str = "cuda",
    video_id: Any = None,
    global_cache: Any = None,
):

    """Compute a set of optical-flow-based metrics using a RAFT flow model.

    The function computes generation flows, optionally caches them, and then
    computes requested metrics among: 'tf' (TemporalFlickering), 'ms'
    (MotionSmoothness), 'dd' (DynamicDegree), and 'ofc' (OpticalFlowCorrelation
    which requires ground-truth frames).

    Args:
        gen_frames: Generated frames tensor [T, C, H, W].
        gt_frames: Optional ground-truth frames tensor.
        metrics_to_compute: Set of metric keys to compute (subset of {'tf','ms','dd','ofc'}).
        flow_model: Preloaded RAFT model used to compute optical flow (required).
        device: Torch device to use (defaults to gen_frames.device).
        video_id: Optional identifier used as cache key.
        global_cache: Optional dict-like cache for storing computed flows.

    Returns:
        Dict mapping metric keys to numeric values (floats). Missing or skipped
        metrics are omitted; if flow computation fails, returns zeros for all.
    """
    if device is None: device = gen_frames.device
    assert flow_model is not None
    if metrics_to_compute is None: metrics_to_compute = set()

    # 1. Gen Flow (缓存读写)
    gen_flow_key = f"flow_gen_{video_id}"
    if global_cache is not None and gen_flow_key in global_cache:
        gen_flows = global_cache[gen_flow_key]
    else:
        gen_norm = (gen_frames * 2.0) - 1.0
        gen_flows = compute_flow(gen_norm, flow_model)
        if global_cache is not None and video_id is not None and gen_flows is not None:
            global_cache[gen_flow_key] = gen_flows

    if gen_flows is None:
        return {'tf': 0.0, 'ms': 0.0, 'dd': 0.0, 'ofc': 0.0, 'ts': 0.0}
    
    results = {}
    if 'dd' in metrics_to_compute:
        results['dd'] = DynamicDegree(gen_flows)
    if 'ms' in metrics_to_compute:
        results['ms'] = MotionSmoothness(gen_flows)
    if 'tf' in metrics_to_compute:
        results['tf'] = TemporalFlickering(gen_frames, gen_flows, device)
    
    if 'ts' in metrics_to_compute:
        # 全局运动池化 (Global Average Pooling)
        global_motion_gen = torch.mean(gen_flows, dim=[2, 3])

        # 计算 TS 分数
        results['ts'] = TrajectorySmoothness(global_motion_gen)
    
    if 'ofc' in metrics_to_compute and gt_frames is not None:
        # 2. GT Flow (缓存读写)
        gt_flow_key = f"flow_gt_{video_id}"
        if global_cache is not None and gt_flow_key in global_cache:
            gt_flows = global_cache[gt_flow_key]
        else:
            if len(gt_frames) >= 2:
                gt_norm = (gt_frames * 2.0) - 1.0
                gt_flows = compute_flow(gt_norm, flow_model)
                if global_cache is not None and video_id is not None:
                    global_cache[gt_flow_key] = gt_flows
            else:
                gt_flows = None

        if gt_flows is not None:
            global_motion_gen = torch.mean(gen_flows, dim=[2, 3])
            results['ofc'] = OpticalFlowCorrelation(global_motion_gen, gt_flows, device)
        else:
            results['ofc'] = 0.0

    return results