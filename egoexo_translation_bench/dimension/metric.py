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


def BackgroundSemanticConsistency(
    ref_img_pil: Any,
    video_gen: Tensor,
    clip_model: Any,
    clip_proc: Any,
    detection_results: Any,
    device: Any,
):  # noqa: ANN201
    """Measure background semantic consistency using CLIP features.

    The function masks out detected person regions in each frame, extracts CLIP
    image features for the remaining background, and compares them to a
    reference image embedding.

    Args:
        ref_img_pil: Reference image as a PIL image.
        video_gen: Generated video tensor [T, 3, H, W].
        clip_model: Loaded CLIP image encoder.
        clip_proc: CLIP preprocessor that converts PIL image to model input.
        detection_results: List of detection results per frame.
        device: Torch device for computation.

    Returns:
        Mean cosine similarity between background embeddings and reference.
    """
    scores = []
    T = video_gen.shape[0]
    H, W = video_gen.shape[2], video_gen.shape[3]
    
    # 预计算参考图特征
    inputs_ref = clip_proc(images=ref_img_pil, return_tensors="pt").to(device)
    with torch.no_grad():
        ref_emb = clip_model.get_image_features(**inputs_ref)
    
    for i in range(T):
        frame_tensor = video_gen[i].clone()  # [3, H, W]
        
        # Mask 掉人物
        res = detection_results[i]
        if res is not None:
            box, _ = res
            x1, y1, x2, y2 = map(int, box.tolist())
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)
            frame_tensor[:, y1:y2, x1:x2] = 0.0  # Black out

        # 提取特征
        img_pil = to_pil_image(frame_tensor.cpu())
        inputs = clip_proc(images=img_pil, return_tensors="pt").to(device)
        
        with torch.no_grad():
            curr_emb = clip_model.get_image_features(**inputs)
            
        # 计算相似度
        sim = F.cosine_similarity(curr_emb, ref_emb)
        scores.append(sim.item())
        
    return sum(scores) / len(scores) if scores else 0.0


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


def CameraTrajectoryError(
    traj_gen: torch.Tensor, 
    traj_gt: torch.Tensor,
    eps: float = 1e-6
):
    """
    计算 CTE (Camera Trajectory Error) 的核心数学实现。
    使用 Umeyama 算法求解 Sim(3) 变换 (s, R, t) 并计算对齐后的 RMSE。

    Args:
        traj_gen: 生成视频的相机轨迹坐标 [N, 3] (Translation Vectors)
        traj_gt:  真实视频的相机轨迹坐标 [N, 3] (Translation Vectors)
        eps: 数值稳定性常数

    Returns:
        float: CTE Score (RMSE after alignment)
    """
    assert traj_gen.shape == traj_gt.shape, f"Shape mismatch: {traj_gen.shape} vs {traj_gt.shape}"
    N, D = traj_gen.shape  # N frames, D=3 dimensions

    # 1. 去中心化 (Centering)
    mu_gen = torch.mean(traj_gen, dim=0, keepdim=True) # [1, 3]
    mu_gt = torch.mean(traj_gt, dim=0, keepdim=True)   # [1, 3]

    y = traj_gen - mu_gen # [N, 3] (对应公式中的 y_i)
    q = traj_gt - mu_gt   # [N, 3] (对应公式中的 q_i)

    # 2. 计算协方差矩阵 (Covariance Matrix)
    # H = sum(y_i * q_i^T) -> [D, D]
    H = torch.matmul(y.transpose(0, 1), q) 

    # 3. SVD 分解求解旋转 R
    # H = U @ Sigma @ V.T
    U, _, Vh = torch.linalg.svd(H) 
    V = Vh.mH # PyTorch svd returns V^H (conjugate transpose), so V = Vh.mH
    
    # R = V @ S @ U.T
    # 构造修正矩阵 S (diag(1, 1, det))
    d = torch.det(torch.matmul(V, U.transpose(0, 1)))
    S_mat = torch.eye(D, device=traj_gen.device)
    S_mat[-1, -1] = d 
    
    R = torch.matmul(torch.matmul(V, S_mat), U.transpose(0, 1)) # [3, 3]

    # 4. 求解缩放 s (Scale)
    # s = sum(y^T R^T q) / sum(|y|^2)
    # 分子: trace(R^T @ y^T @ q) ??? 
    # 更简单的写法: sum element-wise product of (y @ R.T) and q
    y_rotated = torch.matmul(y, R.transpose(0, 1)) # [N, 3]
    
    numerator = torch.sum(y_rotated * q)
    denominator = torch.sum(y * y)
    
    if denominator < eps:
        s = torch.tensor(1.0, device=traj_gen.device)
    else:
        s = numerator / denominator

    # 5. 求解平移 t (Translation)
    # t = mu_gt - s * R * mu_gen
    # 注意维度: [1, 3] - s * [1, 3] @ [3, 3]^T 
    t = mu_gt - s * torch.matmul(mu_gen, R.transpose(0, 1))

    # 6. 变换生成的轨迹 (Apply Transform)
    # p_hat = s * p_gen @ R^T + t
    traj_gen_aligned = s * torch.matmul(traj_gen, R.transpose(0, 1)) + t

    # 7. 计算 CTE (RMSE)
    diff = traj_gt - traj_gen_aligned
    mse = torch.mean(torch.sum(diff ** 2, dim=1))
    cte = torch.sqrt(mse)

    return cte.item()


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