import numpy as np
from scipy.linalg import sqrtm

import torch
from torch import Tensor
import torch.nn.functional as F
from torchvision.transforms.functional import to_pil_image
from typing import Any, Optional, Set

from ..utils.math import p_corr
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


def AppearanceConsistency(
    ref_emb: Tensor,
    video_gen: Tensor,
    dinov2: Any,
    dino_transform: Any,
    detection_results: Any,
    device: Any,
):  # noqa: ANN201
    """Measure appearance consistency (AC) between generated video and a reference.

    For each frame, the function crops the detected person bounding box, extracts
    a DINOv2 embedding for the crop, and computes cosine similarity to the
    provided reference embedding. Small/missing detections are penalized.

    Args:
        ref_emb: Reference embedding tensor of shape [1, D].
        video_gen: Generated video tensor [T, 3, H, W].
        dinov2: DINOv2 model callable that accepts preprocessed PIL input.
        dino_transform: Transform that converts PIL image to model input tensor.
        detection_results: List of detection results per frame or None.
        device: Torch device to run the model on.

    Returns:
        Mean cosine similarity score (float) in [-1, 1], or 0.0 if no valid detections.
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


def OpticalFlowCorrelation(gen_flows: Tensor, gt_flows: Tensor, device: Any):  # noqa: ANN201
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
    min_len = min(len(gen_flows), len(gt_flows))
    g_flow = gen_flows[:min_len]
    t_flow = gt_flows[:min_len]

    g_motion = g_flow.mean(dim=[2, 3])  # [t, 2]
    t_motion = t_flow.mean(dim=[2, 3])  # [t, 2]
    
    corr_x = p_corr(g_motion[:, 0], t_motion[:, 0], device)
    corr_y = p_corr(g_motion[:, 1], t_motion[:, 1], device)
    ofc = ((corr_x + corr_y) / 2.0).item()
    return ofc


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


def TrajectoryAlignment(gen_results, gt_results, H, W):  # noqa: ANN201
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


def ViewpointValidity(detection_results: Any):  # noqa: ANN201
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


def calculate_metrics_based_flow_model(  # noqa: ANN201
    gen_frames: Tensor,
    gt_frames: Optional[Tensor] = None,
    metrics_to_compute: Optional[Set[str]] = None,
    flow_model: Any = None,
    device: Any = None,
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
        return {'tf': 0.0, 'ms': 0.0, 'dd': 0.0, 'ofc': 0.0}
    
    results = {}
    if 'dd' in metrics_to_compute:
        results['dd'] = metric.DynamicDegree(gen_flows)
    if 'ms' in metrics_to_compute:
        results['ms'] = metric.MotionSmoothness(gen_flows)
    if 'tf' in metrics_to_compute:
        results['tf'] = metric.TemporalFlickering(gen_frames, gen_flows, device)

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
            results['ofc'] = metric.OpticalFlowCorrelation(gen_flows, gt_flows, device)
        else:
            results['ofc'] = 0.0

    return results