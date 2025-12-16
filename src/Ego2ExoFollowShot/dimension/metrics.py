import torch
from torch import Tensor
from torchvision.ops import roi_align

import utils.video_kit
from . import metric

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
    gen_flows = utils.video_kit.compute_flow(gen_norm, flow_model) # [T-1, 2, H, W]
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
            gt_flows = utils.video_kit.compute_flow(gt_norm, flow_model)
            results['ofc'] = metric.OpticalFlowCorrelation(gen_flows, gt_flows, device)
        else:
            results['ofc'] = 0.0

    return results