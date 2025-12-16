import torch
from torch import Tensor
import torch.nn.functional as F
import utils

def DynamicDegree(gen_flows: Tensor):
    '''计算视频的动态程度 (from vbench)'''
    # gen_flows: [T-1, 2, H, W]
    flow_mags = torch.norm(gen_flows, p=2, dim=1) # [T-1, H, W]
    dd = flow_mags.mean().item()
    return dd

def MotionSmoothness(gen_flows: Tensor):
    '''计算运动平滑性 (from vbench)'''
    # gen_flows: [T-1, 2, H, W]
    if len(gen_flows) > 1:
        flow_acc = gen_flows[1:] - gen_flows[:-1]
        ms = torch.norm(flow_acc, p=2, dim=1).mean().item()
    else:
        ms = 0.0
    return ms

def TemporalFlickering(gen_frames: Tensor, gen_flows: Tensor, device):
    '''计算时间闪烁 局部的高频细节的不真实 '''
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

def OpticalFlowCorrelation(gen_flows: Tensor, gt_flows: Tensor, device):
    min_len = min(len(gen_flows), len(gt_flows))
    g_flow = gen_flows[:min_len]
    t_flow = gt_flows[:min_len]

    g_motion = g_flow.mean(dim=[2, 3]) # [t, 2]
    t_motion = t_flow.mean(dim=[2, 3]) # [t, 2]
    
    corr_x = utils.math.p_corr(g_motion[:, 0], t_motion[:, 0], device)
    corr_y = utils.math.p_corr(g_motion[:, 1], t_motion[:, 1], device)
    ofc = ((corr_x + corr_y) / 2.0).item()
    return ofc