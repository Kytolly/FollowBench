"""Mathematical utilities for the EgoExo Translation Benchmark.

This module provides mathematical functions and statistical computations
used throughout the benchmark evaluation pipeline.
"""

import torch
from torch import Tensor
from typing import Optional
from scipy.stats import pearsonr


def axis_angle_to_matrix(axis_angle: torch.Tensor):
    """
    使用 Rodrigues 公式将轴角转换为旋转矩阵 [..., 3, 3]
    """
    angle = torch.norm(axis_angle, dim=-1, keepdim=True) # [..., 1]
    axis = axis_angle / (angle + 1e-6) # [..., 3]
    
    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one_minus_cos = 1 - cos
    
    x, y, z = axis[..., 0], axis[..., 1], axis[..., 2]
    
    # 构建旋转矩阵
    # R = I + sin(theta) * K + (1-cos(theta)) * K^2
    # 这里直接展开写
    zeros = torch.zeros_like(x)
    
    R = torch.stack([
        cos.squeeze(-1) + x*x*one_minus_cos.squeeze(-1), 
        x*y*one_minus_cos.squeeze(-1) - z*sin.squeeze(-1), 
        x*z*one_minus_cos.squeeze(-1) + y*sin.squeeze(-1),
        
        y*x*one_minus_cos.squeeze(-1) + z*sin.squeeze(-1), 
        cos.squeeze(-1) + y*y*one_minus_cos.squeeze(-1), 
        y*z*one_minus_cos.squeeze(-1) - x*sin.squeeze(-1),
        
        z*x*one_minus_cos.squeeze(-1) - y*sin.squeeze(-1), 
        z*y*one_minus_cos.squeeze(-1) + x*sin.squeeze(-1), 
        cos.squeeze(-1) + z*z*one_minus_cos.squeeze(-1)
    ], dim=-1).reshape(axis_angle.shape[:-1] + (3, 3))
    
    return R

def wrap_to_pi(angles: torch.Tensor):
    """将角度映射到 [-pi, pi]"""
    return (angles + torch.pi) % (2 * torch.pi) - torch.pi

def p_corr(x: Tensor, y: Tensor, device: Optional[str] = None) -> Tensor:
    """Compute Pearson correlation coefficient between two tensors.
    
    Calculates the Pearson correlation coefficient, which measures the linear
    correlation between two variables. The result ranges from -1 to 1, where:
    - 1 indicates perfect positive correlation
    - 0 indicates no linear correlation  
    - -1 indicates perfect negative correlation
    
    Args:
        x: First input tensor
        y: Second input tensor (must have same shape as x)
        device: Optional device to place the result tensor on
        
    Returns:
        Pearson correlation coefficient as a scalar tensor
        
    Note:
        Returns 0.0 if either input has standard deviation < 1e-6 to avoid
        division by zero in degenerate cases.
    """
    # Center the variables by subtracting their means
    vx = x - x.mean()
    vy = y - y.mean()
    
    # Handle degenerate cases where standard deviation is too small
    if x.std() < 1e-6 or y.std() < 1e-6:
        return torch.tensor(0.0, device=device)
    
    # Compute Pearson correlation coefficient
    numerator = (vx * vy).sum()
    denominator = torch.sqrt((vx**2).sum()) * torch.sqrt((vy**2).sum()) + 1e-8
    
    return numerator / denominator
