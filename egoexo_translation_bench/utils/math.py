"""Mathematical utilities for the EgoExo Translation Benchmark.

This module provides mathematical functions and statistical computations
used throughout the benchmark evaluation pipeline.
"""

import torch
from torch import Tensor
from typing import Optional
from scipy.stats import pearsonr


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
