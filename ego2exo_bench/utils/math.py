import torch
from torch import Tensor

def p_corr(x: Tensor, y: Tensor, device=None):
    # Pearson Correlation
    vx = x - x.mean(); vy = y - y.mean()
    if x.std() < 1e-6 or y.std() < 1e-6: return torch.tensor(0.0, device=device)
    return (vx * vy).sum() / (torch.sqrt((vx**2).sum()) * torch.sqrt((vy**2).sum()) + 1e-8)