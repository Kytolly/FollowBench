import torch
import torch.nn.functional as F
from torchmetrics.image import StructuralSimilarityIndexMeasure
from typing import Any
from . import DimensionEvaluator

class StructuralSimilarityIndexMeasureEvaluator(DimensionEvaluator):
    """
    Structural Similarity Index Measure (SSIM).
    衡量物体轮廓、边缘和光影的保留完整度。↑ 越高越好 (最接近 1)。
    """
    def prepare(self):
        # 初始化 torchmetrics 的 SSIM，假定像素范围是 0.0 ~ 1.0
        self.ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)

    def compute(self, tensor_gen: torch.Tensor, **kwargs: Any):
        tensor_gt = kwargs.get('tensor_exo')
        if tensor_gt is None:
            raise ValueError("SSIMEvaluator requires ground truth 'tensor_exo'.")
            
        if tensor_gen.shape != tensor_gt.shape:
            tensor_gen = F.interpolate(tensor_gen, size=tensor_gt.shape[2:], mode='bilinear')

        with torch.no_grad():
            # torchmetrics SSIM 期望输入格式为 (N, C, H, W)
            score = self.ssim_metric(tensor_gen, tensor_gt).item()
            
        return score