import torch
import torch.nn.functional as F
from torchmetrics.image import StructuralSimilarityIndexMeasure
from typing import Any
import piq

from . import DimensionEvaluator

class StructuralSimilarityIndexMeasureEvaluator(DimensionEvaluator):
    """
    Structural Similarity Index Measure (SSIM).
    """
    def prepare(self):
        self.data_range = 1.0
        self.ssim_metric = StructuralSimilarityIndexMeasure(data_range=self.data_range).to(self.device)

    def compute(self, tensor_gen: torch.Tensor, **kwargs) -> float:
        tensor_gt = kwargs.get('tensor_gt')
        min_frames = min(tensor_gen.shape[0], tensor_gt.shape[0])
        tensor_gen = tensor_gen[:min_frames]
        tensor_gt = tensor_gt[:min_frames]
        
        if tensor_gen.shape[2:] != tensor_gt.shape[2:]:
            tensor_gen = F.interpolate(tensor_gen, size=tensor_gt.shape[2:], mode='bilinear', align_corners=False)
        
        batch_size = 16 
        total_ssim = 0.0

        with torch.no_grad():
            for i in range(0, min_frames, batch_size):
                gen_chunk = tensor_gen[i : i + batch_size].to(self.device)
                gt_chunk = tensor_gt[i : i + batch_size].to(self.device)
                chunk_ssim = piq.ssim(gen_chunk, gt_chunk, data_range=self.data_range)
                actual_batch_frames = gen_chunk.shape[0]
                total_ssim += chunk_ssim.item() * actual_batch_frames
                del gen_chunk, gt_chunk
        return total_ssim / min_frames