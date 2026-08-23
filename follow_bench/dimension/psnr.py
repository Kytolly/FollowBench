import torch
import torch.nn.functional as F
import math
import logging

from . import DimensionEvaluator

logger = logging.getLogger(__name__)

class PeakSignaltoNoiseRatioEvaluator(DimensionEvaluator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data_range = 1.0

    def prepare(self):
        pass

    def compute(self, tensor_gen: torch.Tensor, **kwargs) -> float:
        tensor_gt = kwargs.get('tensor_gt')
        min_frames = min(tensor_gen.shape[0], tensor_gt.shape[0])
        tensor_gen = tensor_gen[:min_frames]
        tensor_gt = tensor_gt[:min_frames]
        
        if tensor_gen.shape[2:] != tensor_gt.shape[2:]:
            tensor_gen = F.interpolate(tensor_gen, size=tensor_gt.shape[2:], mode='bilinear', align_corners=False)
        tensor_gen = tensor_gen.to(self.device)
        tensor_gt = tensor_gt.to(self.device)
        mse = F.mse_loss(tensor_gen, tensor_gt)
        psnr = 10 * torch.log10((self.data_range ** 2) / mse)
        return psnr.item()