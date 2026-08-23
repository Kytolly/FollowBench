import torch
import torch.nn.functional as F
from typing import Any
from . import DimensionEvaluator

class MeanSquaredErrorEvaluator(DimensionEvaluator):
    """
    Mean Squared Error (MSE).
    衡量生成视频与 GT 视频在像素排列上的绝对差异。↓ 越低越好。
    """
    def prepare(self):
        pass

    def compute(self, tensor_gen: torch.Tensor, **kwargs: Any):
        tensor_gt = kwargs.get('tensor_gt')
        if tensor_gen.shape != tensor_gt.shape:
            tensor_gen = F.interpolate(tensor_gen, size=tensor_gt.shape[2:], mode='bilinear')
        mse_score = F.mse_loss(tensor_gen, tensor_gt).item()
        return mse_score