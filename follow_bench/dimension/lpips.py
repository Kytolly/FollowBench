import torch
import torch.nn.functional as F
import lpips
from typing import Any
from . import DimensionEvaluator

class LearnedPerceptualImagePatchSimilarityEvaluator(DimensionEvaluator):
    """
    Learned Perceptual Image Patch Similarity (LPIPS).
    高度契合人类主观视觉感受，衡量语义和质感差异。↓ 越低越好。
    """
    def prepare(self):
        # 默认使用 VGG 网络，这是 LPIPS 的标准配置
        # 预训练权重将在第一次运行时自动下载
        self.loss_fn = lpips.LPIPS(net='vgg').to(self.device)

    def compute(self, tensor_gen: torch.Tensor, **kwargs: Any):
        tensor_gt = kwargs.get('tensor_exo')
        if tensor_gt is None:
            raise ValueError("LPIPSEvaluator requires ground truth 'tensor_exo'.")

        if tensor_gen.shape != tensor_gt.shape:
            tensor_gen = F.interpolate(tensor_gen, size=tensor_gt.shape[2:], mode='bilinear')

        # LPIPS 库期望输入范围是 [-1, 1]
        gen_scaled = tensor_gen * 2.0 - 1.0
        gt_scaled = tensor_gt * 2.0 - 1.0

        with torch.no_grad():
            # 逐帧计算 LPIPS，然后求平均代表整个视频的感知误差
            # 输入: (T, C, H, W)
            loss = self.loss_fn(gen_scaled, gt_scaled).mean()
            
        return loss.item()