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
        self.eval_size = (256, 256)
        self.batch_size = 8
        self.loss_fn = lpips.LPIPS(net='vgg').to(self.device)

    def compute(self, tensor_gen: torch.Tensor, **kwargs) -> float:
        tensor_gt = kwargs.get('tensor_gt')
        num_frames = tensor_gen.shape[0]
        total_score = 0.0

        with torch.no_grad():
            for i in range(0, num_frames, self.batch_size):
                gen_batch = tensor_gen[i : i + self.batch_size].to(self.device)
                gt_batch = tensor_gt[i : i + self.batch_size].to(self.device)
                gen_batch = F.interpolate(gen_batch, size=self.eval_size, mode='bilinear', align_corners=False)
                gt_batch = F.interpolate(gt_batch, size=self.eval_size, mode='bilinear', align_corners=False)
                
                gen_batch = gen_batch * 2.0 - 1.0
                gt_batch = gt_batch * 2.0 - 1.0
                score = self.loss_fn(gen_batch, gt_batch).mean()
                total_score += score.item() * gen_batch.shape[0]
        return total_score / num_frames