from . import DimensionEvaluator
from dimension.metrics import OpticalFlowCorrelation
from utils.video_kit import tensor_to_numpy

class OpticalFlowCorrelationEvaluator(DimensionEvaluator):
    def compute(self, video_gen, video_ego, video_gt, path_ref, **kwargs):
        # 转换: GPU Tensor -> CPU Numpy List
        gen_frames = tensor_to_numpy(video_gen)
        gt_frames = tensor_to_numpy(video_gt) # OFC 对比的是 GT
        
        return OpticalFlowCorrelation(gen_frames, gt_frames) 