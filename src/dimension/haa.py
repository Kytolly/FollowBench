from torch import Tensor
from torchvision.models.detection import keypointrcnn_resnet50_fpn, KeypointRCNN_ResNet50_FPN_Weights

from . import DimensionEvaluator
from . import metric
import utils.pretrain

class HumanActionAlignmentEvaluator(DimensionEvaluator):
    def prepare(self):
        self.model = keypointrcnn_resnet50_fpn(weights=KeypointRCNN_ResNet50_FPN_Weights.DEFAULT).to(self.device).eval()

    def compute(self, **kwargs):
        """
        计算人体动作对齐度 (HAA)
        """
        video_gen: Tensor = kwargs.get('tensor_gen')
        video_gt: Tensor = kwargs.get('tensor_gt')
        video_id = kwargs.get('video_id')
        global_cache = kwargs.get('global_cache')
        
        gen_key = f"keypoint_gen_{video_id}"
        if global_cache is not None and gen_key in global_cache:
            # cache hits
            kp_gen = global_cache[gen_key]
        else: # cache not hits
            kp_gen = utils.pretrain.get_keypoint_results(video_gen, self.model)
            if global_cache is not None: 
                global_cache[gen_key] = kp_gen
            

        gt_key = f"keypoint_gt_{video_id}"
        if global_cache is not None and gt_key in global_cache:
            # cache hits
            kp_gt = global_cache[gt_key]
        else: # cache not hits
            kp_gt = utils.pretrain.get_keypoint_results(video_gt, self.model)
            if global_cache is not None: 
                global_cache[gt_key] = kp_gt

        H, W = video_gen.shape[2], video_gen.shape[3]
        return metric.HumanActionAlignment(kp_gen, kp_gt, H, W)