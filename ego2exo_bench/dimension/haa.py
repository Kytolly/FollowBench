from torch import Tensor
from torchvision.models.detection import keypointrcnn_resnet50_fpn, KeypointRCNN_ResNet50_FPN_Weights

from . import DimensionEvaluator
from .metric import HumanActionAlignment
from ..utils.pretrain import get_keypoint_results, load_keypoint_rcnn

class HumanActionAlignmentEvaluator(DimensionEvaluator):
    def prepare(self):
        self.model = load_keypoint_rcnn(self.device)
        super().prepare()

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
            kp_gen = get_keypoint_results(video_gen, self.model)
            if global_cache is not None: 
                global_cache[gen_key] = kp_gen
            

        gt_key = f"keypoint_gt_{video_id}"
        if global_cache is not None and gt_key in global_cache:
            # cache hits
            kp_gt = global_cache[gt_key]
        else: # cache not hits
            kp_gt = get_keypoint_results(video_gt, self.model)
            if global_cache is not None: 
                global_cache[gt_key] = kp_gt

        H, W = video_gen.shape[2], video_gen.shape[3]
        return HumanActionAlignment(kp_gen, kp_gt, H, W)