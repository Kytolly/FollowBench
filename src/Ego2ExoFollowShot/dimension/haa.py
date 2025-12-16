from . import DimensionEvaluator
from dimension.metrics import HumanActionAlignment
from torchvision.models.detection import keypointrcnn_resnet50_fpn, KeypointRCNN_ResNet50_FPN_Weights
from utils.video_kit import tensor_to_numpy

class HumanActionAlignmentEvaluator(DimensionEvaluator):
    def prepare(self):
        self.model = keypointrcnn_resnet50_fpn(weights=KeypointRCNN_ResNet50_FPN_Weights.DEFAULT).to(self.device)
        self.model.eval()

    def compute(self, video_gen, video_ego, video_gt, path_ref, **kwargs):
        # HAA 需要 Numpy List 格式
        gen_frames = tensor_to_numpy(video_gen)
        gt_frames = tensor_to_numpy(video_gt)
        
        return HumanActionAlignment(gen_frames, gt_frames, self.model, self.device)