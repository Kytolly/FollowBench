from . import DimensionEvaluator
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
import torch

class ViewpointValidityEvaluator(DimensionEvaluator):
    def prepare(self):
        self.det = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT).to(self.device)
        self.det.eval()

    def compute(self, video_gen, video_ego, video_gt, path_ref, **kwargs):
        detected_count = 0
        total_samples = 0
        step = 5
        
        with torch.no_grad():
            for i in range(0, len(video_gen), step):
                total_samples += 1
                # video_gen[i]: [C, H, W] -> unsqueeze -> [1, C, H, W]
                pred = self.det(video_gen[i].unsqueeze(0))[0]
                # 检查是否有人
                if ((pred['labels'] == 1) & (pred['scores'] > 0.7)).any():
                    detected_count += 1
                    
        return detected_count / total_samples if total_samples > 0 else 0.0