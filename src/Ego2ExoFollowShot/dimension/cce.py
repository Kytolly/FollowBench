from . import DimensionEvaluator
from .metrics import CameraCenteringError
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
import torch

class CameraCenteringErrorEvaluator(DimensionEvaluator):
    def prepare(self, device='cuda'):
        self.device = device
        self.det = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT).to(device)
        self.det.eval()

    def compute(self, video_gen, video_ego, video_gt, path_ref):
        errors = []
        step = 5
        H, W = video_gen.shape[2], video_gen.shape[3]
        
        with torch.no_grad():
            for i in range(0, len(video_gen), step):
                pred = self.det(video_gen[i].unsqueeze(0))[0]
                valid = (pred['labels'] == 1) & (pred['scores'] > 0.7)
                if valid.any():
                    best_idx = torch.argmax(pred['scores'][valid])
                    box = pred['boxes'][valid][best_idx]
                    errors.append(CameraCenteringError(box, H, W))
                else:
                    errors.append(1.0) # 没检测到人，误差最大
        return sum(errors) / len(errors) if errors else 1.0