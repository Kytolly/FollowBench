from . import DimensionEvaluator
from utils import pretrain, image_kit
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from Ego2ExoFollowShot.dimension.metrics import AppearanceConsistency
from PIL import Image

class AppearanceConsistencyEvaluator(DimensionEvaluator):
    def prepare(self):
        self.dinov2, self.dino_transform = pretrain.load_dinov2(self.device)
        self.det = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT).to(self.device)
        self.det.eval()

    def compute(self, video_gen, video_ego, video_gt, path_ref, **kwargs):
        # 1. 准备参考图 Feature
        ref_img = Image.open(path_ref).convert('RGB')
        ref_emb = image_kit.prepare_ref_embedding(self.dinov2, self.dino_transform, ref_img)
        
        # 2. 计算 (复用 metrics.py)
        return AppearanceConsistency(
            self.dinov2,
            self.dino_transform,
            self.det,
            ref_emb,
            video_gen # Tensor
        )
        
    def clear(self):
        del self.dinov2
        del self.det
        super().clear()