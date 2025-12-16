from . import DimensionEvaluator
from transformers import CLIPProcessor, CLIPModel
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from Ego2ExoFollowShot.dimension.metrics import BackgroundSemanticConsistency
from PIL import Image

class BackgroundSemanticConsistencyEvaluator(DimensionEvaluator):
    def prepare(self):
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.det = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT).to(self.device)
        self.det.eval()

    def compute(self, video_gen, video_ego, video_gt, path_ref, **kwargs):
        ref_img = Image.open(path_ref).convert('RGB')
        
        return BackgroundSemanticConsistency(
            self.clip,
            self.proc,
            self.det,
            ref_img,
            video_gen # Tensor
        )

    def clear(self):
        del self.clip
        del self.proc
        del self.det
        super().clear()