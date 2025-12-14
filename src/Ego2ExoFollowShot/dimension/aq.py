from . import DimensionEvaluator
from utils import pretrain, video_kit
import torch

class AestheticQualityEvaluator(DimensionEvaluator):
    def __init__(self):
        super().__init__()
        
    def prepare(self):
        self.model = pretrain.load_aesthetic_metric(self.device)

    def compute(self, path_gen, *kwargs):
        video_tensor = video_kit.load_video_as_tensor(str(path_gen)).to(self.device) # [T, C, H, W]
        if self.model is None: return 0.0
        with torch.no_grad():
            score = self.model(video_tensor).mean().item()
        return score