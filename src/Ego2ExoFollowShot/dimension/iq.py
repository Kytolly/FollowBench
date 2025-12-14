from . import DimensionEvaluator
from utils import pretrain
import torch

class ImagingQualityEvaluator(DimensionEvaluator):
    def __init__(self):
        super().__init__()
        
    def prepare(self, ):
        self.model = pretrain.load_imaging_quality_metric(self.device) # MUSIQ

    def compute(self, video_gen, *args):
        if self.model is None or video_gen is None: return 0.0
        
        batch_size = 4
        scores = []
        with torch.no_grad():
            for i in range(0, len(video_gen), batch_size):
                batch = video_gen[i : i + batch_size]
                scores.append(self.model(batch))
        
        if not scores: return 0.0
        return torch.cat(scores).mean().item()