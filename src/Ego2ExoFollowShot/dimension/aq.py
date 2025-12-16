import torch

from . import DimensionEvaluator
from utils import pretrain, video_kit
from .metric import AestheticQuality

class AestheticQualityEvaluator(DimensionEvaluator):
    def prepare(self):
        self.model = pretrain.load_aesthetic_metric(self.device)

    def compute(self, **kwargs):
        video_gen = kwargs.get('tensor_gen')
        if video_gen is None:
            return 0.0 
        return AestheticQuality(video_gen, self.model)