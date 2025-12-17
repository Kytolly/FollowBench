from . import DimensionEvaluator
from src.utils.pretrain import load_musiq
from .metric import ImagingQuality

class ImagingQualityEvaluator(DimensionEvaluator):
    def prepare(self):
        self.model = load_musiq(self.device)

    def compute(self, **kwargs):
        video_gen = kwargs.get('tensor_gen')
        if video_gen is None:
            return 0.0
        return ImagingQuality(video_gen, self.model)