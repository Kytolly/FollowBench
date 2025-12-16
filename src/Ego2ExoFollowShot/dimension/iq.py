from . import DimensionEvaluator
from utils import pretrain
from .metric import ImagingQuality

class ImagingQualityEvaluator(DimensionEvaluator):
    def prepare(self):
        self.model = pretrain.load_imaging_quality_metric(self.device)

    def compute(self, **kwargs):
        video_gen = kwargs.get('tensor_gen')
        if video_gen is None:
            return 0.0
        return ImagingQuality(video_gen, self.model)