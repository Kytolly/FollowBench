from . import DimensionEvaluator
from ..utils.pretrain import load_laion_aes_vit
from .metric import AestheticQuality

class AestheticQualityEvaluator(DimensionEvaluator):
    def prepare(self):
        self.model = load_laion_aes_vit(self.device)
        super().prepare()

    def compute(self, **kwargs):
        video_gen = kwargs.get('tensor_gen')
        if video_gen is None:
            return 0.0 
        return AestheticQuality(video_gen, self.model)