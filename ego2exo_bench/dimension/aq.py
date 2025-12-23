from . import DimensionEvaluator
from ..utils.pretrain import load_laion_aes_vit
from .metric import AestheticQuality

class AestheticQualityEvaluator(DimensionEvaluator):
    """Evaluator wrapper for aesthetic quality (LAION-Aesthetics).

    Loads the pretrained LAION aesthetic ViT model and computes mean aesthetics
    score for generated videos using `AestheticQuality`.
    """

    def prepare(self):
        """Load the LAION aesthetic model and call base prepare."""
        self.model = load_laion_aes_vit(self.device)
        super().prepare()

    def compute(self, **kwargs):
        """Compute the aesthetics score for the generated video.

        Args:
            **kwargs: expects 'tensor_gen' containing video frames.

        Returns:
            Mean aesthetic score (float) or 0.0 if no frames provided.
        """
        video_gen = kwargs.get('tensor_gen')
        if video_gen is None:
            return 0.0 
        return AestheticQuality(video_gen, self.model)