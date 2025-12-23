from typing import Any

from . import DimensionEvaluator
from ..utils.pretrain import load_musiq
from .metric import ImagingQuality


class ImagingQualityEvaluator(DimensionEvaluator):
    """Evaluator for Imaging Quality (IQ).

    Uses a pretrained MUSIQ model to evaluate image/video perceptual quality.
    """

    def prepare(self):  # noqa: ANN201
        """Load the MUSIQ model onto the configured device and call superclass prepare.

        The method stores the loaded model on ``self.model`` and then calls
        ``super().prepare()`` to retain the base class preparation steps.
        """
        self.model = load_musiq(self.device)
        super().prepare()

    def compute(self, **kwargs: Any):  # noqa: ANN201
        """Compute IQ for the generated video.

        Args:
            **kwargs: Keyword arguments from the evaluation pipeline. Expects
                'tensor_gen' for the generated video frames.

        Returns:
            A numeric score (float) representing imaging quality, or 0.0 if
            no generated frames are provided.
        """
        video_gen = kwargs.get('tensor_gen')
        if video_gen is None:
            return 0.0
        return ImagingQuality(video_gen, self.model)
