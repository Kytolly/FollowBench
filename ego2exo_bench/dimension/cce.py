from torch import Tensor
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights

from . import DimensionEvaluator
from .metric import CameraCenteringError
from ..utils.pretrain import get_detection_results, load_faster_rcnn

class CameraCenteringErrorEvaluator(DimensionEvaluator):
    def prepare(self):
        self.model = load_faster_rcnn(self.device)
        super().prepare()

    def compute(self, **kwargs):
        # parse kwargs
        video_gen: Tensor = kwargs.get('tensor_gen')
        video_id = kwargs.get('video_id')
        global_cache = kwargs.get('global_cache')

        cache_key = f"detection_gen_{video_id}"
        if global_cache is not None and cache_key in global_cache:        
            # cache hits
            detections = global_cache[cache_key]
        else:# cache not hits
            detections = get_detection_results(video_gen, self.model)
            if global_cache is not None:
                global_cache[cache_key] = detections
        
        H, W = video_gen.shape[2], video_gen.shape[3]
        return CameraCenteringError(detections, H, W)