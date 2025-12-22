from PIL import Image

from torch import Tensor
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights

from . import DimensionEvaluator
from .metric import AppearanceConsistency
from ..utils.pretrain import load_dinov2, load_faster_rcnn, get_detection_results
from ..utils.image_kit import prepare_ref_embedding

class AppearanceConsistencyEvaluator(DimensionEvaluator):
    def prepare(self):
        self.dinov2, self.dino_transform = load_dinov2(self.device)
        self.det = load_faster_rcnn(self.device)
        super().prepare()

    def compute(self, **kwargs):
        # parse kwargs
        video_gen = kwargs.get('tensor_gen')
        pillow_ref: Image = kwargs.get('pillow_ref')
        video_id = kwargs.get('video_id')
        global_cache = kwargs.get('global_cache')
        
        # 准备参考图 Embedding
        ref_emb: Tensor = prepare_ref_embedding(self.dinov2, self.dino_transform, pillow_ref, self.device)
        assert ref_emb is not None
        ref_emb.to(self.device)
        
        # 获取检测结果
        cache_key = f"detection_gen_{video_id}"
        if global_cache is not None and cache_key in global_cache:
            # cache hits
            detections = global_cache[cache_key]
        else:
            detections = get_detection_results(video_gen, self.det)
            if global_cache is not None:
                global_cache[cache_key] = detections

        return AppearanceConsistency(
            ref_emb=ref_emb,
            video_gen=video_gen,
            dinov2=self.dinov2, 
            dino_transform=self.dino_transform, 
            detection_results=detections,
            device=self.device
        )
        
    def clear(self):
        del self.dinov2
        del self.det
        super().clear()