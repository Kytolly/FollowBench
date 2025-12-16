from PIL import Image

from torch import Tensor
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights

from . import metric
from . import DimensionEvaluator
from utils import pretrain, image_kit

class AppearanceConsistencyEvaluator(DimensionEvaluator):
    def prepare(self):
        # 加载 DINOv2
        self.dinov2, self.dino_transform = pretrain.load_dinov2(self.device)
        # 加载 Detector (以备缓存未命中)
        self.det = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT).to(self.device)
        self.det.eval()

    def compute(self, **kwargs):
        # parse kwargs
        video_gen = kwargs.get('tensor_gen')
        pillow_ref: Image = kwargs.get('pillow_ref')
        video_id = kwargs.get('video_id')
        global_cache = kwargs.get('global_cache')
        
        # 准备参考图 Embedding
        self.prepare()
        ref_emb: Tensor = image_kit.prepare_ref_embedding(self.dinov2, self.dino_transform, pillow_ref)
        ref_emb.to(self.device)
        
        # 获取检测结果 (优先读缓存)
        cache_key = f"detection_gen_{video_id}"
        if global_cache is not None and cache_key in global_cache:
            # cache hits
            detections = global_cache[cache_key]
        else:
            detections = pretrain.get_detection_results(video_gen, self.det)
            if global_cache is not None:
                global_cache[cache_key] = detections

        return metric.AppearanceConsistency(
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