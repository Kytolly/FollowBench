from PIL import Image

from transformers import CLIPProcessor, CLIPModel
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights

from . import metric
from . import DimensionEvaluator
from utils import pretrain

class BackgroundSemanticConsistencyEvaluator(DimensionEvaluator):
    def prepare(self):
        # 加载 CLIP
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        # 加载 Detector
        self.det = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT).to(self.device)
        self.det.eval()

    def compute(self, **kwargs):
        # parse kwargs
        video_gen = kwargs.get('tensor_gen')
        pillow_ref: Image = kwargs.get('pillow_ref')
        video_id = kwargs.get('video_id')
        global_cache = kwargs.get('global_cache')
        self.prepare()
        

        cache_key = f"detection_gen_{video_id}"
        if global_cache is not None and cache_key in global_cache:
            # cache hits
            detections = global_cache[cache_key]
        else: # cache not hits
            detections = pretrain.get_detection_results(video_gen, self.det)
            if global_cache is not None:
                global_cache[cache_key] = detections

        return metric.BackgroundSemanticConsistency(
            ref_img_pil=pillow_ref,
            video_gen=video_gen,
            clip_model=self.clip,
            clip_proc=self.proc,
            detection_results=detections,
            device=self.device
        )

    def clear(self):
        del self.clip
        del self.proc
        del self.det
        super().clear()