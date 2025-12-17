from torchvision.models.optical_flow import raft_small, Raft_Small_Weights

from . import DimensionEvaluator
from .metric import calculate_metrics_based_flow_model
from src.utils.pretrain import load_raft

class MotionSmoothnessEvaluator(DimensionEvaluator):
    def prepare(self):
        self.model = load_raft(self.device)
        super().prepare()

    def compute(self, **kwargs):
        # parse kwargs
        video_gen = kwargs.get('tensor_gen')
        video_id = kwargs.get('video_id')
        global_cache = kwargs.get('global_cache')
        metrics_to_compute = kwargs.get('metrics_to_compute', {'ms'}) # 默认只算自己
        
        results = calculate_metrics_based_flow_model(
            gen_frames=video_gen,
            metrics_to_compute=metrics_to_compute,
            flow_model=self.model,
            device=self.device,
            video_id=video_id,
            global_cache=global_cache
        )
        return results['ms']