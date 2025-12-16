from torchvision.models.optical_flow import raft_small, Raft_Small_Weights

from . import DimensionEvaluator
from .metrics import calculate_metrics_based_flow_model

class DynamicDegreeEvaluator(DimensionEvaluator):
    def prepare(self):
        self.model = raft_small(weights=Raft_Small_Weights.DEFAULT).to(self.device).eval()

    def compute(self, **kwargs):
        # parse kwargs
        tensor_gen = kwargs.get('tensor_gen')
        tensor_gt = kwargs.get('tensor_gt')
        video_id = kwargs.get('video_id')
        global_cache = kwargs.get('global_cache')
        metrics_to_compute = kwargs.get('metrics_to_compute', {'dd'}) # 默认只算自己
        
        # cache hits
        cache_key = f"temporal_consistency_{video_id}"
        if global_cache is not None and cache_key in global_cache:
            return global_cache[cache_key]['dd']
        
        # cache not hits
        self.prepare()
        metrics = calculate_metrics_based_flow_model(
                gen_frames=tensor_gen, 
                gt_frames=tensor_gt,
                metrics_to_compute=metrics_to_compute,
                flow_model=self.model, 
                device=self.device)
        if global_cache is not None and video_id is not None:
            global_cache[cache_key] = metrics
        return metrics['dd']