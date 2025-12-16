from torchvision.models.optical_flow import raft_small, Raft_Small_Weights

from . import DimensionEvaluator
from .metrics import calculate_all_flow_metrics

class OpticalFlowCorrelationEvaluator(DimensionEvaluator):
    def prepare(self, device='cuda'):
        self.device = device
        self.model = raft_small(weights=Raft_Small_Weights.DEFAULT).to(device).eval()

    def compute(self, **kwargs):
        # parse kwargs
        tensor_gen = kwargs.get('tensor_gen')
        tensor_gt = kwargs.get('tensor_gt')
        video_id = kwargs.get('video_id')
        global_cache = kwargs.get('global_cache')
        metrics_to_compute = kwargs.get('metrics_to_compute', {'ofc'}) # 默认只算自己
        
        # cache hits
        cache_key = f"temporal_consistency_{video_id}"
        if global_cache is not None and cache_key in global_cache:
            return global_cache[cache_key]['ofc']
        
        # cache not hits
        metrics = calculate_all_flow_metrics(
                gen_frames=tensor_gen, 
                gt_frames=tensor_gt,
                metrics_to_compute=metrics_to_compute,
                flow_model=self.flow_model, 
                device=self.device)
        if global_cache is not None and video_id is not None:
            global_cache[cache_key] = metrics
        return metrics['ofc']

    def clear(self):
        del self.flow_model
        super().clear()