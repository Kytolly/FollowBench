# src/Ego2ExoFollowShot/dimension/dd.py
from . import DimensionEvaluator
from .metrics import calculate_temporal_consistency
from torchvision.models.optical_flow import raft_small, Raft_Small_Weights

class DynamicDegreeEvaluator(DimensionEvaluator):
    def prepare(self, device='cuda'):
        self.device = device
        self.flow_model = raft_small(weights=Raft_Small_Weights.DEFAULT).to(device).eval()

    def compute(self, video_gen, video_ego, video_gt, path_ref, video_id=None, global_cache=None):
        # 缓存命中
        cache_key = f"temporal_consistency_{video_id}"
        if global_cache is not None and cache_key in global_cache:
            return global_cache[cache_key]['dd']
        
        # 缓存未命中
        tf, ms, dd = calculate_temporal_consistency(video_gen, flow_model=self.flow_model, device=self.device)
        if global_cache is not None and video_id is not None:
            global_cache[cache_key] = {'tf': tf, 'ms': ms, 'dd': dd}
        return dd

    def clear(self):
        del self.flow_model
        super().clear()