from . import DimensionEvaluator
from dimension.metrics import calculate_all_flow_metrics
from torchvision.models.optical_flow import raft_small, Raft_Small_Weights

class TemporalFlickeringEvaluator(DimensionEvaluator):
    def prepare(self, device='cuda'):
        self.device = device
        self.flow_model = raft_small(weights=Raft_Small_Weights.DEFAULT).to(device).eval()

    def compute(self, video_gen, video_ego, video_gt, path_ref, video_id=None, global_cache=None):
        # 缓存命中
        cache_key = f"temporal_consistency_{video_id}"
        if global_cache is not None and cache_key in global_cache:
            return global_cache[cache_key]['tf'] # 直接返回 Flickering
        
        # 缓存未命中
        metrics = calculate_all_flow_metrics(
            video_gen, 
            gt_frames=video_gt, # 传入 GT，激活 OFC 计算
            flow_model=self.flow_model
        )
        if global_cache is not None and video_id is not None:
            global_cache[cache_key] = {
                'tf': tf,
                'ms': ms,
                'dd': dd
            }
        return tf

    def clear(self):
        del self.flow_model
        super().clear()