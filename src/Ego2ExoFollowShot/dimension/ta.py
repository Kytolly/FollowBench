from . import DimensionEvaluator
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
import torch
import numpy as np

class TrajectoryAlignmentEvaluator(DimensionEvaluator):
    def prepare(self):
        self.det = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT).to(self.device)
        self.det.eval()

    def _get_traj(self, video_tensor):
        traj = []
        if video_tensor is None or len(video_tensor) == 0: return []
        
        H, W = video_tensor.shape[2], video_tensor.shape[3]
        diag = np.sqrt(H**2 + W**2)
        
        with torch.no_grad():
            # 可以考虑 batch 处理以加速，这里简化为逐帧
            for i in range(len(video_tensor)):
                pred = self.det(video_tensor[i].unsqueeze(0))[0]
                valid = (pred['labels'] == 1) & (pred['scores'] > 0.7)
                if valid.any():
                    best_idx = torch.argmax(pred['scores'][valid])
                    box = pred['boxes'][valid][best_idx].cpu().numpy()
                    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                    traj.append(np.array([cx/diag, cy/diag]))
                else:
                    traj.append(None)
        return traj

    def compute(self, video_gen, video_ego, video_gt, path_ref, **kwargs):
        traj_gen = self._get_traj(video_gen)
        traj_gt = self._get_traj(video_gt)
        
        min_len = min(len(traj_gen), len(traj_gt))
        if min_len < 2: return 1.0
        
        dists = []
        for i in range(min_len):
            if traj_gen[i] is not None and traj_gt[i] is not None:
                dists.append(np.linalg.norm(traj_gen[i] - traj_gt[i]))
        
        return np.mean(dists) if dists else 1.0