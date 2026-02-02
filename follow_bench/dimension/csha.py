from typing import Any
import logging
logger = logging.getLogger(__name__)

import torch

from ..dimension import DimensionEvaluator
from .metric import CameraSubjectHeadingAlignment
from ..utils.wham_wrapper import WhamWrapper

class CameraSubjectHeadingAlignmentEvaluator(DimensionEvaluator):
    """
    Camera-Subject Heading Alignment (CSHA) Evaluator.
    
    Metric:
        Measures the stability of the relative azimuth angle between the camera 
        and the subject's heading direction.
        
        Low 'stability_score' -> Camera strictly follows the subject's rotation.
        Low 'locking_score'   -> Camera stays strictly behind the subject.
    """
    def prepare(self):
        self.solver = WhamWrapper(device=self.device)
            
    def compute(self, **kwargs: Any):
        """
        Compute CSHA score for the GENERATED video.
        CSHA is often an 'Absolute Quality' metric. We want to know if the 
        generated camera control is stable, regardless of whether the GT 
        camera was stable or erratic.
        
        (Optional: You can also compute |CSHA_gen - CSHA_gt| if strict adherence 
        to GT's specific camera behavior is required).
        """
        prediction = kwargs.get('tensor_gen')
        # 1. 提取生成视频的几何信息
        try:
            geo_info = self.solver.extract_full_geometry(prediction)
        except Exception as e:
            logger.error(f"CSHA Geometry Extraction Failed: {e}")
            return 0.0
            
        if geo_info is None:
            return 0.0

        # 2. 计算 CSHA 数学指标
        # 假设 SMPL Forward 是 +Z (Standard for many pipelines, adjust if WHAM differs)
        metrics = CameraSubjectHeadingAlignment(
            cam_pos=geo_info['cam_pos'],
            subj_pos=geo_info['subj_pos'],
            subj_orient_aa=geo_info['subj_orient'],
            forward_axis='+z' 
        )
        
        # 3. 返回分数
        # 这里默认返回 Stability Score (越低越好，表示夹角方差小，运镜稳)
        # 如果您的任务是 "Locked Follow" (如赛车/TPS游戏)，也可以返回 locking_score
        return metrics['stability_score']

    def clear(self):
        if self.solver is not None:
            self.solver.clear()
            self.solver = None
        torch.cuda.empty_cache()