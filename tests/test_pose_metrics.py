# tests/test_pose_metrics.py

import unittest
from unittest.mock import MagicMock, patch
import torch
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.Ego2ExoFollowShot.dimension.haa import HumanActionAlignmentEvaluator
from src.Ego2ExoFollowShot.dimension.metric import HumanActionAlignment 

class TestPoseMetrics(unittest.TestCase):
    def setUp(self):
        self.device = 'cpu'
        self.video_id = "vid_pose_test"
        self.shared_cache = {}
        # H=224, W=224
        self.H, self.W = 224, 224
        self.dummy_tensor = torch.zeros(2, 3, self.H, self.W)
        
        # 构造关键点
        self.kp_center = torch.ones(17, 3) * 100.0
        self.kp_center[:, 2] = 1.0 
        
        self.mock_kp_dets = [
            (self.kp_center, torch.tensor(0.95)),
            (self.kp_center, torch.tensor(0.95))
        ]

    @patch('utils.pretrain.get_keypoint_results')
    @patch('src.Ego2ExoFollowShot.dimension.metric.HumanActionAlignment')
    def test_haa_caching(self, mock_calc_haa, mock_get_kp):
        """测试 HAA 的缓存机制"""
        # ... (这部分逻辑不变，略) ...
        mock_get_kp.return_value = self.mock_kp_dets
        mock_calc_haa.return_value = 0.05
        
        evaluator = HumanActionAlignmentEvaluator(self.device)
        evaluator.model = MagicMock()
        
        evaluator.compute(
            video_gen=self.dummy_tensor, video_gt=self.dummy_tensor,
            video_ego=None, path_ref=None,
            video_id=self.video_id, global_cache=self.shared_cache
        )
        self.assertEqual(mock_get_kp.call_count, 2)
        # ...

    def test_haa_calculation_accuracy(self):
        """测试 HAA 的核心计算逻辑 (动态归一化)"""
        H, W = 200, 200 # 使用方便计算的尺寸
        diag = np.sqrt(H**2 + W**2)
        
        # Case 1: 完美重合
        kp_gen = torch.ones(17, 3) * 50.0
        kp_gen[:, 2] = 0.9 
        kp_gt = kp_gen.clone()
        
        gen_res = [(kp_gen, 0.9)]
        gt_res = [(kp_gt, 0.9)]
        
        # [FIX] 传入 H, W
        score = HumanActionAlignment(gen_res, gt_res, H, W)
        self.assertAlmostEqual(score, 0.0, places=4)

        # Case 2: 固定偏移
        # 1 个关键点偏移 100 像素
        kp_gt_shift = kp_gen.clone()
        kp_gt_shift[0, 0] += 100.0 
        
        gen_res = [(kp_gen, 0.9)]
        gt_res = [(kp_gt_shift, 0.9)]

        # 理论计算:
        # 平均像素误差 = (100.0 + 0*16) / 17 ≈ 5.882
        # 归一化误差 = 5.882 / diag
        avg_pixel_error = 100.0 / 17.0
        expected_score = avg_pixel_error / diag
        
        # [FIX] 传入 H, W
        score = HumanActionAlignment(gen_res, gt_res, H, W)
        self.assertAlmostEqual(score, expected_score, places=4)
        
        print(f"✅ HAA Calculation Verified (Diag={diag:.2f}, Score={score:.4f})")

if __name__ == '__main__':
    unittest.main()