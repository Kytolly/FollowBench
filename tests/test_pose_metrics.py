import unittest
from unittest.mock import MagicMock, patch
import torch
import numpy as np
import sys
import os

# 确保能导入 src
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
        
        # 构造关键点: [17, 3] (x, y, confidence)
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
        # 设置 Mock 返回值
        mock_get_kp.return_value = self.mock_kp_dets
        mock_calc_haa.return_value = 0.05
        
        evaluator = HumanActionAlignmentEvaluator(self.device)
        # Mock 模型以避免加载权重
        evaluator.model = MagicMock()
        
        # 1. 运行 HAA (缓存未命中)
        # 注意：这里使用 tensor_gen 和 tensor_gt，因为 haa.py 代码中使用 kwargs.get('tensor_gen')
        print("\n[Test] HAA: Running (Cache Miss)...")
        evaluator.compute(
            tensor_gen=self.dummy_tensor, 
            tensor_gt=self.dummy_tensor,
            video_id=self.video_id, 
            global_cache=self.shared_cache
        )
        
        # 验证: 检测函数被调用了 2 次 (一次 Gen, 一次 GT)
        self.assertEqual(mock_get_kp.call_count, 2)
        # 验证: 缓存中存在 Gen 和 GT 的结果
        self.assertIn(f"keypoint_gen_{self.video_id}", self.shared_cache)
        self.assertIn(f"keypoint_gt_{self.video_id}", self.shared_cache)

        # 2. 再次运行 (缓存命中)
        print("[Test] HAA: Running (Cache Hit)...")
        mock_get_kp.reset_mock() # 重置计数器
        
        evaluator.compute(
            tensor_gen=self.dummy_tensor, 
            tensor_gt=self.dummy_tensor,
            video_id=self.video_id, 
            global_cache=self.shared_cache
        )
        
        # 验证: 检测函数未被调用 (使用了缓存)
        mock_get_kp.assert_not_called()
        print("✅ HAA Caching Verified.")

    def test_haa_calculation_accuracy(self):
        """测试 HAA 的核心计算逻辑 (metric.py)"""
        H, W = 200, 200 
        # metric.py 中使用了 + 1e-6 防止除以零
        diag = np.sqrt(H**2 + W**2) + 1e-6
        
        # Case 1: 完美重合 (Expected: 0.0)
        kp_gen = torch.ones(17, 3) * 50.0
        kp_gen[:, 2] = 0.9 # High confidence
        kp_gt = kp_gen.clone()
        
        gen_res = [(kp_gen, 0.9)]
        gt_res = [(kp_gt, 0.9)]
        
        # 传入 H, W
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
        
        score = HumanActionAlignment(gen_res, gt_res, H, W)
        self.assertAlmostEqual(score, expected_score, places=4)
        
        print(f"✅ HAA Calculation Verified (Score={score:.4f})")

if __name__ == '__main__':
    unittest.main()