import unittest
import torch
import numpy as np
import sys
import os

# 确保能导入 src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.Ego2ExoFollowShot.dimension.metric import CameraCenteringError, ViewpointValidity, TrajectoryAlignment

class TestDetectionMetricAccuracy(unittest.TestCase):
    def test_cce_calculation(self):
        # ... (保持不变) ...
        H, W = 200, 200
        box_center = torch.tensor([90, 90, 110, 110], dtype=torch.float32)
        dets = [(box_center, 1.0)]
        score = CameraCenteringError(dets, H, W)
        self.assertAlmostEqual(score, 0.0, places=4)
        
        box_corner = torch.tensor([-10, -10, 10, 10], dtype=torch.float32)
        dets = [(box_corner, 1.0)]
        score = CameraCenteringError(dets, H, W)
        self.assertAlmostEqual(score, 1.0, places=4)

        dets = [None]
        score = CameraCenteringError(dets, H, W)
        self.assertEqual(score, 1.0)

    def test_vv_calculation(self):
        # ... (保持不变) ...
        dets = [("box", 1), ("box", 1), None, ("box", 1)]
        score = ViewpointValidity(dets)
        self.assertEqual(score, 0.75)
        self.assertEqual(ViewpointValidity([]), 0.0)

    def test_ta_calculation(self):
        """测试 TA (Trajectory Alignment) 计算"""
        H, W = 100, 100
        diag = np.sqrt(100**2 + 100**2)
        
        def make_det(x, y):
            # Box center at (x,y)
            return (torch.tensor([x-5, y-5, x+5, y+5], dtype=torch.float32), 1.0)
            
        # Case 1: 完美重合 (至少2帧)
        gen_dets = [make_det(10,10), make_det(20,20)] # [FIX] 增加到2帧
        gt_dets = [make_det(10,10), make_det(20,20)]
        score = TrajectoryAlignment(gen_dets, gt_dets, H, W)
        self.assertAlmostEqual(score, 0.0, places=4)
        
        # Case 2: 固定偏移 (至少2帧)
        # Gen: (10,10) -> (10,10)
        # GT:  (13,14) -> (13,14)
        # 偏移都是 dx=3, dy=4 -> dist=5
        gen_dets = [make_det(10,10), make_det(10,10)] # [FIX] 增加到2帧
        gt_dets = [make_det(13,14), make_det(13,14)]
        
        score = TrajectoryAlignment(gen_dets, gt_dets, H, W)
        expected = 5.0 / diag
        self.assertAlmostEqual(score, expected, places=4)
        
        # Case 3: 包含 None (漏检)
        # Gen: [Box, None]
        # GT:  [Box, Box]
        # 第一帧有距离，第二帧跳过，平均距离 = 第一帧距离
        gen_dets = [make_det(10,10), None]
        gt_dets = [make_det(13,14), make_det(20,20)]
        score = TrajectoryAlignment(gen_dets, gt_dets, H, W)
        self.assertAlmostEqual(score, expected, places=4) # 只有第一帧有效

if __name__ == '__main__':
    unittest.main()