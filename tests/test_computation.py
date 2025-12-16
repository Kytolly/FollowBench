import unittest
import torch
import numpy as np
import sys
import os

# 确保能导入 src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 导入纯计算函数
from src.Ego2ExoFollowShot.dimension.metric import CameraCenteringError, ViewpointValidity, TrajectoryAlignment

class TestDetectionMetricAccuracy(unittest.TestCase):
    def test_cce_calculation(self):
        """测试 CCE (Camera Centering Error) 计算"""
        H, W = 200, 200
        center_x, center_y = 100.0, 100.0
        
        # Case 1: 完美居中
        # Box: [90, 90, 110, 110], Center=(100, 100)
        box_center = torch.tensor([90, 90, 110, 110], dtype=torch.float32)
        dets = [(box_center, 1.0)]
        score = CameraCenteringError(dets, H, W)
        self.assertAlmostEqual(score, 0.0, places=4, msg="Centered box should have 0 error")
        
        # Case 2: 位于左上角 (0,0)
        # Box Center at (0,0) -> distance to (100,100) is sqrt(100^2 + 100^2)
        # Max dist (normalization factor) is also sqrt(100^2 + 100^2)
        # So score should be 1.0
        box_corner = torch.tensor([-10, -10, 10, 10], dtype=torch.float32)
        dets = [(box_corner, 1.0)]
        score = CameraCenteringError(dets, H, W)
        self.assertAlmostEqual(score, 1.0, places=4, msg="Corner box should have 1.0 error")

        # Case 3: 未检测到 (None)
        dets = [None]
        score = CameraCenteringError(dets, H, W)
        self.assertEqual(score, 1.0, msg="Missed detection should result in max error 1.0")

    def test_vv_calculation(self):
        """测试 VV (Viewpoint Validity) 计算"""
        # 4帧: 3帧有人, 1帧无人 -> 0.75
        dets = [("box", 1), ("box", 1), None, ("box", 1)]
        score = ViewpointValidity(dets)
        self.assertEqual(score, 0.75)
        
        # 0帧
        self.assertEqual(ViewpointValidity([]), 0.0)

    def test_ta_calculation(self):
        """测试 TA (Trajectory Alignment) 计算"""
        H, W = 100, 100
        diag = np.sqrt(100**2 + 100**2)
        
        def make_det(x, y):
            # Box center at (x,y)
            return (torch.tensor([x-5, y-5, x+5, y+5], dtype=torch.float32), 1.0)
            
        # Case 1: 完美重合
        gen_dets = [make_det(10,10), make_det(20,20)]
        gt_dets = [make_det(10,10), make_det(20,20)]
        score = TrajectoryAlignment(gen_dets, gt_dets, H, W)
        self.assertAlmostEqual(score, 0.0, places=4)
        
        # Case 2: 固定偏移
        # Gen: (10,10)
        # GT: (13, 14) -> dx=3, dy=4 -> dist=5
        # Score = 5 / diag
        gen_dets = [make_det(10,10)]
        gt_dets = [make_det(13,14)]
        
        score = TrajectoryAlignment(gen_dets, gt_dets, H, W)
        expected = 5.0 / diag
        self.assertAlmostEqual(score, expected, places=4)
        
        # Case 3: 包含 None (漏检)
        # Gen: [Box, None]
        # GT:  [Box, Box]
        # 应该只计算第一帧的距离
        gen_dets = [make_det(10,10), None]
        gt_dets = [make_det(10,10), make_det(20,20)]
        score = TrajectoryAlignment(gen_dets, gt_dets, H, W)
        self.assertAlmostEqual(score, 0.0, places=4)

if __name__ == '__main__':
    unittest.main()