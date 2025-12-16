import unittest
import torch
import numpy as np
from unittest.mock import MagicMock
import sys
import os

# 确保能导入 src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.Ego2ExoFollowShot.dimension.metrics import calculate_all_flow_metrics

class TestVBenchConsistency(unittest.TestCase):
    def setUp(self):
        self.device = 'cpu'
        self.height = 32
        self.width = 32
        self.frames = 5 # 增加帧数以便构造变化的序列
        
        self.mock_flow_model = MagicMock()
        
    def _create_moving_video_and_flow(self, shift_x=1.0, shift_y=1.0):
        """构造匀速运动"""
        T = self.frames
        base_frame = torch.zeros(1, 3, self.height, self.width)
        base_frame[:, :, 10:22, 15:17] = 1.0 
        base_frame[:, :, 15:17, 10:22] = 1.0
        
        video_frames = []
        for t in range(T):
            dx = -1.0 * t * shift_x
            dy = -1.0 * t * shift_y
            theta = torch.tensor([[1.0, 0.0, -dx*2/self.width], [0.0, 1.0, -dy*2/self.height]]).unsqueeze(0)
            grid = torch.nn.functional.affine_grid(theta, base_frame.size(), align_corners=True)
            frame = torch.nn.functional.grid_sample(base_frame, grid, align_corners=True)
            video_frames.append(frame)
            
        video_tensor = torch.cat(video_frames, dim=0)
        
        # 完美光流：全图恒定值
        flow_tensor = torch.zeros(T-1, 2, self.height, self.width)
        flow_tensor[:, 0, :, :] = shift_x
        flow_tensor[:, 1, :, :] = shift_y
        
        return video_tensor, flow_tensor

    def test_metrics_constant_motion(self):
        """
        场景：匀速运动
        用于测试 DD, MS, TF。
        OFC 在此场景下因方差为0会返回0，故不在此测试 OFC=1.0。
        """
        shift_x, shift_y = 1.0, 1.0
        expected_speed = np.sqrt(shift_x**2 + shift_y**2)
        
        gen_video, perfect_flow = self._create_moving_video_and_flow(shift_x, shift_y)
        
        # Mock 返回恒定光流
        self.mock_flow_model.side_effect = lambda img1, img2: [perfect_flow]
        
        metrics = calculate_all_flow_metrics(
            gen_frames=gen_video, 
            gt_frames=gen_video,
            flow_model=self.mock_flow_model,
            metrics_to_compute={'tf', 'ms', 'dd'} # 不测 ofc
        )
        
        print("\n[Test] Constant Motion Results:")
        print(f"  DD (Expected ~{expected_speed:.3f}): {metrics['dd']:.4f}")
        print(f"  MS (Expected 0.0): {metrics['ms']:.4f}")
        print(f"  TF (Expected ~0.0): {metrics['tf']:.4f}")

        self.assertAlmostEqual(metrics['dd'], expected_speed, places=4)
        self.assertAlmostEqual(metrics['ms'], 0.0, places=4)
        self.assertLess(metrics['tf'], 0.01)

    def test_ofc_correctness_variable_motion(self):
        """
        场景：变速运动 (专门测试 OFC)
        构造一个速度变化的序列，确保方差 > 0，从而 Pearson Correlation 有意义。
        """
        # 手动构造变化的全局运动序列: 1, 2, 3, 4
        # T=5 帧 -> 4 个光流场
        T = 5
        flows = []
        for i in range(T-1):
            f = torch.zeros(1, 2, self.height, self.width)
            val = float(i + 1) # 速度 1.0, 2.0, 3.0, 4.0
            f[:, 0, :, :] = val # x轴运动
            f[:, 1, :, :] = val # y轴运动
            flows.append(f)
        
        variable_flow = torch.cat(flows, dim=0) # [4, 2, H, W]
        
        # Mock 模型
        self.mock_flow_model.side_effect = lambda img1, img2: [variable_flow]
        
        # 随便造个视频tensor占位，因为光流被mock了
        dummy_video = torch.zeros(T, 3, self.height, self.width)
        
        # 计算：传入相同的视频作为 Gen 和 GT
        metrics = calculate_all_flow_metrics(
            gen_frames=dummy_video, 
            gt_frames=dummy_video, # GT = Gen
            flow_model=self.mock_flow_model,
            metrics_to_compute={'ofc'}
        )
        
        print("\n[Test] Variable Motion (OFC Check):")
        print(f"  OFC (Expected 1.0): {metrics['ofc']:.4f}")
        
        # 因为 Gen 和 GT 的光流完全一致且有变化，相关性应为 1.0
        self.assertAlmostEqual(metrics['ofc'], 1.0, places=4, msg="OFC should be 1.0 for identical variable motion")

if __name__ == '__main__':
    unittest.main()