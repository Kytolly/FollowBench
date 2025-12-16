import unittest
from unittest.mock import MagicMock, patch
import torch
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.Ego2ExoFollowShot.dimension.aq import AestheticQualityEvaluator
from src.Ego2ExoFollowShot.dimension.iq import ImagingQualityEvaluator
from src.Ego2ExoFollowShot.dimension.fvd import FrechetVideoDistanceEvaluator

class TestQualityMetrics(unittest.TestCase):
    def setUp(self):
        self.device = 'cpu'
        self.video_id = "vid_qual_test"
        self.shared_cache = {}
        # [T, C, H, W]
        self.dummy_tensor = torch.rand(5, 3, 224, 224) 

    @patch('utils.pretrain.load_aesthetic_metric')
    def test_aq(self, mock_load):
        """测试 AQ (运行真实 Metric 逻辑，Mock 模型)"""
        # 1. 配置 Mock 模型
        mock_model = MagicMock()
        # 模型返回 Tensor [B, 1]
        mock_model.return_value = torch.tensor([0.8]) 
        mock_load.return_value = mock_model
        
        # 2. 初始化 Evaluator
        evaluator = AestheticQualityEvaluator(self.device)
        evaluator.model = mock_model # 强制注入 Mock
        
        # 3. 运行
        score = evaluator.compute(tensor_gen=self.dummy_tensor)
        
        # 4. 验证
        # 5 帧，batch=8 -> 1次推理，返回 0.8 -> mean = 0.8
        self.assertAlmostEqual(score, 0.8)

    @patch('utils.pretrain.load_imaging_quality_metric')
    def test_iq(self, mock_load):
        """测试 IQ (运行真实 Metric 逻辑，Mock 模型)"""
        # 1. 配置 Mock 模型
        mock_model = MagicMock()
        mock_model.return_value = torch.tensor([0.6])
        mock_load.return_value = mock_model
        
        evaluator = ImagingQualityEvaluator(self.device)
        evaluator.model = mock_model
        
        # 2. 运行
        score = evaluator.compute(tensor_gen=self.dummy_tensor)
        
        # 3. 验证
        self.assertAlmostEqual(score, 0.6)

    @patch('utils.video_kit.extract_i3d_features')
    @patch('src.Ego2ExoFollowShot.dimension.metric.FrechetVideoDistance')
    @patch('utils.pretrain.load_i3d')
    def test_fvd_caching(self, mock_load, mock_dist, mock_extract):
        """测试 FVD 缓存 (Mock 特征提取)"""
        # 模拟特征提取: 返回 Numpy Array (因为 metric.extract_i3d_features 现在返回 numpy)
        feat_dim = 10
        # 两次调用: Gen 和 GT
        mock_extract.side_effect = [np.zeros((1, feat_dim)), np.ones((1, feat_dim))]
        mock_dist.return_value = 100.0
        
        evaluator = FrechetVideoDistanceEvaluator(self.device)
        evaluator.model = MagicMock()
        
        # 1. 运行 (Cache Miss)
        print("\n[Test] FVD: Running (Cache Miss)...")
        evaluator.compute(
            tensor_gen=self.dummy_tensor,
            tensor_gt=self.dummy_tensor,
            video_id=self.video_id,
            global_cache=self.shared_cache
        )
        self.assertEqual(mock_extract.call_count, 2)
        self.assertIn(f"i3d_feat_gen_{self.video_id}", self.shared_cache)
        
        # 2. 运行 (Cache Hit)
        print("[Test] FVD: Running (Cache Hit)...")
        mock_extract.reset_mock()
        evaluator.compute(
            tensor_gen=self.dummy_tensor,
            tensor_gt=self.dummy_tensor,
            video_id=self.video_id,
            global_cache=self.shared_cache
        )
        mock_extract.assert_not_called()
        print("✅ FVD Caching Verified.")

if __name__ == '__main__':
    unittest.main()