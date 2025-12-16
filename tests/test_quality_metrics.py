import unittest
from unittest.mock import MagicMock, patch
import torch
import numpy as np
import sys
import os

# 确保能导入 src
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

    @patch('src.Ego2ExoFollowShot.dimension.metric.AestheticQuality')
    @patch('utils.pretrain.load_aesthetic_metric')
    def test_aq(self, mock_load, mock_calc):
        """测试 AQ 流程"""
        mock_load.return_value = MagicMock()
        mock_calc.return_value = 5.5
        
        evaluator = AestheticQualityEvaluator(self.device)
        # Mock 内部模型
        evaluator.model = MagicMock()
        
        score = evaluator.compute(tensor_gen=self.dummy_tensor)
        self.assertEqual(score, 5.5)
        mock_calc.assert_called_once()

    @patch('src.Ego2ExoFollowShot.dimension.metric.ImagingQuality')
    @patch('utils.pretrain.load_imaging_quality_metric')
    def test_iq(self, mock_load, mock_calc):
        """测试 IQ 流程"""
        mock_load.return_value = MagicMock()
        mock_calc.return_value = 0.8
        
        evaluator = ImagingQualityEvaluator(self.device)
        evaluator.model = MagicMock()
        
        score = evaluator.compute(tensor_gen=self.dummy_tensor)
        self.assertEqual(score, 0.8)

    @patch('src.Ego2ExoFollowShot.dimension.metric.extract_i3d_features')
    @patch('src.Ego2ExoFollowShot.dimension.metric.FrechetVideoDistance')
    @patch('utils.pretrain.load_i3d')
    def test_fvd_caching(self, mock_load, mock_dist, mock_extract):
        """测试 FVD 缓存"""
        # 模拟特征提取返回 Numpy Array (因为 metric.extract_i3d_features 现在返回 numpy)
        feat_dim = 10
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