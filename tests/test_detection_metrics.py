import unittest
from unittest.mock import MagicMock, patch
import torch
import sys
import os

# 确保能导入 src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.Ego2ExoFollowShot.dimension.cce import CameraCenteringErrorEvaluator
from src.Ego2ExoFollowShot.dimension.vv import ViewpointValidityEvaluator
from src.Ego2ExoFollowShot.dimension.ta import TrajectoryAlignmentEvaluator

class TestDetectionCaching(unittest.TestCase):
    def setUp(self):
        self.device = 'cpu'
        self.video_id = "vid_cache_test"
        self.shared_cache = {}
        # 构造 Dummy Tensor (用于传参，内容不重要，因为我们Mock了底层函数)
        self.dummy_tensor = torch.zeros(10, 3, 224, 224)
        
        # 模拟检测结果: 2帧，都有检测框
        # 格式: List[(box_tensor, score_tensor)]
        self.mock_dets = [
            (torch.tensor([0,0,10,10]), torch.tensor(0.9)),
            (torch.tensor([20,20,30,30]), torch.tensor(0.8))
        ]

    @patch('utils.pretrain.get_detection_results')
    @patch('src.Ego2ExoFollowShot.dimension.metric.CameraCenteringError')
    def test_cce_vv_sharing(self, mock_calc_cce, mock_get_det):
        """测试 CCE 和 VV 是否共享检测结果缓存"""
        mock_get_det.return_value = self.mock_dets
        mock_calc_cce.return_value = 0.5 # 假定返回值
        
        # 初始化
        cce_eval = CameraCenteringErrorEvaluator(self.device)
        vv_eval = ViewpointValidityEvaluator(self.device)
        cce_eval.model = MagicMock()
        vv_eval.model = MagicMock()
        
        # 1. 运行 CCE (应该触发检测)
        print("\n[Test] Running CCE (Trigger Detection)...")
        cce_eval.compute(tensor_gen=self.dummy_tensor, video_id=self.video_id, global_cache=self.shared_cache)
        
        # 验证: 检测被调用1次
        mock_get_det.assert_called_once()
        # 验证: 缓存已写入
        cache_key = f"detection_gen_{self.video_id}"
        self.assertIn(cache_key, self.shared_cache)
        self.assertEqual(self.shared_cache[cache_key], self.mock_dets)
        
        # 2. 运行 VV (应该命中缓存)
        print("[Test] Running VV (Cache Hit)...")
        mock_get_det.reset_mock() # 重置调用计数
        vv_eval.compute(tensor_gen=self.dummy_tensor, video_id=self.video_id, global_cache=self.shared_cache)
        
        # 验证: 检测函数没有被再次调用
        mock_get_det.assert_not_called()
        print("✅ CCE & VV Caching Verified.")

    @patch('utils.pretrain.get_detection_results')
    @patch('src.Ego2ExoFollowShot.dimension.metric.TrajectoryAlignment')
    def test_ta_gen_gt_caching(self, mock_calc_ta, mock_get_det):
        """测试 TA 的 Gen 缓存命中和 GT 独立检测"""
        mock_get_det.return_value = self.mock_dets
        mock_calc_ta.return_value = 0.8
        
        ta_eval = TrajectoryAlignmentEvaluator(self.device)
        ta_eval.model = MagicMock()
        
        # 预先在缓存中填入 Gen 的结果 (模拟之前已经跑过 CCE)
        gen_key = f"detection_gen_{self.video_id}"
        self.shared_cache[gen_key] = self.mock_dets
        
        # 运行 TA
        print("\n[Test] Running TA (Gen Hit, GT Trigger)...")
        ta_eval.compute(
            tensor_gen=self.dummy_tensor, 
            tensor_gt=self.dummy_tensor, 
            video_id=self.video_id, 
            global_cache=self.shared_cache
        )
        
        # 验证: get_detection_results 应该只被调用 1 次 (为了 GT)
        # 因为 Gen 已经命中缓存了
        self.assertEqual(mock_get_det.call_count, 1)
        
        # 验证: GT 的结果也被缓存了
        gt_key = f"detection_gt_{self.video_id}"
        self.assertIn(gt_key, self.shared_cache)
        print("✅ TA Caching Verified.")

if __name__ == '__main__':
    unittest.main()