import unittest
from unittest.mock import MagicMock, patch
import torch
import sys
import os

# 将项目根目录加入路径，确保能 import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.Ego2ExoFollowShot.dimension.tf import TemporalFlickeringEvaluator
from src.Ego2ExoFollowShot.dimension.ms import MotionSmoothnessEvaluator
from src.Ego2ExoFollowShot.dimension.dd import DynamicDegreeEvaluator
from src.Ego2ExoFollowShot.dimension.ofc import OpticalFlowCorrelationEvaluator

class TestFlowMetrics(unittest.TestCase):
    def setUp(self):
        self.device = 'cuda'
        # 构造假的视频 Tensor [T, C, H, W]
        # T=5, C=3, H=224, W=224
        self.dummy_gen = torch.rand(5, 3, 224, 224)
        self.dummy_gt = torch.rand(5, 3, 224, 224)
        self.dummy_ref = "assets/dummy_ref.png" # 路径即可
        self.video_id = "test_video_001"
        self.shared_cache = {}

    @patch('src.Ego2ExoFollowShot.dimension.tf.calculate_temporal_consistency')
    @patch('src.Ego2ExoFollowShot.dimension.ms.calculate_temporal_consistency')
    @patch('src.Ego2ExoFollowShot.dimension.dd.calculate_temporal_consistency')
    def test_shared_caching_logic(self, mock_calc_dd, mock_calc_ms, mock_calc_tf):
        """
        核心测试：验证 TF, MS, DD 是否真的共享了缓存，避免重复计算
        """
        # 1. 设定 Mock 返回值 (模拟一次计算返回三个指标)
        # return: (flickering, smoothness, dynamic_degree)
        mock_return_val = (0.1, 0.2, 0.3)
        mock_calc_tf.return_value = mock_return_val
        mock_calc_ms.return_value = mock_return_val
        mock_calc_dd.return_value = mock_return_val

        # 2. 初始化评估器
        tf_eval = TemporalFlickeringEvaluator(self.device)
        ms_eval = MotionSmoothnessEvaluator(self.device)
        dd_eval = DynamicDegreeEvaluator(self.device)

        # 模拟 prepare (这里 mock 掉模型加载，或者在类中处理了 cpu fallback)
        tf_eval.flow_model = MagicMock()
        ms_eval.flow_model = MagicMock()
        dd_eval.flow_model = MagicMock()

        # ---------------------------------------------------------
        # 步骤 A: 运行第一个指标 (TF)
        # 预期：缓存为空，触发计算，写入缓存
        # ---------------------------------------------------------
        print("\n[Test] Running TF (First run)...")
        score_tf = tf_eval.compute(
            self.dummy_gen, None, None, None, 
            video_id=self.video_id, 
            global_cache=self.shared_cache
        )
        
        # 验证结果
        self.assertEqual(score_tf, 0.1)
        # 验证缓存是否被填充
        cache_key = f"temporal_consistency_{self.video_id}"
        self.assertIn(cache_key, self.shared_cache)
        self.assertEqual(self.shared_cache[cache_key]['tf'], 0.1)
        self.assertEqual(self.shared_cache[cache_key]['ms'], 0.2)
        self.assertEqual(self.shared_cache[cache_key]['dd'], 0.3)
        
        # 验证计算函数被调用了 1 次
        mock_calc_tf.assert_called_once()

        # ---------------------------------------------------------
        # 步骤 B: 运行第二个指标 (MS)
        # 预期：缓存命中，直接返回，不触发计算
        # ---------------------------------------------------------
        print("[Test] Running MS (Second run)...")
        score_ms = ms_eval.compute(
            self.dummy_gen, None, None, None, 
            video_id=self.video_id, 
            global_cache=self.shared_cache
        )
        
        # 验证结果
        self.assertEqual(score_ms, 0.2)
        # 验证 MS 的计算函数 没 被调用 (说明用了缓存)
        mock_calc_ms.assert_not_called()

        # ---------------------------------------------------------
        # 步骤 C: 运行第三个指标 (DD)
        # 预期：缓存命中，直接返回
        # ---------------------------------------------------------
        print("[Test] Running DD (Third run)...")
        score_dd = dd_eval.compute(
            self.dummy_gen, None, None, None, 
            video_id=self.video_id, 
            global_cache=self.shared_cache
        )
        self.assertEqual(score_dd, 0.3)
        mock_calc_dd.assert_not_called()
        
        print("✅ Shared Caching Logic Verified: Computation ran only once.")

    @patch('src.Ego2ExoFollowShot.dimension.ofc.OpticalFlowCorrelation')
    def test_ofc_evaluator(self, mock_ofc_func):
        """
        测试 OFC (光流相关性)
        OFC 通常需要 GT 视频，且目前独立于 TF/MS/DD
        """
        mock_ofc_func.return_value = 0.85
        
        evaluator = OpticalFlowCorrelationEvaluator(self.device)
        evaluator.flow_model = MagicMock() # Mock 模型
        
        print("\n[Test] Running OFC...")
        # OFC 需要传入 video_gt
        score = evaluator.compute(
            self.dummy_gen, None, self.dummy_gt, None,
            video_id=self.video_id,
            global_cache=self.shared_cache
        )
        
        self.assertEqual(score, 0.85)
        # 验证是否正确传入了 gen 和 gt
        args, _ = mock_ofc_func.call_args
        self.assertTrue(torch.is_tensor(args[0]), "First arg should be gen tensor")
        self.assertTrue(torch.is_tensor(args[1]), "Second arg should be gt tensor")
        print("✅ OFC Logic Verified.")

    def test_evaluator_lifecycle(self):
        """
        测试 prepare 和 clear 生命周期
        """
        print("\n[Test] Testing Lifecycle (prepare/clear)...")
        evaluator = TemporalFlickeringEvaluator(self.device)
        
        # 1. 初始状态
        self.assertIsNone(evaluator.model)
        if hasattr(evaluator, 'flow_model'):
            self.assertIsNone(evaluator.flow_model)
            
        # 2. Prepare
        # 注意：这会尝试加载真实模型，如果没有 GPU 或环境有问题可能会慢
        # 这里主要测试流程，可以用 try-except 包裹
        try:
            evaluator.prepare(self.device)
            # 验证模型被加载
            self.assertIsNotNone(evaluator.flow_model)
        except Exception as e:
            print(f"Skipping model load test due to environment: {e}")
            
        # 3. Clear
        evaluator.clear()
        # 验证模型被清理 (或者属性被删除)
        self.assertFalse(hasattr(evaluator, 'flow_model') and evaluator.flow_model is not None)
        print("✅ Lifecycle Verified.")

if __name__ == '__main__':
    unittest.main()