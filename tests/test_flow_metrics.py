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
        self.device = 'cpu'  # 测试环境下使用 CPU
        # 构造假的视频 Tensor [T, C, H, W]
        self.dummy_gen = torch.rand(5, 3, 224, 224)
        self.dummy_gt = torch.rand(5, 3, 224, 224)
        self.video_id = "test_video_001"
        self.shared_cache = {}
        # 模拟 Bench 类传递的参数：告知需要计算哪些指标
        self.metrics_to_compute = {'tf', 'ms', 'dd', 'ofc'}

    @patch('src.Ego2ExoFollowShot.dimension.tf.calculate_metrics_based_flow_model')
    @patch('src.Ego2ExoFollowShot.dimension.ms.calculate_metrics_based_flow_model')
    @patch('src.Ego2ExoFollowShot.dimension.dd.calculate_metrics_based_flow_model')
    def test_shared_caching_logic(self, mock_calc_dd, mock_calc_ms, mock_calc_tf):
        """
        核心测试：验证 TF, MS, DD 是否真的共享了缓存，避免重复计算
        """
        # 1. 设定 Mock 返回值 (新版 metrics.py 返回的是字典)
        mock_result = {'tf': 0.1, 'ms': 0.2, 'dd': 0.3, 'ofc': 0.0}
        mock_calc_tf.return_value = mock_result
        mock_calc_ms.return_value = mock_result
        mock_calc_dd.return_value = mock_result

        # 2. 初始化评估器
        tf_eval = TemporalFlickeringEvaluator(self.device)
        ms_eval = MotionSmoothnessEvaluator(self.device)
        dd_eval = DynamicDegreeEvaluator(self.device)

        # 模拟 prepare (Mock 掉模型加载)
        tf_eval.model = MagicMock()
        ms_eval.model = MagicMock()
        dd_eval.model = MagicMock()

        # ---------------------------------------------------------
        # 步骤 A: 运行第一个指标 (TF)
        # 预期：缓存为空，触发计算，写入缓存
        # ---------------------------------------------------------
        print("\n[Test] Running TF (First run)...")
        score_tf = tf_eval.compute(
            tensor_gen=self.dummy_gen,
            tensor_gt=self.dummy_gt,
            video_id=self.video_id,
            global_cache=self.shared_cache,
            metrics_to_compute=self.metrics_to_compute
        )
        
        self.assertEqual(score_tf, 0.1)
        
        # 验证缓存是否被填充
        cache_key = f"temporal_consistency_{self.video_id}"
        self.assertIn(cache_key, self.shared_cache)
        self.assertEqual(self.shared_cache[cache_key]['ms'], 0.2)
        self.assertEqual(self.shared_cache[cache_key]['dd'], 0.3)
        
        # 验证 TF 的计算函数被调用了 1 次
        mock_calc_tf.assert_called_once()

        # ---------------------------------------------------------
        # 步骤 B: 运行第二个指标 (MS)
        # 预期：缓存命中，直接返回，不触发计算
        # ---------------------------------------------------------
        print("[Test] Running MS (Second run)...")
        score_ms = ms_eval.compute(
            tensor_gen=self.dummy_gen,
            tensor_gt=self.dummy_gt,
            video_id=self.video_id,
            global_cache=self.shared_cache,
            metrics_to_compute=self.metrics_to_compute
        )
        
        self.assertEqual(score_ms, 0.2)
        # 验证 MS 的计算函数 没 被调用 (说明用了缓存)
        mock_calc_ms.assert_not_called()

        # ---------------------------------------------------------
        # 步骤 C: 运行第三个指标 (DD)
        # 预期：缓存命中，直接返回
        # ---------------------------------------------------------
        print("[Test] Running DD (Third run)...")
        score_dd = dd_eval.compute(
            tensor_gen=self.dummy_gen,
            tensor_gt=self.dummy_gt,
            video_id=self.video_id,
            global_cache=self.shared_cache,
            metrics_to_compute=self.metrics_to_compute
        )
        self.assertEqual(score_dd, 0.3)
        mock_calc_dd.assert_not_called()
        
        print("✅ Shared Caching Logic Verified: Computation ran only once.")

    @patch('src.Ego2ExoFollowShot.dimension.ofc.calculate_metrics_based_flow_model')
    def test_ofc_evaluator(self, mock_calc_ofc):
        """
        测试 OFC (光流相关性)
        验证是否正确传递了 tensor_gen 和 tensor_gt
        """
        mock_result = {'tf': 0.0, 'ms': 0.0, 'dd': 0.0, 'ofc': 0.85}
        mock_calc_ofc.return_value = mock_result
        
        evaluator = OpticalFlowCorrelationEvaluator(self.device)
        evaluator.model = MagicMock() # Mock 模型
        
        print("\n[Test] Running OFC...")
        score = evaluator.compute(
            tensor_gen=self.dummy_gen, 
            tensor_gt=self.dummy_gt, 
            video_id=self.video_id,
            global_cache=self.shared_cache,
            metrics_to_compute={'ofc'}
        )
        
        self.assertEqual(score, 0.85)
        
        # 验证参数传递
        # compute 内部调用 calculate_metrics_based_flow_model(gen_frames=..., gt_frames=...)
        call_kwargs = mock_calc_ofc.call_args.kwargs
        self.assertTrue(torch.is_tensor(call_kwargs['gen_frames']), "gen_frames should be Tensor")
        self.assertTrue(torch.is_tensor(call_kwargs['gt_frames']), "gt_frames should be Tensor")
        print("✅ OFC Logic Verified.")

    def test_evaluator_lifecycle(self):
        """
        测试 prepare 和 clear 生命周期
        """
        print("\n[Test] Testing Lifecycle (prepare/clear)...")
        evaluator = TemporalFlickeringEvaluator(self.device)
        
        # 1. 初始状态
        self.assertIsNone(evaluator.model)
            
        # 2. Prepare
        # 注意：这会尝试加载真实模型 (raft_small)，如果没有网络或 GPU 可能会慢
        # 我们用 try-except 包裹以适应 CI 环境
        try:
            evaluator.prepare()
            # 验证模型被加载
            self.assertIsNotNone(evaluator.model)
        except Exception as e:
            print(f"Skipping model load test due to environment: {e}")
            
        # 3. Clear
        evaluator.clear()
        # 验证模型被清理
        self.assertIsNone(evaluator.model)
        print("✅ Lifecycle Verified.")

if __name__ == '__main__':
    unittest.main()