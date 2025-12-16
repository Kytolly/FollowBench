import unittest
from unittest.mock import MagicMock, patch
import torch
import sys
import os
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.Ego2ExoFollowShot.dimension.ac import AppearanceConsistencyEvaluator
from src.Ego2ExoFollowShot.dimension.bsc import BackgroundSemanticConsistencyEvaluator

class TestFeatureMetrics(unittest.TestCase):
    def setUp(self):
        self.device = 'cpu'
        self.video_id = "vid_feat_test"
        self.shared_cache = {}
        # 构造 Dummy Video: [2 frames, 3, 64, 64]
        self.dummy_gen = torch.rand(2, 3, 64, 64)
        
        # 构造 Dummy Reference Image (Path)
        self.ref_path = "tests/dummy_ref.png"
        # 创建一个临时的 dummy 图片
        Image.new('RGB', (64, 64)).save(self.ref_path)
        
        # Mock Detection Results: 2 frames
        # Frame 0: Box [10, 10, 50, 50]
        # Frame 1: Box [0, 0, 20, 20]
        self.mock_dets = [
            (torch.tensor([10, 10, 50, 50], dtype=torch.float), torch.tensor(0.9)),
            (torch.tensor([0, 0, 20, 20], dtype=torch.float), torch.tensor(0.8))
        ]

    def tearDown(self):
        if os.path.exists(self.ref_path):
            os.remove(self.ref_path)

    @patch('src.Ego2ExoFollowShot.dimension.metric.AppearanceConsistency')
    @patch('utils.pretrain.get_detection_results')
    @patch('utils.pretrain.load_dinov2')
    @patch('utils.image_kit.prepare_ref_embedding')
    def test_ac_flow(self, mock_prep_emb, mock_load_dino, mock_get_det, mock_calc_ac):
        """测试 AC: 模型加载 -> 检测(或缓存) -> 计算"""
        # Setup Mocks
        mock_dinov2 = MagicMock()
        mock_transform = MagicMock()
        mock_load_dino.return_value = (mock_dinov2, mock_transform)
        mock_get_det.return_value = self.mock_dets
        mock_calc_ac.return_value = 0.85
        mock_prep_emb.return_value = torch.randn(1, 768) # Dummy embedding

        evaluator = AppearanceConsistencyEvaluator(self.device)
        evaluator.det = MagicMock() # Mock internal detector

        # 1. 运行 (触发检测)
        print("\n[Test] AC: Running (Cache Miss)...")
        score = evaluator.compute(
            video_gen=self.dummy_gen, 
            video_ego=None, video_gt=None, 
            path_ref=self.ref_path,
            video_id=self.video_id,
            global_cache=self.shared_cache
        )
        
        self.assertEqual(score, 0.85)
        mock_get_det.assert_called_once()
        # 验证缓存写入
        self.assertIn(f"detection_gen_{self.video_id}", self.shared_cache)

        # 2. 再次运行 (命中缓存)
        print("[Test] AC: Running (Cache Hit)...")
        mock_get_det.reset_mock()
        evaluator.compute(
            video_gen=self.dummy_gen, 
            video_ego=None, video_gt=None, 
            path_ref=self.ref_path,
            video_id=self.video_id,
            global_cache=self.shared_cache
        )
        mock_get_det.assert_not_called() # 应该直接用缓存
        print("✅ AC Flow Verified.")

    @patch('src.Ego2ExoFollowShot.dimension.metric.BackgroundSemanticConsistency')
    @patch('utils.pretrain.get_detection_results')
    @patch('transformers.CLIPModel.from_pretrained')
    @patch('transformers.CLIPProcessor.from_pretrained')
    def test_bsc_flow(self, mock_clip_proc, mock_clip_model, mock_get_det, mock_calc_bsc):
        """测试 BSC: 共享缓存检测结果"""
        # Setup Mocks
        mock_get_det.return_value = self.mock_dets
        mock_calc_bsc.return_value = 0.75
        
        evaluator = BackgroundSemanticConsistencyEvaluator(self.device)
        evaluator.det = MagicMock()

        # 预先填充缓存 (模拟 AC 或 CCE 已经跑过)
        self.shared_cache[f"detection_gen_{self.video_id}"] = self.mock_dets

        print("\n[Test] BSC: Running (Cache Hit)...")
        score = evaluator.compute(
            video_gen=self.dummy_gen, 
            video_ego=None, video_gt=None, 
            path_ref=self.ref_path,
            video_id=self.video_id,
            global_cache=self.shared_cache
        )
        
        self.assertEqual(score, 0.75)
        # 验证没有调用检测
        mock_get_det.assert_not_called()
        
        # 验证 metric 函数被正确调用
        args, kwargs = mock_calc_bsc.call_args
        
        # 验证位置参数为空 (因为你使用了 kwargs 调用)
        self.assertEqual(len(args), 0, "Expected zero positional arguments in the call.")
        
        # 验证 detection_results 在 kwargs 中被正确传递
        self.assertEqual(kwargs['detection_results'], self.mock_dets, msg="Detections must be passed correctly via kwargs.")
        print("✅ BSC Flow Verified.")

if __name__ == '__main__':
    unittest.main()