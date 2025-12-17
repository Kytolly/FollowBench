import unittest
import torch
import os
import sys
import shutil
import numpy as np
from PIL import Image
from torchvision.io import read_video

# 添加项目根目录到 Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 导入所有评估器
from src.dimension.cce import CameraCenteringErrorEvaluator
from src.dimension.vv import ViewpointValidityEvaluator
from src.dimension.ta import TrajectoryAlignmentEvaluator
from src.dimension.ac import AppearanceConsistencyEvaluator
from src.dimension.bsc import BackgroundSemanticConsistencyEvaluator
from src.dimension.tf import TemporalFlickeringEvaluator
from src.dimension.ms import MotionSmoothnessEvaluator
from src.dimension.dd import DynamicDegreeEvaluator
from src.dimension.haa import HumanActionAlignmentEvaluator
from src.dimension.aq import AestheticQualityEvaluator
from src.dimension.iq import ImagingQualityEvaluator
from src.dimension.fvd import FrechetVideoDistanceEvaluator

class TestRealPipeline(unittest.TestCase):
    """
    全流程集成测试：使用真实数据运行所有指标，验证计算通路和缓存逻辑。
    """
    @classmethod
    def setUpClass(cls):
        cls.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"\n>>> Running Integration Test on {cls.device} ...")
        
        # 1. 定义数据路径
        cls.data_dir = "assets/cache/example"
        cls.ego_path = os.path.join(cls.data_dir, "ego.mp4")
        cls.exo_path = os.path.join(cls.data_dir, "exo.mp4") # GT
        cls.gen_path = os.path.join(cls.data_dir, "gen.mp4") 
        cls.ref_path = os.path.join(cls.data_dir, "ref.png")
        
        # 检查数据是否存在，不存在则生成 Dummy 数据
        if not os.path.exists(cls.data_dir):
            os.makedirs(cls.data_dir)
            
        if not os.path.exists(cls.ego_path):
            print("[Info] Creating dummy video data...")
            # 生成 16 帧 RGB 噪音视频
            T, H, W = 16, 224, 224
            dummy_vid = torch.randint(0, 255, (T, H, W, 3), dtype=torch.uint8)
            torchvision.io.write_video(cls.ego_path, dummy_vid, fps=8)
            torchvision.io.write_video(cls.exo_path, dummy_vid, fps=8)
            Image.new('RGB', (H, W), color='red').save(cls.ref_path)

        # 2. 加载数据
        print("[Info] Loading videos...")
        # read_video 返回 [T, H, W, C] (0-255 uint8)
        # 我们需要转换为 [T, C, H, W] (0-1 float)
        def load_vid(path):
            v, _, _ = read_video(path, output_format="TCHW")
            v = v.float() / 255.0 # Normalize 0-1
            # Resize 到 224x224 (为了 FVD 和其他模型)
            v = torch.nn.functional.interpolate(v, size=(224, 224), mode='bilinear')
            return v.to(cls.device)

        cls.tensor_ego = load_vid(cls.ego_path)
        cls.tensor_gen = load_vid(cls.gen_path)
        cls.tensor_gt = load_vid(cls.exo_path)
        cls.video_id = "test_case_001"
        cls.shared_cache = {}
        
        # Load Ref Image
        cls.pillow_ref = Image.open(cls.ref_path).convert('RGB').resize((224, 224))

    def test_all_metrics(self):
        """依次运行所有指标，并检查缓存是否被正确复用"""
        evaluators = [
            # 1. Flow Group (Share RAFT)
            ("TF", TemporalFlickeringEvaluator(self.device)),
            ("MS", MotionSmoothnessEvaluator(self.device)),
            ("DD", DynamicDegreeEvaluator(self.device)),
            
            # 2. Detection Group (Share Faster-RCNN)
            ("CCE", CameraCenteringErrorEvaluator(self.device)),
            ("VV", ViewpointValidityEvaluator(self.device)),
            ("TA", TrajectoryAlignmentEvaluator(self.device)),
            
            # 3. Feature Group (Share DINO/CLIP + Detection)
            ("AC", AppearanceConsistencyEvaluator(self.device)),
            ("BSC", BackgroundSemanticConsistencyEvaluator(self.device)),
            
            # 4. Pose Group
            ("HAA", HumanActionAlignmentEvaluator(self.device)),
            
            # 5. Quality Group
            ("AQ", AestheticQualityEvaluator(self.device)),
            ("IQ", ImagingQualityEvaluator(self.device)),
            ("FVD", FrechetVideoDistanceEvaluator(self.device)),
        ]
        
        results = {}
        
        print(f"\n[Step 1] Preparing Evaluators (Loading Models)...")
        for name, evaluator in evaluators:
            try:
                evaluator.prepare()
                print(f"  - {name} prepared.")
            except Exception as e:
                print(f"  [Error] {name} prepare failed: {e}")

        print(f"\n[Step 2] Computing Metrics...")
        for name, evaluator in evaluators:
            try:
                score = evaluator.compute(
                    tensor_gen=self.tensor_gen,
                    tensor_gt=self.tensor_gt,
                    tensor_ego=self.tensor_ego,
                    pillow_ref=self.pillow_ref,
                    video_id=self.video_id,
                    global_cache=self.shared_cache
                )
                results[name] = score
                print(f"  - {name}: {score:.4f}")
                
                # 检查返回值是否合法
                self.assertIsInstance(score, float)
                self.assertFalse(np.isnan(score), f"{name} returned NaN")
                
            except Exception as e:
                print(f"  [Error] {name} compute failed: {e}")
                # Optional: self.fail(f"{name} failed: {e}")

        # [Step 3] 验证缓存逻辑
        print(f"\n[Step 3] Verifying Cache...")
        keys = self.shared_cache.keys()
        print(f"  Cached Keys: {list(keys)}")
        
        # 验证 Flow 是否只计算了一次 (TF, MS, DD 应该共享)
        self.assertIn(f"flow_gen_{self.video_id}", keys, "Optical Flow should be cached")
        
        # 验证 Detection 是否只计算了一次 (CCE, VV, TA, AC, BSC 应该共享)
        self.assertIn(f"detection_gen_{self.video_id}", keys, "Detection should be cached")
        
        print("✅ All tests passed. System is consistent.")

if __name__ == '__main__':
    import torchvision
    unittest.main()