import unittest
import os
import json
import shutil
import torch
import glob
from pathlib import Path

# 导入项目模块
# 假设你已经按照之前的建议重构了 src/__init__.py (Bench) 和 src/record/recoder.py
from src import Bench
from src.dataflow.submission import Submission
from configs import CONFIG

class TestLocalIntegration(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """
        环境准备：读取配置，构建临时的 annotation 和 submission 文件
        """
        # 1. 读取配置
        cls.cfg = CONFIG
        cls.assets_root = Path(cls.cfg['path_assets'])
        cls.sub_json_path = Path(cls.cfg['submission']['submission_path'])
        cls.sub_source_path = Path(cls.cfg['submission']['source_path'])
        cls.output_dir = Path("./results/integration_test")
        
        # 确保输出目录存在
        os.makedirs(cls.output_dir, exist_ok=True)
        # 确保 submission 目录存在
        os.makedirs(cls.sub_json_path.parent, exist_ok=True)

        print(f"\n[Setup] Assets Root: {cls.assets_root}")
        
        # 2. 扫描真实视频文件
        # 假设 assets/Test 下直接存放了 mp4 文件
        # 如果你的目录结构不同 (e.g. assets/Test/Third_Video/*.mp4)，请修改这里的 glob
        video_files = list(cls.assets_root.glob("*.mp4"))
        
        if not video_files:
            # 如果没有文件，为了防止测试报错，生成一个 dummy 视频
            print("[Warning] No mp4 found in assets/Test, generating dummy video...")
            dummy_path = cls.assets_root / "test_dummy.mp4"
            cls._create_dummy_video(dummy_path)
            video_files = [dummy_path]
        
        print(f"[Setup] Found {len(video_files)} videos for testing.")

        # 3. 构建 annotation.json (用于 DataLoader)
        # 我们将同一个视频同时作为 Ego 和 Exo (GT) 进行自测
        cls.anno_data = {}
        cls.sub_results = {}
        
        for idx, vpath in enumerate(video_files):
            case_id = f"test-case-{idx}"
            rel_path = vpath.name # 相对于 assets_root 的路径
            
            # 构造 Annotation (Loader 使用)
            cls.anno_data[case_id] = {
                "the first view": rel_path, # Ego
                "the third view": rel_path, # GT (Exo)
                # 如果没有 Ref 图片，暂时复用视频文件路径 (Loader 可能会报错，最好有一个真实的 png)
                # 这里为了演示，假设有一个 ref.png，或者 Loader 有容错
                "reference": rel_path.replace(".mp4", ".png") if (vpath.with_suffix('.png').exists()) else rel_path, 
                "prompt": {
                    "positive": {"for fullymodal model": "test prompt"},
                    "negative": "test negative"
                }
            }
            
            # 构造 Submission Mapping (Gen 使用)
            # Submission 类会用 source_path / rel_path 来找文件
            cls.sub_results[case_id] = rel_path 

        # 保存临时 annotation.json
        cls.temp_anno_path = cls.assets_root / "temp_annotation_test.json"
        with open(cls.temp_anno_path, 'w') as f:
            json.dump(cls.anno_data, f, indent=4)
            
        # 4. 构建 submission.json
        cls.sub_meta = {
            "team_name": "IntegrationTest",
            "model_name": "GroundTruth-SelfTest",
            "modal": "fullymodal",
            "mode": "easy",
            "timestamp": "2025-01-01"
        }
        
        cls.submission_content = {
            "meta": cls.sub_meta,
            "results": cls.sub_results
        }
        
        with open(cls.sub_json_path, 'w') as f:
            json.dump(cls.submission_content, f, indent=4)
            
        print("[Setup] Configuration files generated.")

    @staticmethod
    def _create_dummy_video(path):
        import cv2
        import numpy as np
        height, width = 256, 256
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(str(path), fourcc, 10, (width, height))
        for _ in range(10):
            frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
            video.write(frame)
        video.release()

    def test_full_pipeline(self):
        """
        运行完整 Pipeline: Loader -> Bench -> Recorder
        """
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"\n[Test] Running pipeline on {device}...")
        
        # 1. 初始化 Bench
        # 注意：这里我们传入 assets_root，Bench 内部会去读 annotation
        # 为了让 Bench 读到我们刚生成的 temp_annotation_test.json，
        # 我们需要在 Bench 初始化后手动替换 Options 或者 确保 Bench 逻辑能找到它。
        # 根据之前的重构，Bench 内部会初始化 Options。
        # *关键*: 我们需要在 Bench 外部或修改 Bench 使其能接受 annotation 路径，
        # 或者我们暂时 overwrite 默认的 annotation.json。
        
        # 这里为了稳健，假设 Bench 代码允许自动查找或我们在 env 中没有硬编码 annotation 文件名。
        # 如果 src/__init__.py 中硬编码了 'annotation.json'，我们可能需要 rename 临时文件。
        
        real_anno_path = self.assets_root / "annotation.json"
        backup_anno_path = self.assets_root / "annotation.json.bak"
        
        # 备份原有 annotation (如果存在) 并替换
        if real_anno_path.exists():
            shutil.move(real_anno_path, backup_anno_path)
        shutil.copy(self.temp_anno_path, real_anno_path)
        
        try:
            # 实例化 Bench
            bench = Bench(device=device, assets_root=str(self.assets_root))
            
            # 运行评估
            # 仅测试几个快速指标，避免跑太久
            test_metrics = ['TemporalFlickering', 'MotionSmoothness', 'CameraCenteringError'] 
            
            report_path = bench.evaluate(
                source_path=CONFIG['path_assets'],
                submission_path=str(self.sub_json_path),
                output_dir=str(self.output_dir),
                metrics_list=test_metrics,
                batch_size=1
            )
            
            # 验证输出
            self.assertTrue(os.path.exists(report_path), "Report file should exist")
            
            with open(report_path, 'r') as f:
                report = json.load(f)
            
            # 验证 Recorder 格式 (检查 meta 和 指标)
            self.assertIn("meta", report)
            self.assertEqual(report["meta"]["model_name"], "GroundTruth-SelfTest")
            self.assertIn("TemporalFlickering", report)
            
            # 验证数值 (自测 GT，MotionSmoothness 应该很小但非0，如果是 FVD 则为 0)
            print("\n[Result] Report Content Sample:")
            print(json.dumps(report, indent=2)[:500] + "...")
            
        finally:
            # 恢复环境
            if real_anno_path.exists():
                os.remove(real_anno_path)
            if backup_anno_path.exists():
                shutil.move(backup_anno_path, real_anno_path)

    @classmethod
    def tearDownClass(cls):
        # 清理临时文件
        if hasattr(cls, 'temp_anno_path') and cls.temp_anno_path.exists():
            os.remove(cls.temp_anno_path)
        # 清理生成的 submission
        # if cls.sub_json_path.exists():
        #     os.remove(cls.sub_json_path)
        print("\n[Teardown] Cleaned up temporary files.")

if __name__ == '__main__':
    unittest.main()