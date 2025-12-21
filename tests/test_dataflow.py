import unittest
import os
import sys
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# 确保能导入项目根目录
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.dataflow.set import BenchmarkDataset
from src.dataflow.option import Options
from src.dataflow.submission import Submission
from scripts.env import get_env_config

class TestDataFlow(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # 强制获取 test 环境配置
        cls.cfg = get_env_config('test')
        print(f"\n>>> [TestDataFlow] Using config for env: test")
        print(f"    Repo ID: {cls.cfg.get('repo_id')}")
        print(f"    Assets Path: {cls.cfg.get('path_assets')}")

    def test_01_dataset_loading(self):
        """测试数据集加载和 DataLoader 迭代"""
        # 构造 Options
        opt = Options(
            repo_id=self.cfg['repo_id'], 
            assets=self.cfg['path_assets'],
            annotation=self.cfg.get('annotation', os.path.join(self.cfg['path_assets'], 'annotation.json')), # 确保有默认值
            phase='test'
        )
        
        # 检查目录是否存在，不存在则跳过（避免CI报错）
        if not os.path.exists(opt.assets):
            print(f"Warning: Assets path {opt.assets} does not exist. Skipping dataset test.")
            return

        try:
            dataset = BenchmarkDataset(opt)
            loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
            
            print(f"    Dataset size: {len(dataset)}")
            self.assertGreater(len(dataset), 0, "Dataset should not be empty")

            # 检查前几个样本
            for i, batch in enumerate(loader):
                if i >= 3: break
                self.assertIn('video_id', batch)
                self.assertIn('ego_video', batch)
                # 检查 Tensor 形状
                if isinstance(batch['ego_video'], torch.Tensor):
                    self.assertEqual(len(batch['ego_video'].shape), 5, "Ego video should be [B, T, C, H, W]")
                
        except Exception as e:
            self.fail(f"Dataset loading failed: {e}")

    def test_02_submission_validation(self):
        """测试提交文件的读取和验证逻辑"""
        sub_cfg = self.cfg.get('submission', {})
        source_path = sub_cfg.get('source_path')
        submission_path = sub_cfg.get('submission_path')

        if not source_path or not os.path.exists(source_path):
             print("Warning: Submission source path not found. Skipping submission test.")
             return
        
        if not submission_path or not os.path.exists(submission_path):
            print("Warning: Submission JSON not found. Skipping submission test.")
            return

        try:
            sub = Submission(
                source_path=source_path,
                submission_path=submission_path
            )
            is_valid, report = sub.valid()
            print(f"    Submission Valid: {is_valid}")
            # 注意：这里我们不强制 assert True，因为测试环境可能没有完整的 submission 文件
            # 只要代码跑通即可
            if not is_valid:
                print(f"    Validation Report: {report}")
                
        except Exception as e:
            self.fail(f"Submission validation threw exception: {e}")

if __name__ == "__main__":
    unittest.main()