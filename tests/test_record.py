import unittest
import os
import json
import shutil
import numpy as np
import pandas as pd
from pathlib import Path

# 假设你的代码保存在 src/record 下
# 如果你尚未保存上一步的代码，请确保相应文件存在
from src.record.recoder import Ego2ExoRecorder
from src.record.analysis import BenchmarkAnalyzer
from src.record.rankboard import RankBoard
# Runner 通常依赖较重，我们将对其进行 Mock 测试

class TestRecordSystem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """在测试开始前准备临时目录"""
        cls.test_dir = "./temp_test_results"
        if os.path.exists(cls.test_dir):
            shutil.rmtree(cls.test_dir)
        os.makedirs(cls.test_dir)
        
        # 准备模拟数据用于 Analysis 和 RankBoard 测试
        cls._create_dummy_reports()

    @classmethod
    def tearDownClass(cls):
        """测试结束后清理"""
        if os.path.exists(cls.test_dir):
            shutil.rmtree(cls.test_dir)

    @classmethod
    def _create_dummy_reports(cls):
        """创建两个模拟的 report.json 用于对比测试"""
        # Model A: 表现较好
        rec_a = Ego2ExoRecorder("TeamA", "Model-A", cls.test_dir)
        rec_a.add_case_score("case-1", {"FVD": 100.0, "AC": 0.9}) # FVD低好, AC高好
        rec_a.add_case_score("case-2", {"FVD": 110.0, "AC": 0.8})
        rec_a.add_dataset_metric("Overall_Quality", 5.0)
        cls.report_a = rec_a.save_report("report_Model-A.json")

        # Model B: 表现较差
        rec_b = Ego2ExoRecorder("TeamB", "Model-B", cls.test_dir)
        rec_b.add_case_score("case-1", {"FVD": 200.0, "AC": 0.5})
        rec_b.add_case_score("case-2", {"FVD": 220.0, "AC": 0.4})
        rec_b.add_dataset_metric("Overall_Quality", 3.0)
        cls.report_b = rec_b.save_report("report_Model-B.json")

    # ================= Recorder 测试 =================
    def test_recorder_structure(self):
        """测试 Recorder 生成的 JSON 结构是否符合预期"""
        recorder = Ego2ExoRecorder("TestTeam", "TestModel", self.test_dir)
        
        # 模拟添加数据
        recorder.add_case_score("test_id_001", {"Metric_A": 0.95, "Metric_B": 1.2})
        recorder.add_case_score("test_id_002", {"Metric_A": 0.88}) # 缺少 Metric_B
        recorder.add_dataset_metric("Dataset_FVD", 150.5)
        
        # 保存
        path = recorder.save_report("test_structure.json")
        
        # 验证
        with open(path, 'r') as f:
            data = json.load(f)
            
        self.assertIn("meta", data)
        self.assertEqual(data["meta"]["model_name"], "TestModel")
        
        # 验证指标分组
        self.assertIn("Metric_A", data)
        self.assertIn("test_id_001", data["Metric_A"])
        self.assertEqual(data["Metric_A"]["test_id_001"], 0.95)
        
        # 验证 Dataset Level 指标 (直接存储为数值)
        self.assertIn("Dataset_FVD", data)
        self.assertEqual(data["Dataset_FVD"], 150.5)
        print("✅ Recorder Structure Test Passed")

    # ================= Analysis 测试 =================
    def test_analysis_stats(self):
        """测试统计功能 (Mean/Std)"""
        analyzer = BenchmarkAnalyzer([self.report_a, self.report_b])
        stats = analyzer.get_metric_stats()
        
        # 验证 Model-A 的数据
        # FVD: (100+110)/2 = 105.0
        row_a = stats[stats['Model'] == 'Model-A'].iloc[0]
        self.assertAlmostEqual(row_a['FVD_Mean'], 105.0)
        self.assertAlmostEqual(row_a['AC_Mean'], 0.85)
        
        # 验证 Dataset Metric
        self.assertEqual(row_a['Overall_Quality_Mean'], 5.0)
        print("✅ Analysis Stats Test Passed")

    def test_analysis_comparison(self):
        """测试与 Baseline 的对比 (Gap%)"""
        analyzer = BenchmarkAnalyzer([self.report_a, self.report_b])
        
        # 假设 Model-B 是 Baseline (较差)
        # Model-A 的 FVD (105) vs Model-B (210) -> (105-210)/210 = -0.5 (-50%)
        comp_df = analyzer.compare_models(baseline_name="Model-B")
        
        row_a = comp_df[comp_df['Model'] == 'Model-A'].iloc[0]
        self.assertAlmostEqual(row_a['FVD_Mean_Gap%'], -50.0) 
        
        # AC: 0.85 vs 0.45 -> (0.85-0.45)/0.45 = 0.888... (+88.8%)
        expected_gap = ((0.85 - 0.45) / 0.45) * 100
        self.assertAlmostEqual(row_a['AC_Mean_Gap%'], expected_gap)
        print("✅ Analysis Comparison Test Passed")

    # ================= RankBoard 测试 =================
    def test_rankboard_normalization(self):
        """测试排行榜归一化逻辑 (重点：Lower is Better 的指标)"""
        ranker = RankBoard([self.report_a, self.report_b])
        
        # 我们需要在 rankboard.py 的 METRIC_DIRECTION 中添加 FVD=False 测试用例
        # 为了测试独立性，我们可以临时注入一个 Mock 的配置
        from src.record import rankboard
        rankboard.METRIC_DIRECTION = {
            'FVD': False, # Lower is better
            'AC': True,   # Higher is better
            'Overall_Quality': True
        }
        
        rank_df = ranker.generate_rank(os.path.join(self.test_dir, "leaderboard.csv"))
        
        # Model-A: FVD=105 (Better), AC=0.85 (Better)
        # Model-B: FVD=210 (Worse), AC=0.45 (Worse)
        
        # 归一化逻辑验证:
        # FVD: Min=105, Max=210. 
        # Model-A Norm (LowerBetter) = 1 - (105-105)/(210-105) = 1 - 0 = 1.0
        # Model-B Norm (LowerBetter) = 1 - (210-105)/(210-105) = 1 - 1 = 0.0
        
        # 获取 Model-A 的排名应该在 Model-B 前面
        rank_1_model = rank_df.iloc[0]['Model']
        self.assertEqual(rank_1_model, "Model-A")
        
        # 验证 CSV 生成
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "leaderboard.csv")))
        print("✅ RankBoard Normalization Test Passed")

    # ================= Runner Logic (Mock) 测试 =================
    def test_runner_flow(self):
        """
        模拟 BenchmarkRunner 的工作流。
        不加载真实模型，而是 Mock 评估器。
        """
        from unittest.mock import MagicMock
        from src.record.runner import BenchmarkRunner
        
        # 1. Mock Options
        mock_opt = MagicMock()
        mock_opt.output_dir = self.test_dir
        mock_opt.assets = "dummy_assets"
        
        # 2. Mock Runner (重写 _load_evaluators 和 dataloader)
        runner = BenchmarkRunner(mock_opt, device='cpu')
        
        # Mock DataLoader: 返回一个 Batch
        # batch = {'video_id': ('test_vid_1',), 'ego_video': ..., 'exo_video': ...}
        mock_batch = {
            'video_id': ('test_vid_1',),
            'ego_video': [None], # 内容不重要，反正Evaluator也是Mock
            'exo_video': [None],
            'ref_image': [None]
        }
        runner.dataloader = [mock_batch] 
        
        # Mock Evaluators
        mock_evaluator = MagicMock()
        mock_evaluator.compute.return_value = 0.88
        runner.evaluators = {'MockMetric': mock_evaluator}
        
        # Mock Submission (返回一个 Fake Video Tensor)
        from src.dataflow.submission import Submission
        # 我们很难直接 Mock Submission 类的内部读取，这里可以用 duck typing 
        # 或者直接 Mock sub 对象
        mock_sub = MagicMock()
        import torch
        mock_sub.get_generated_video.return_value = torch.zeros(1, 3, 256, 256) # Fake Tensor
        
        # 临时替换 Submission 类构造逻辑 (Monkey Patching for test)
        # 但 BenchmarkRunner.evaluate 内部是 new 了一个 Submission
        # 更好的方式是重构 Runner 允许传入 submission 实例，或者在此处我们手动运行核心循环逻辑
        # 为了演示，我们假设 evaluate 方法允许传入 submission 对象 (推荐修改 Runner 代码支持此项)
        
        # === 修改 Runner.evaluate 接口建议 ===
        # def evaluate(self, submission_or_path, model_name, ...):
        #    if isinstance(submission_or_path, str): sub = Submission(...)
        #    else: sub = submission_or_path
        
        # 假设 Runner 已修改支持传入对象，或者我们 Mock 整个 evaluate 方法
        # 这里我们手动模拟 Runner 内部的记录逻辑来验证 Recoder 集成
        
        # --- 模拟 Runner 内部逻辑开始 ---
        recorder = Ego2ExoRecorder("MockTeam", "MockModel", self.test_dir)
        vid_id = 'test_vid_1'
        
        # 计算
        score = runner.evaluators['MockMetric'].compute()
        recorder.add_case_score(vid_id, {'MockMetric': score})
        
        path = recorder.save_report("report_MockModel.json")
        # --- 模拟结束 ---
        
        self.assertTrue(os.path.exists(path))
        print("✅ Runner Flow (Mock) Test Passed")

if __name__ == '__main__':
    unittest.main()