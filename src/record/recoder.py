import os
import json
import numpy as np
from datetime import datetime

class Ego2ExoRecorder:
    def __init__(self, team_name, model_name, output_dir, modal="fullymodal", mode="easy"):
        self.meta = {
            "team_name": team_name,
            "model_name": model_name,
            "modal": modal,
            "mode": mode,
            "contact": "N/A",
            "timestamp": datetime.now().isoformat()
        }
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 内部存储结构: { MetricName: { CaseID: Score } }
        self.results = {}

    def add_case_score(self, case_id, scores_dict):
        """
        添加单个 Case 的多个指标得分
        scores_dict: {'AestheticQuality': 5.5, 'FVD': 120.3, ...}
        """
        for metric, score in scores_dict.items():
            if metric not in self.results:
                self.results[metric] = {}
            
            # 确保 score 是 Python float 类型 (非 Tensor/Numpy)
            if isinstance(score, (np.float32, np.float64)):
                score = float(score)
            
            self.results[metric][case_id] = score

    def add_dataset_metric(self, metric_name, score):
        """添加数据集级别的指标 (如 FVD)"""
        if metric_name not in self.results:
            self.results[metric_name] = score # 直接存值，而不是 dict
        else:
            self.results[metric_name] = score

    def save_report(self, filename=None):
        if filename is None:
            filename = f"report_{self.meta['model_name']}.json"
        
        save_path = os.path.join(self.output_dir, filename)
        
        final_data = {
            "meta": self.meta,
            **self.results # 解包 metrics
        }
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(final_data, f, indent=4)
        
        print(f"Report saved to {save_path}")
        return save_path

    @staticmethod
    def load_report(path):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data