import pandas as pd
import numpy as np
import json
import os

class Analyzer:
    def __init__(self, report_paths):
        """
        Args:
            report_paths: list of str, 多个 report.json 的路径
        """
        self.data = {} # {model_name: report_data}
        for p in report_paths:
            if os.path.exists(p):
                rpt = self._load_json(p)
                model_name = rpt.get('meta', {}).get('model_name', 'Unknown')
                self.data[model_name] = rpt

    def _load_json(self, path):
        with open(path, 'r') as f: return json.load(f)

    def get_metric_stats(self):
        """计算每个模型在每个指标上的均值和方差"""
        stats = []
        
        for model, content in self.data.items():
            row = {'Model': model}
            
            for key, val in content.items():
                if key == 'meta': continue
                
                # 如果是 Dict (Case-level metrics)
                if isinstance(val, dict):
                    scores = list(val.values())
                    mean_val = np.mean(scores)
                    std_val = np.std(scores)
                    row[f"{key}_Mean"] = mean_val
                    row[f"{key}_Std"] = std_val
                # 如果是 Scalar (Dataset-level metrics like FVD)
                elif isinstance(val, (int, float)):
                    row[f"{key}_Mean"] = val
                    row[f"{key}_Std"] = 0.0
            
            stats.append(row)
        
        return pd.DataFrame(stats)

    def compare_models(self, baseline_name):
        """计算其他模型相对于 Baseline 的提升/下降百分比"""
        df = self.get_metric_stats()
        if baseline_name not in df['Model'].values:
            print(f"Baseline {baseline_name} not found.")
            return df
            
        base_row = df[df['Model'] == baseline_name].iloc[0]
        
        comparison = df.copy()
        metrics = [c for c in df.columns if c.endswith('_Mean')]
        
        for m in metrics:
            base_val = base_row[m]
            # 避免除以0
            if base_val == 0: base_val = 1e-6
            
            # 计算 Gap %
            comparison[f"{m}_Gap%"] = ((comparison[m] - base_val) / base_val) * 100
            
        return comparison