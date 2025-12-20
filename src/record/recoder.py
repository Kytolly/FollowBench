import os
import json
import numpy as np
from datetime import datetime

class Recorder:
    def __init__(self, meta, output_dir):
        """
        初始化 Recorder
        Args:
            meta (dict): 包含 team_name, model_name等元数据
            output_dir (str): 报告保存目录
        """
        self.meta = meta
        self.record_time = datetime.now().isoformat()
        self.meta["timestamp"] = self.record_time
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 存储结构: { MetricName: { CaseID: Score } } 或 { MetricName: Score }
        self.data = {}

    def update(self, metric_name, results):
        """
        更新特定指标的计算结果。
        
        Args:
            metric_name (str): 指标名称 (e.g., 'AestheticQuality', 'FrechetVideoDistance')
            results (dict | float): 
                - 如果是 dict: {case_id: score, ...} (Case-level metrics)
                - 如果是 float/int: score (Dataset-level metrics like FVD)
        """
        # 1. 处理 Case-level 结果 (Dict)
        if isinstance(results, dict):
            # 数据清洗：将 numpy 类型转为 python原生类型，确保 json 可序列化
            clean_results = {}
            for k, v in results.items():
                if isinstance(v, (np.floating, float)):
                    clean_results[k] = float(v)
                elif isinstance(v, (np.integer, int)):
                    clean_results[k] = int(v)
                else:
                    clean_results[k] = v
            
            # 存入数据
            self.data[metric_name] = clean_results
        
        # 2. 处理 Dataset-level 结果 (Scalar)
        elif isinstance(results, (np.floating, float, np.integer, int)):
            self.data[metric_name] = float(results)
            
        else:
            print(f"[Recorder] Warning: Unexpected result type for {metric_name}: {type(results)}")
            self.data[metric_name] = results

    def save_report(self, filename=None):
        """
        将结果保存为符合 report.json 格式的文件
        """
        if filename == None: filename = f'{self.record_time}_results.json'
        report_path = os.path.join(self.output_dir, filename)
        
        # 构造最终的 JSON 结构: meta 在最顶层，随后是各指标
        final_report = {
            "meta": self.meta,
            **self.data  # 解包指标数据
        }
        
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(final_report, f, indent=4, ensure_ascii=False)
            print(f"Evaluation Report saved to: {report_path}")
        except Exception as e:
            print(f"[Recorder] Error saving report: {e}")
            
        return report_path