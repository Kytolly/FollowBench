import os
import pandas as pd
import numpy as np
from datetime import datetime

class Recorder:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.records = [] # List of dicts

    def update(self, video_id, metrics):
        """
        Args:
            video_id (str): 视频 ID
            metrics (dict): { 'FID': 0.1, 'FVD': 100.0, ... }
        """
        row = {'video_id': video_id}
        row.update(metrics)
        self.records.append(row)

    def save_report(self):
        if not self.records:
            print("No records to save.")
            return

        df = pd.DataFrame(self.records)
        
        # 1. 保存明细
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        detail_path = os.path.join(self.output_dir, f"report_detail_{timestamp}.csv")
        df.to_csv(detail_path, index=False)
        
        # 2. 计算统计值 (Mean / Std)
        # 排除非数值列
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        stats = df[numeric_cols].agg(['mean', 'std'])
        
        summary_path = os.path.join(self.output_dir, f"report_summary_{timestamp}.csv")
        stats.to_csv(summary_path)
        
        print("="*40)
        print(f"Evaluation Complete. Report Saved to {self.output_dir}")
        print("Summary:")
        print(stats)
        print("="*40)