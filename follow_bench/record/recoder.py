"""Recording utilities for evaluation results.

This module defines :class:`Recorder` which accumulates metric results and
serializes a final report JSON file.
"""

import os
import json
import numpy as np
from datetime import datetime
import logging
from typing import Any, Dict, Union
logger = logging.getLogger()

from ..configs import MetaConfig

class Recorder:
    """Collect and save evaluation results.

    Args:
        meta: Metadata dictionary (must include team_name, model_name, etc.).
        output_dir: Directory where reports will be saved.
    """
    def __init__(self, meta: MetaConfig, output_dir: str) -> None:
        self.meta: MetaConfig = meta
        self.record_time: str = meta.record_time
        self.output_dir: str = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 存储结构: { MetricName: { CaseID: Score } } 或 { MetricName: Score }
        self.data: Dict[str, Union[Dict[str, float], float, Any]] = {}

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
            logger.warning(f"Unexpected result type for {metric_name}: {type(results)}")
            self.data[metric_name] = results

    def save_report(self, filename: str | None = None) -> str:
        """Save the aggregated results to a JSON report file.

        Args:
            filename: Optional filename. If omitted a timestamped name will be used.

        Returns:
            The full path to the saved report file.
        """
        if filename is None:
            now = datetime.now()
            timestamp_safe = now.strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"{timestamp_safe}_results.json"
        else:
            filename = filename.replace(':', '-')
        report_path = os.path.join(self.output_dir, filename)
        
        # 构造最终的 JSON 结构: meta 在最顶层，随后是各指标
        final_report = {
            "meta": dict(self.meta),
            **self.data  # 解包指标数据
        }
        
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(final_report, f, indent=4, ensure_ascii=False)
            logger.info(f"Evaluation Report saved to: {report_path}")
        except Exception as e:
            logger.error(f"Error saving report: {e}")
            
        return report_path