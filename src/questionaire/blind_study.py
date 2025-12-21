import gradio as gr
import json
import random
import os
import time
import uuid
from pathlib import Path
from huggingface_hub import HfApi, upload_file
from filelock import FileLock

from . import ASSETS_DIR

class BlindStudyLoader:
    def __init__(self):
        self.cases = self._scan_cases()
    
    def _scan_cases(self):
        """
        扫描 assets/Test 目录，构建可用于对比的 Case 列表。
        这里假设结构: assets/Test/<case_id>/{ego.mp4, exo.mp4 (GT), ...}
        """
        test_root = ASSETS_DIR / "Test" if (ASSETS_DIR / "Test").exists() else ASSETS_DIR
        valid_cases = []
        
        if test_root.exists():
            for case_dir in test_root.iterdir():
                if case_dir.is_dir():
                    ego_path = case_dir / "ego.mp4"
                    gt_path = case_dir / "exo.mp4"
                    
                    # 寻找待测模型结果 (这里简化逻辑：对比 GT 和生成的某个结果)
                    # 实际项目中，您可能需要从 output/ 目录加载生成的视频
                    # 这里为了演示，假设我们有一个 "EgoGen-V1" 的结果就在 output/ 下或者是同目录的一个文件
                    gen_path = case_dir / "exo.mp4" # <--- 替换为您真实的生成视频路径
                    
                    if ego_path.exists() and gt_path.exists():
                        valid_cases.append({
                            "id": case_dir.name,
                            "input": str(ego_path),
                            "candidates": {
                                "GroundTruth": str(gt_path),
                                "EgoGen-V1": str(gen_path) 
                            }
                        })
        return valid_cases

    def get_blind_pair(self):
        """
        随机抽取一个 Case，并随机交换左右位置 (A/B)
        返回: (ego_path, video_a_path, video_b_path, state_metadata)
        """
        if not self.cases:
            return None, None, None, None
            
        case = random.choice(self.cases)
        
        # 目前只做两两对比：GT vs EgoGen-V1
        model_names = list(case["candidates"].keys())
        if len(model_names) < 2:
            return None, None, None, None
            
        m1, m2 = model_names[0], model_names[1]
        path1, path2 = case["candidates"][m1], case["candidates"][m2]
        
        # 随机洗牌 (Shuffle)
        if random.random() > 0.5:
            # Plan A: Left=m1, Right=m2
            video_a, video_b = path1, path2
            mapping = {"a": m1, "b": m2}
        else:
            # Plan B: Left=m2, Right=m1
            video_a, video_b = path2, path1
            mapping = {"a": m2, "b": m1}
            
        meta = {
            "case_id": case["id"],
            "mapping": mapping, # 记录谁是A，谁是B
            "timestamp": time.time()
        }
        
        return case["input"], video_a, video_b, meta