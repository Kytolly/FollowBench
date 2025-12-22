import os
import random
from pathlib import Path
import logging
logger = logging.getLogger()

class DatasetLoader:
    def __init__(self, assets_dir: str):
        self.assets_dir = Path(assets_dir)
        self.test_root = self.assets_dir / "test"
        self.cases = []
        self._scan()

    def _scan(self):
        """扫描目录结构，建立可用案例索引"""
        if not self.test_root.exists():
            logger.warning(f"Dataset root not found: {self.test_root}")
            return

        valid_cases = []
        for case_dir in self.test_root.iterdir():
            if not case_dir.is_dir():
                continue
            
            # 必须包含的基础文件
            ego_path = case_dir / "ego.mp4"
            if not ego_path.exists():
                continue

            # 寻找该 Case 下可用的模型结果
            # 这里定义你的模型文件命名规则
            # 假设结构: assets/Test/case_001/exo.mp4 (GT), output/gen_video.mp4 (Model)
            # 为了演示，我们假设在 case 目录下有不同后缀或文件夹来区分模型
            
            # 示例：直接在 case 目录下找对比项 (实际项目中可能需要从 output 目录 join)
            candidates = {}
            
            # 1. Ground Truth
            gt_path = case_dir / "exo.mp4"
            if gt_path.exists():
                candidates["GroundTruth"] = str(gt_path)

            # 2. 假设生成的视频在 case_dir / "ours.mp4" (仅作示例，请根据实际情况修改)
            # 或者你可以传入 output_dir 到 __init__ 来扫描生成结果
            ours_path = case_dir / "ours.mp4" 
            if ours_path.exists():
                 candidates["Ours"] = str(ours_path)
            elif gt_path.exists(): 
                # 如果没有生成结果，为了测试 UI，暂时用 GT 复制一个假装是 Ours
                candidates["Ours"] = str(gt_path) 

            # 至少要有两个候选才能做 A/B 测试
            if len(candidates) >= 2:
                valid_cases.append({
                    "id": case_dir.name,
                    "input": str(ego_path),
                    "candidates": candidates
                })
        
        self.cases = valid_cases
        logger.info(f"Loaded {len(self.cases)} valid cases for user study.")

    def get_random_case(self):
        """随机返回一个 Case 对象"""
        if not self.cases:
            return None
        return random.choice(self.cases)