import os
from pathlib import Path
from dataclasses import dataclass, field

from dotenv import load_dotenv
import yaml
import dataclasses
from dataclasses import dataclass, field, asdict
from typing import List
import datetime

# 加载环境变量
load_dotenv()
DEBUG = os.getenv("DEBUG", "False").lower() == "true"

# -----------------------------------------------------------------------------
# 0. 路径定义 (核心修正)
# -----------------------------------------------------------------------------
# __file__ = .../ego2exo_bench/configs/__init__.py
_CONFIGS_DIR = Path(__file__).resolve().parent      # .../ego2exo_bench/configs
_PACKAGE_DIR = _CONFIGS_DIR.parent                  # .../ego2exo_bench
_PROJECT_ROOT = _PACKAGE_DIR.parent                 # .../ (Repo Root)

# 导出给其他模块使用
globals()["PROJECT_ROOT"] = str(_PROJECT_ROOT)
globals()["PACKAGE_DIR"] = str(_PACKAGE_DIR)
globals()["CONFIG_DIR"] = str(_CONFIGS_DIR)

# -----------------------------------------------------------------------------
# 1. 定义配置 Schema
# -----------------------------------------------------------------------------
@dataclass
class ModelsConfig:
    wham_vit_h: str = "models/wham_vit_h.pth"
    yolo: str = "models/yolov8x.pt"
    
@dataclass
class RulesConfig:
    num_frames: int = 149
    fps: int = 60
    height: int = 704
    width: int = 1280

@dataclass
class SubmissionConfig:
    json_path: str = "cache/gen/submission.json"
    source_path: str = "assets/test"
    # total_cases_num: int = 120
    # resolution_height: int = 240
    # resolution_width: int = 426
    # clip_fps: int = 60
    # clip_len: int = 300

@dataclass
class AssetsConfig:
    path: str = "assets/followbench"

@dataclass
class OutputConfig:
    path: str = "output/"

@dataclass
class MetaConfig:
    team_name: str = "A team name",
    model_name: str = "A model name",
    split: str = "test_unseen",
    contact: str = "email@example.com"
    record_time: str = datetime.datetime.now().isoformat()
    
@dataclass
class BaseEnvConfig:
    """环境配置基类"""
    repo_id: str = "Kytolly/FollowBench"
    device: str = "cuda"
    metrics: List[str] = field(default_factory=lambda: ['mse', 'ssim', 'lpips'])
    assets: AssetsConfig = field(default_factory=AssetsConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    rules: RulesConfig = field(default_factory=RulesConfig)
    submission: SubmissionConfig = field(default_factory=SubmissionConfig)
    meta: MetaConfig = field(default_factory=MetaConfig) 
    
    def update_from_dict(self, data: dict, target_obj=None):
        """
        递归更新嵌套配置 (Level 3: 命令行字典覆盖)
        """
        if target_obj is None:
            target_obj = self
            
        if not data or not isinstance(data, dict):
            return

        for key, value in data.items():
            if value is None:
                continue
            if hasattr(target_obj, key):
                attr = getattr(target_obj, key)
                if dataclasses.is_dataclass(attr) and isinstance(value, dict):
                    self.update_from_dict(value, target_obj=attr)
                else:
                    # 如果是普通属性，直接覆盖
                    setattr(target_obj, key, value)

    def update_from_yaml(self, yaml_path: str):
        """
        从 YAML 文件更新配置 (Level 2: 配置文件覆盖)
        """
        if yaml_path and os.path.exists(yaml_path):
            with open(yaml_path, 'r', encoding='utf-8') as f:
                yaml_data = yaml.safe_load(f) or {}
                self.update_from_dict(yaml_data)
        elif yaml_path:
            print(f"⚠️ 警告: 未找到配置文件 {yaml_path}，将使用内置系统默认配置。")

    def __str__(self):
        """优美地打印当前生效的所有层级配置"""
        return yaml.dump(asdict(self), default_flow_style=False, sort_keys=False)