import os
from pathlib import Path
from typing import List, Union
from dataclasses import dataclass, field

from dotenv import load_dotenv
from omegaconf import OmegaConf, DictConfig

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
    wham_vit_h: int = "assets/models/wham_vit_h.pth"
    
@dataclass
class RulesConfig:
    clip_len: int = 300
    fps: int = 60
    resolution_height: int = 240
    resolution_width: int = 426

@dataclass
class SubmissionConfig:
    submission_path: str = "cache/gen/submission.json"
    source_path: str = "assets/test"
    total_cases_num: int = 120
    resolution_height: int = 240
    resolution_width: int = 426
    clip_fps: int = 60
    clip_len: int = 300

@dataclass
class BaseEnvConfig:
    """环境配置基类"""
    repo_id: str = "Kytolly/examples_Ego2ExoFollowCamera"
    repo_feeaback_id: str = "Kytolly/feedback_Ego2ExoFollowCamera"
    repo_submission_id: str = "Kytolly/submission_Ego2ExoFollowCamera"
    
    # 这里的默认路径现在会被解析为 PROJECT_ROOT/assets/
    path_assets: str = "assets/"
    
    models: ModelsConfig = field(default_factory=ModelsConfig)
    rules: RulesConfig = field(default_factory=RulesConfig)
    submission: SubmissionConfig = field(default_factory=SubmissionConfig)

# -----------------------------------------------------------------------------
# 2. 配置加载逻辑
# -----------------------------------------------------------------------------

def load_config(config_file_name: str = "config.yml"):
    """
    加载配置并将相对路径解析为相对于 PROJECT_ROOT 的绝对路径。
    """
    env = os.getenv("APP_ENV", "dev").lower()
    
    # --- A. 寻找配置文件 ---
    config_path = os.getenv("CONFIG_PATH", config_file_name)
    path_to_use = None
    p = Path(config_path)

    # 1) 绝对路径
    if p.is_absolute():
        if p.exists(): path_to_use = p
    else:
        # 查找顺序: 
        # 1. ego2exo_bench/configs/ (Package Configs)
        # 2. Project Root (Repo Root)
        # 3. CWD
        candidates = [
            _CONFIGS_DIR / config_path,
            _PROJECT_ROOT / config_path,
            Path.cwd() / config_path
        ]
        
        for candidate in candidates:
            if candidate.exists():
                path_to_use = candidate
                break

    if path_to_use is None:
        # 找不到时，为了鲁棒性，默认回退到 Package Configs 下的路径（即使不存在也指向那里，方便报错信息明确）
        path_to_use = _CONFIGS_DIR / config_file_name
        if not path_to_use.exists():
             print(f"Warning: Config file '{config_file_name}' not found. Using defaults.")

    # --- B. 加载 YAML 并 Merge Schema ---
    if path_to_use and path_to_use.exists():
        yaml_cfg = OmegaConf.load(path_to_use)
    else:
        yaml_cfg = OmegaConf.create({}) # 空配置

    # 提取特定环境配置
    if env in yaml_cfg:
        env_yaml_cfg = yaml_cfg[env]
    else:
        # 如果 yaml 没定义该环境，使用空字典，完全依赖 Schema 默认值
        env_yaml_cfg = OmegaConf.create({}) 
    
    # 创建 Schema 实例 (默认值)
    schema_cfg = OmegaConf.structured(BaseEnvConfig)
    
    # 合并: Schema Defaults + User YAML
    cfg = OmegaConf.merge(schema_cfg, env_yaml_cfg)

    # --- C. 路径规范化逻辑 (相对于 PROJECT_ROOT) ---
    def _is_path_like(key: str, val: str):
        """启发式判断是否是路径字段"""
        if not isinstance(val, str): return False
        if Path(val).is_absolute(): return False # 已经是绝对路径忽略
        
        key = (key or "").lower()
        keywords = ["path", "dir", "file", "repo", "asset", "source", "submission", "config", "data", "root", "output"]
        if any(k in key for k in keywords): return True
        if ("/" in val) or ("\\" in val) or val.startswith("."): return True
        return False

    def _normalize(obj: object, parent_key: str = "") -> object:
        if isinstance(obj, dict):
            return {k: _normalize(v, k) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return type(obj)([_normalize(x, parent_key) for x in obj])
        
        if isinstance(obj, str) and _is_path_like(parent_key, obj):
            # [核心逻辑] 将相对路径解析为基于 PROJECT_ROOT 的绝对路径
            # 例如: "assets/test" -> "/abs/path/to/project/assets/test"
            # resolve() 会消除 .. 但不会检查文件是否存在 (对于生成路径很有用)
            try:
                # 注意：如果 obj 是类似 "cache/gen/..." 这种新建路径，
                # (root / obj).resolve() 在某些 OS/Python版本如果不存可能会报错，
                # 但通常 pathlib.Path.resolve() 在非 strict 模式下（Python 3.10+默认）是安全的。
                # 为了兼容性，这里我们手动拼接并 abspath
                abs_path = (_PROJECT_ROOT / obj).resolve()
                return str(abs_path)
            except Exception:
                return str(_PROJECT_ROOT / obj)
                
        return obj

    # 执行路径转换
    container = OmegaConf.to_container(cfg, resolve=True)
    normalized_container = _normalize(container)
    final_cfg = OmegaConf.create(normalized_container)
    
    OmegaConf.set_readonly(final_cfg, True)
    
    # 导出实际使用的配置文件路径
    globals()["CONFIG_FILE"] = str(path_to_use)

    return final_cfg

# 导出全局单例 CONFIG
CONFIG: DictConfig = load_config()