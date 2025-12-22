import os
import yaml
from typing import Dict, Any, AnyStr
from dotenv import load_dotenv
# Note: avoid importing heavy packages at module import time (e.g., torch) in config module

load_dotenv()
DEBUG = os.getenv("DEBUG", "False").lower() == "true"
from pathlib import Path


def load_config(config_path: str = "config.yml") -> Dict[str, Any]:
    """
    根据 APP_ENV 环境变量加载对应配置。

    查找顺序：
      1. 如果环境变量 CONFIG_PATH 指定路径，则优先使用它（可以是绝对或相对路径）。
      2. 在 package 的 `configs` 目录下查找（即随包发布的配置，优先级最高）。
      3. 在包根目录下查找（仓库被直接 import 的场景）。
      4. 在当前工作目录下查找（保持向后兼容）。

    Returns: dict: 对应环境的配置字典
    """
    env = os.getenv("APP_ENV", "dev").lower()

    # 环境变量覆盖路径
    env_config_path = os.getenv("CONFIG_PATH")
    if env_config_path:
        config_path = env_config_path

    tried = []
    path_to_use = None
    p = Path(config_path)

    # 1) 绝对路径直接使用
    if p.is_absolute():
        tried.append(str(p))
        if p.exists():
            path_to_use = p
        else:
            raise FileNotFoundError(f"Config file not found at absolute path: {p}")
    else:
        # 2) package 内的 configs 目录（一般为安装后或在仓库内）
        package_configs = Path(__file__).resolve().parent
        candidate = package_configs / config_path
        tried.append(str(candidate))
        if candidate.exists():
            path_to_use = candidate

        # 3) 包根目录（例如仓库根）
        if path_to_use is None:
            repo_root_candidate = package_configs.parent / config_path
            tried.append(str(repo_root_candidate))
            if repo_root_candidate.exists():
                path_to_use = repo_root_candidate

        # 4) 当前工作目录
        if path_to_use is None:
            cwd_candidate = Path.cwd() / config_path
            tried.append(str(cwd_candidate))
            if cwd_candidate.exists():
                path_to_use = cwd_candidate

    if path_to_use is None:
        raise FileNotFoundError(
            f"Config file not found. Tried: {', '.join(tried)}. Set CONFIG_PATH to override."
        )

    with open(path_to_use, "r", encoding="utf-8") as f:
        all_configs = yaml.safe_load(f)

    if not isinstance(all_configs, dict):
        raise ValueError(f"Config file {path_to_use} did not contain a mapping at top level.")

    if env not in all_configs:
        available = ", ".join(all_configs.keys())
        raise ValueError(f"Unknown environment '{env}'. Available: {available}")

    cfg = all_configs[env]

    # 规范化：将配置中相对路径（相对于包根）转换为绝对路径，便于在任意位置 import 使用
    def _is_path_like(key: str, val: str) -> bool:
        if not isinstance(val, str):
            return False
        # 已为绝对路径
        if Path(val).is_absolute():
            return False
        key = (key or "").lower()
        keywords = ["path", "dir", "file", "repo", "asset", "source", "submission", "config", "data", "root"]
        if any(k in key for k in keywords):
            return True
        # 含有路径分隔符或以当前/父目录开头
        if ("/" in val) or ("\\" in val) or val.startswith("."):
            return True
        # 看起来像文件名（有扩展名）
        parts = val.rsplit(".", 1)
        if len(parts) == 2 and 1 <= len(parts[1]) <= 5:
            return True
        return False

    package_configs = Path(__file__).resolve().parent
    package_root = package_configs.parent

    def _normalize(obj: object, parent_key: str = "") -> object:
        if isinstance(obj, dict):
            for k, v in list(obj.items()):
                obj[k] = _normalize(v, k)
            return obj
        if isinstance(obj, (list, tuple)):
            new = [_normalize(x, parent_key) for x in obj]
            return type(obj)(new)
        if isinstance(obj, str):
            if _is_path_like(parent_key, obj):
                # 将相对路径解析到包根
                return str((package_root / obj).resolve())
            return obj
        return obj

    cfg = _normalize(cfg)

    # 将找到的配置文件路径以及包相关位置导出，方便其他模块使用
    globals()["CONFIG_FILE"] = str(path_to_use)
    globals()["PACKAGE_ROOT"] = str(package_root)
    globals()["CONFIG_DIR"] = str(package_configs)

    return cfg

# 导出带类型的常用配置对象
CONFIG: Dict[str, Any] = load_config()