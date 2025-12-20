import os
import yaml
from typing import Dict, Any
from dotenv import load_dotenv

load_dotenv()
DEBUG = os.getenv("DEBUG", "False").lower() == "true"
def load_config(config_path: str = "configs/config.yml"):
    """
    根据 APP_ENV 环境变量加载对应配置。
    Returns: dict: 对应环境的配置字典
    """
    # 1. 读取环境变量，默认为 'dev'
    env = os.getenv("APP_ENV", "dev").lower()
    # print(env)
    
    # 2. 加载整个 YAML 文件
    with open(config_path, "r", encoding="utf-8") as f:
        all_configs = yaml.safe_load(f)
    
    # 3. 检查是否存在该环境配置
    if env not in all_configs:
        available = ", ".join(all_configs.keys())
        raise ValueError(f"Unknown environment '{env}'. Available: {available}")
    
    # 4. 返回对应环境的配置
    return all_configs[env]

CONFIG = load_config()