import os
from pathlib import Path

from configs import CONFIG

STYLE_PATH = Path("configs/user_study.json")
FEEDBACK_PATH = Path("configs/feedback_example.json")
ASSETS_DIR = Path("assets")
OUTPUT_DIR = Path("output")
HF_TOKEN = os.environ.get("HF_TOKEN")
FEEDBACK_REPO = CONFIG['repo_id']