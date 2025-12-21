import os
from pathlib import Path

from configs import CONFIG

RESULT_DIR = Path("output")
ASSETS_DIR = Path("assets")
CONFIG_DIR = Path("configs")
RESULT_DIR.mkdir(exist_ok=True)
HF_TOKEN = os.environ.get("HF_TOKEN")
SUBMISSION_REPO = CONFIG['repo_submission_id']