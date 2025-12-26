import gradio as gr
import json
import random
import os
import time
import uuid
from pathlib import Path
from huggingface_hub import HfApi, upload_file
from filelock import FileLock

from ..questionaire import (
    HF_TOKEN,
    FEEDBACK_REPO
)

class FeedbackSaver:
    """负责将用户反馈保存到本地并同步到 HF Dataset"""
    def __init__(self, local_path):
        self.local_path = Path(local_path)
        self.lock_path = self.local_path.with_suffix(".lock")
        self._ensure_file()

    def _ensure_file(self):
        if not self.local_path.exists():
            self.local_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.local_path, 'w', encoding='utf-8') as f:
                json.dump([], f)

    def save_vote(self, vote_data):
        """
        保存单次投票数据
        vote_data: list of dicts (每个问题一条记录)
        """
        if not vote_data:
            return "⚠️ No data to save."

        try:
            # 1. 本地追加 (带锁)
            with FileLock(self.lock_path):
                with open(self.local_path, 'r', encoding='utf-8') as f:
                    try:
                        history = json.load(f)
                    except json.JSONDecodeError:
                        history = []
                
                history.extend(vote_data)
                
                with open(self.local_path, 'w', encoding='utf-8') as f:
                    json.dump(history, f, indent=4, ensure_ascii=False)
            
            msg = "✅ Feedback saved locally."

            # 2. 云端同步
            if HF_TOKEN and FEEDBACK_REPO:
                try:
                    api = HfApi(token=HF_TOKEN)
                    api.upload_file(
                        path_or_fileobj=self.local_path,
                        path_in_repo="feedback.json",
                        repo_id=FEEDBACK_REPO,
                        repo_type="dataset",
                        commit_message=f"New votes added at {time.strftime('%H:%M:%S')}"
                    )
                    msg += " & Synced to Cloud ☁️"
                except Exception as e:
                    print(f"Cloud sync failed: {e}")
                    msg += " (Local only)"
            
            return msg
        except Exception as e:
            return f"❌ Save failed: {e}"

