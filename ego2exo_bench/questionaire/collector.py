import json
import time
from pathlib import Path
from huggingface_hub import HfApi
from filelock import FileLock

class FeedbackCollector:
    def __init__(self, local_path, hf_token=None, repo_id=None):
        self.local_path = Path(local_path)
        self.lock_path = self.local_path.with_suffix(".lock")
        self.hf_token = hf_token
        self.repo_id = repo_id
        self._init_storage()

    def _init_storage(self):
        if not self.local_path.exists():
            self.local_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.local_path, 'w', encoding='utf-8') as f:
                json.dump([], f)

    def save_batch(self, records):
        """保存一批记录"""
        if not records: 
            return "No data"
            
        try:
            # 1. 本地保存
            with FileLock(self.lock_path):
                with open(self.local_path, 'r', encoding='utf-8') as f:
                    try:
                        data = json.load(f)
                    except:
                        data = []
                data.extend(records)
                with open(self.local_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=4, ensure_ascii=False)
            
            msg = "✅ Saved locally"

            # 2. 云端同步
            if self.hf_token and self.repo_id:
                try:
                    api = HfApi(token=self.hf_token)
                    api.upload_file(
                        path_or_fileobj=self.local_path,
                        path_in_repo="feedback.json",
                        repo_id=self.repo_id,
                        repo_type="dataset",
                        commit_message=f"Sync feedback {len(records)} items"
                    )
                    msg += " & Synced to HF"
                except Exception as e:
                    print(f"Sync error: {e}")
            
            return msg
        except Exception as e:
            return f"❌ Error: {e}"