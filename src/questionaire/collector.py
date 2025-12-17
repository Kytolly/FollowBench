import os
import pandas as pd
from huggingface_hub import snapshot_download

class HumanEvalCollector:
    def __init__(self, hf_dataset_id, local_dir="./assets/human_feedback"):
        self.hf_dataset_id = hf_dataset_id
        self.local_dir = local_dir
        os.makedirs(local_dir, exist_ok=True)

    def sync(self):
        """从 Hugging Face 同步最新的投票数据"""
        print(f"Syncing human feedback from {self.hf_dataset_id}...")
        try:
            # 下载 dataset 仓库中的所有文件
            snapshot_download(
                repo_id=self.hf_dataset_id,
                repo_type="dataset",
                local_dir=self.local_dir,
                allow_patterns=["*.csv", "*.json"] # 只下载数据文件
            )
            print("Sync complete.")
            return True
        except Exception as e:
            print(f"Sync failed: {e}")
            return False

    def load_raw_data(self):
        """读取目录下所有的 CSV 合并为一个 DataFrame"""
        all_files = [os.path.join(self.local_dir, f) for f in os.listdir(self.local_dir) if f.endswith('.csv')]
        if not all_files:
            return pd.DataFrame()
        return pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)