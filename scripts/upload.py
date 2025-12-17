import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from huggingface_hub import HfApi, upload_folder
from scripts.env import get_env_config
import argparse

def upload_to_hub(env_mode):
    config = get_env_config(env_mode)
    local_dir = config['path_assets']
    repo_id = config['repo_id']
    print(f"Uploading {local_dir} to {repo_id}...")
    
    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
    upload_folder(
        folder_path=local_dir,
        repo_id=repo_id,
        repo_type="dataset",
        # multi_commits=True,
        # multi_commits_verbose=True
    )
    print("Upload complete!")

def main():
    parser = argparse.ArgumentParser(description="uploading HuggingFace Hub")
    parser.add_argument('-e', '--env', default='dev', help='[eg.]dev prod test')
    args: dict = parser.parse_args()
    
    upload_to_hub(args.env)
    
if __name__ == "__main__":
    main()