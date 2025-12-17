from huggingface_hub import HfApi, upload_folder

def upload_to_hub():
    repo_id = "Kytolly/Ego2ExoFollowShotBenchmark"
    local_dir = "/opt/liblibai-models/user-workspace2/dataset/ego/formatted_dataset" # 你整理好的本地数据集目录
    
    print(f"Uploading {local_dir} to {repo_id}...")
    
    api = HfApi()
    
    # 1. 创建仓库 (如果不存在)
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
    
    # 2. 上传文件夹
    upload_folder(
        folder_path=local_dir,
        repo_id=repo_id,
        repo_type="dataset",
        multi_commits=True, # 如果文件很多，分批提交
        multi_commits_verbose=True
    )
    print("Upload complete!")

if __name__ == "__main__":
    upload_to_hub()