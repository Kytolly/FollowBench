from huggingface_hub import upload_file

def push_to_hub(csv_path):
    """将本地的 leaderboard.csv 推送到 Hugging Face Space"""
    print("Pushing leaderboard to Hugging Face Space...")
    try:
        upload_file(
            path_or_fileobj=csv_path,
            path_in_repo="leaderboard.csv",
            repo_id="YourUsername/Ego2Exo-Leaderboard",
            repo_type="space",
            token="YOUR_HF_WRITE_TOKEN" # 建议从环境变量读取
        )
        print("Successfully updated leaderboard!")
    except Exception as e:
        print(f"Failed to push leaderboard: {e}")
        
def update_human_scores(self, elo_ratings):
        """
        elo_ratings: dict {model_name: score}
        将 Elo 分数更新到 Human_Elo 列
        """
        # 读取现有榜单
        df = pd.read_csv(self.path)
        
        # 映射分数
        df['Human_Elo'] = df['Team'].map(elo_ratings)
        
        # 填充没有人工评分的模型 (可选：填默认分或 NaN)
        df['Human_Elo'] = df['Human_Elo'].fillna(1000) 
        
        # 重新计算总分 (如果总分包含人工分)
        # df['Total_Score'] = ...
        
        df.to_csv(self.path, index=False)