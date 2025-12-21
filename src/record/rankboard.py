import pandas as pd
from .analysis import Analyzer

# 定义指标方向 (True: Higher is Better, False: Lower is Better)
METRIC_DIRECTION = {
    'AestheticQuality': True,
    'ImagingQuality': True,
    'AppearanceConsistency': True,
    'BackgroundSemanticConsistency': True,
    'HumanActionAlignment': False, # 假设是误差/距离
    'CameraCenteringError': False,
    'FrechetVideoDistance': False,
    'TemporalFlickering': False,
    'MotionSmoothness': False, # 假设是加速度，越小越稳
    'DynamicDegree': True, # 视情况而定，假设动态越大越好
    'OpticalFlowCorrelation': True,
    'TrajectoryAlignment': False,
    'ViewpointValidity': True
}

class RankBoard:
    def __init__(self, report_paths):
        self.analyzer = Analyzer(report_paths)
        
    def generate_rank(self, output_csv="leaderboard.csv"):
        df_stats = self.analyzer.get_metric_stats()
        
        # 只取 Mean 列
        metric_cols = [c for c in df_stats.columns if c.endswith('_Mean')]
        rank_df = df_stats[['Model']].copy()
        
        # 1. 归一化 (Min-Max Normalization)
        # 将所有指标映射到 [0, 1] 区间，且修正方向为 1=Best
        normalized_scores = pd.DataFrame()
        
        for col in metric_cols:
            raw_metric_name = col.replace('_Mean', '')
            higher_is_better = METRIC_DIRECTION.get(raw_metric_name, True)
            
            vals = df_stats[col]
            min_v, max_v = vals.min(), vals.max()
            
            if max_v == min_v:
                norm_v = 1.0 # 无法区分
            else:
                norm_v = (vals - min_v) / (max_v - min_v)
                
            if not higher_is_better:
                norm_v = 1.0 - norm_v
                
            normalized_scores[raw_metric_name] = norm_v
            
        # 2. 计算加权总分 (简单平均)
        rank_df['Total_Score'] = normalized_scores.mean(axis=1) * 100
        
        # 3. 合并原始分数用于展示
        final_df = pd.concat([rank_df, df_stats[metric_cols]], axis=1)
        
        # 4. 排序
        final_df = final_df.sort_values(by='Total_Score', ascending=False).reset_index(drop=True)
        
        # 5. 添加排名列
        final_df.insert(0, 'Rank', final_df.index + 1)
        
        final_df.to_csv(output_csv, index=False)
        print(f"Leaderboard saved to {output_csv}")
        return final_df