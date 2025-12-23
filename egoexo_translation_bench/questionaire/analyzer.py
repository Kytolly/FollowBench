import pandas as pd
import numpy as np

class HumanEvalAnalyzer:
    def __init__(self, k_factor=32, initial_rating=1000):
        self.k = k_factor
        self.initial_rating = initial_rating

    def compute_elo(self, df_votes):
        """
        df_votes cols: [model_a, model_b, winner]
        winner: 'a', 'b', or 'tie'
        """
        ratings = {} # {model_name: rating}

        # 初始化所有出现的模型
        models = set(df_votes['model_a'].unique()) | set(df_votes['model_b'].unique())
        for m in models:
            ratings[m] = self.initial_rating

        # 遍历每一票更新分数
        for _, row in df_votes.iterrows():
            r_a = ratings[row['model_a']]
            r_b = ratings[row['model_b']]

            # 计算期望胜率
            e_a = 1 / (1 + 10 ** ((r_b - r_a) / 400))
            e_b = 1 / (1 + 10 ** ((r_a - r_b) / 400))

            # 实际得分
            if row['winner'] == 'a':
                s_a, s_b = 1, 0
            elif row['winner'] == 'b':
                s_a, s_b = 0, 1
            else: # tie
                s_a, s_b = 0.5, 0.5

            # 更新
            ratings[row['model_a']] += self.k * (s_a - e_a)
            ratings[row['model_b']] += self.k * (s_b - e_b)

        return ratings