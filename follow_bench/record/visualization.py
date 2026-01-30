import matplotlib.pyplot as plt
import pandas as pd
import json
import numpy as np
from math import pi

metrics_names = [
    'Frechet Video Distance',
    'Aesthetic Quality',
    'Imaging Quality',
    'Temporal Flickering',
    'Motion Smoothness',
    'Dynamic Degree',
    'Camera Centering Error',
    'Appearance Consistency',
    'Viewpoint Validity',
    'Background Semantic Consistency',
    'Human Action Alignment',
    'Optical Flow Correlation',
    'Trajectory Alignment',
    ]
baseline_names = [
    'CogVideo', 
    'LTX', 
    'SVDXT', 
    'WanI2V', 
    'WanVACE'
    ]
METRIC_NAME_MAP = {
    'FVD': 'FVD (Quality)',
    'AQ': 'Aesthetic',
    'IQ': 'Imaging',
    'TF': 'Flickering',      # Temporal Flickering
    'MS': 'Smoothness',      # Motion Smoothness (Raw value is usually acceleration error)
    'DD': 'Dynamic',         # Dynamic Degree
    'CCE': 'Centering',      # Camera Centering Error
    'AC': 'Appearance',      # Appearance Consistency
    'VV': 'Validity',        # Viewpoint Validity / SDR
    'BSC': 'Background',     # Background Semantic Consistency
    'HAA': 'Action Align',   # Human Action Alignment
    'OFC': 'Flow Corr',      # Optical Flow Correlation
    'TA': 'Trajectory',      # Trajectory Alignment (ADE)
}
LOWER_IS_BETTER = [
    'FVD',  # Fréchet Video Distance
    'TF',   # Temporal Flickering (误差)
    'MS',   # Motion Smoothness (加速度变化量，越低越稳)
    'CCE',  # Camera Centering Error (误差)
    'TA',   # Trajectory Alignment (ADE 距离误差)
]    

def load_and_merge_reports(report_paths):
    """
    读取多个 report.json 并合并为 DataFrame (Model x Metric)
    """
    data = {}
    for path in report_paths:
        with open(path, 'r') as f:
            rpt = json.load(f)
        
        model_name = rpt['meta']['model_name']
        model_scores = {}
        
        for key, val in rpt.items():
            if key == 'meta': continue
            
            # 取平均值
            if isinstance(val, dict):
                score = np.mean(list(val.values()))
            else:
                score = float(val)
                
            model_scores[key] = score
            
        data[model_name] = model_scores
        
    return pd.DataFrame(data).T # Transpose: Rows=Models, Cols=Metrics

def main_visualize(report_paths, output_img='radar_chart.png'):
    df = load_and_merge_reports(report_paths)
    
    # 筛选只在 MAP 中存在的指标
    valid_cols = [c for c in df.columns if c in METRIC_NAME_MAP]
    df = df[valid_cols]
    
    # 归一化 (复用之前的 normalize_data 逻辑，需确保 df 结构一致)
    # ... (此处调用 normalize_data) ...
    df_norm = normalize_data(df) # 假设该函数已定义
    
    plot_radar_chart(df_norm)
    
def preprocess_data(raw_results):
    """
    将原始结果字典转换为 DataFrame 并处理特殊结构 (如 TA)
    """
    # 重构数据：{Model: {Metric: Value}}
    models = list(raw_results['FVD'].keys()) # 假设所有指标的模型列表一致
    processed_data = {model: {} for model in models}
    
    for metric_key, model_dict in raw_results.items():
        if metric_key not in METRIC_NAME_MAP: continue
        for model, value in model_dict.items():
            if metric_key == 'TA' and isinstance(value, dict):
                scalar_value = value.get('ADE', 0)
            elif isinstance(value, dict):
                scalar_value = list(value.values())[0] if value else 0
            else:
                scalar_value = value
            processed_data[model][metric_key] = scalar_value
    return pd.DataFrame(processed_data).T # 行是模型，列是指标
    
def normalize_data(df):
    """Min-Max 归一化，并处理 Lower-is-Better 指标"""
    df_norm = df.copy()
    
    for col in df.columns:
        min_val = df[col].min()
        max_val = df[col].max()
        range_val = max_val - min_val if max_val != min_val else 1.0
        
        if col in LOWER_IS_BETTER:
            df_norm[col] = (max_val - df[col]) / range_val
        else:
            df_norm[col] = (df[col] - min_val) / range_val
    df_norm = df_norm * 0.9 + 0.1 
    return df_norm

def plot_radar_chart(df_norm, title="Ego2Exo Benchmark Evaluation"):
    """绘制雷达图
    """
    # 准备角度
    categories = [METRIC_NAME_MAP.get(c, c) for c in df_norm.columns]
    N = len(categories)
    
    # 计算角度 (最后一个点重复第一个点以闭合)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    # 初始化绘图
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': 'polar'})
    
    # 设置偏移，使第一个轴在正上方
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    
    # 绘制轴标签
    plt.xticks(angles[:-1], categories, color='grey', size=10)
    
    # 设置 Y 轴标签 (不需要显示具体数值，因为归一化了)
    ax.set_rlabel_position(0)
    plt.yticks([0.25, 0.5, 0.75], ["", "", ""], color="grey", size=7)
    plt.ylim(0, 1)
    
    # 颜色列表
    colors = plt.cm.get_cmap("tab10", len(df_norm))
    
    # 绘制每个模型
    for idx, (model_name, row) in enumerate(df_norm.iterrows()):
        values = row.values.flatten().tolist()
        values += values[:1] # 闭合
        
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=model_name, color=colors(idx))
        ax.fill(angles, values, color=colors(idx), alpha=0.1)
        
    # 添加图例
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title(title, size=15, y=1.05)
    
    plt.tight_layout()
    plt.savefig('benchmark_radar_chart.png', dpi=300)
    print("Radar chart saved as 'benchmark_radar_chart.png'")
    plt.show()