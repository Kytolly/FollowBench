import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import pi

# 1. 读取 CSV 文件
# 确保 'benchmark_results.csv' 在当前目录下
df = pd.read_csv('output/comparision.csv')

# 2. 配置雷达图指标
# 我们选取 6 个最具代表性的核心指标来展示 Trade-off
labels = ['FVD', 'SF', 'HAA', 'CTE', 'SSDC', 'TAA']
all_labels = ['FVD', 'IQ', 'AQ', 'TF', 'DD', 'SF', 'AC', 'HAA', 'SCCR', 'TAA', 'SSDC', 'ADE', 'CCE', 'CTE', 'SCDE', 'CSHA']
labels = all_labels
# 定义每个指标的方向：-1 表示数值越低越好，1 表示数值越高越好
directions = {
    # --- I. 视觉质量 (Visual Quality) ---
    'FVD': -1,   # Fréchet Video Distance (距离越小，分布越接近真实)
    'IQ': 1,     # Imaging Quality (画质评分越高越好)
    'AQ': 1,     # Aesthetic Quality (美学评分越高越好)

    # --- II. 时序动态 (Temporal Dynamics) ---
    'TF': -1,    # Temporal Flickering (闪烁越少越稳定，越低越好)
    'DD': 1,     # Dynamic Degree (动态程度越高越好，防止生成静态图)

    # --- III. 语义与动作 (Semantics & Action) ---
    'SF': 1,     # Structural Fidelity (结构相似度越高越好)
    'AC': 1,     # Appearance Consistency (外观一致性越高越好)
    'HAA': 1,    # Human Action Alignment (动作对齐度越高越好)
    'SCCR': 1,   # Source Control Condition Recall (召回率越高越好)
    'TAA': 1,    # Temporal Attention Alignment (对角线能量越高越好)

    # --- IV. 几何与透视 (Geometry & Perspective) ---
    'SSDC': 1,   # Side-by-Side Depth Consistency (深度图 SSIM 越高越好)
    'ADE': -1,   # Average Displacement Error / Geometric Drift (漂移误差越小越好)

    # --- V. 相机运镜 (Camera Control) ---
    'CCE': -1,   # Camera Centering Error (中心偏移误差越小越好)
    'CTE': -1,   # Camera Trajectory Error (轨迹误差越小越好)
    'SCDE': -1,  # Subject-Camera Distance Error (距离波动误差越小越好)
    'CSHA': -1   # Camera-Subject Heading Alignment (虽然叫 Alignment，但计算的是相对角度的方差 Var，所以越低越好)
}

# 3. 数据归一化 (Min-Max Scaling)
# 将所有指标映射到 [0, 1] 区间，其中 1 代表该指标下的“最佳表现”
df_norm = df.copy()

for col in labels:
    min_val = df[col].min()
    max_val = df[col].max()
    
    # 防止分母为 0
    if max_val == min_val:
        df_norm[col] = 1.0
        continue
        
    if directions[col] == 1: # 越高越好
        df_norm[col] = (df[col] - min_val) / (max_val - min_val)
    else: # 越低越好 (反转)
        df_norm[col] = (max_val - df[col]) / (max_val - min_val)

# 为了视觉美观，给最小值加一点偏移，避免完全缩在中心点
df_norm[labels] = df_norm[labels] + 0.1 

# 4. 绘图
categories = labels
N = len(categories)

# 计算角度
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1] # 闭合回路

plt.figure(figsize=(10, 10), dpi=100)
ax = plt.subplot(111, polar=True)

# 设置坐标标签
plt.xticks(angles[:-1], categories, color='black', size=14)

# 隐藏径向标签（0.2, 0.4...）以保持整洁
ax.set_yticklabels([]) 
plt.ylim(0, 1.2)

# 定义样式
colors = ['#1f77b4', '#d62728', '#2ca02c'] # 蓝(WW), 红(Wan), 绿(Ours)
markers = ['o', 'x', '*']
linestyles = ['--', ':', '-'] # 虚线、点线、实线(Ours)

# 循环绘制每条线
for i, row in df_norm.iterrows():
    values = row[labels].tolist()
    values += values[:1] # 闭合
    
    ax.plot(angles, values, linewidth=2.5, linestyle=linestyles[i], 
            label=row['Method'], color=colors[i], marker=markers[i])
    
    # 填充颜色 (透明度设低一点)
    ax.fill(angles, values, color=colors[i], alpha=0.1)

# 添加图例
plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1), fontsize=12)

# 标题
plt.title('Performance Comparison\n(Outward = Better)', 
          size=16, weight='bold', y=1.08)

# 保存或显示
plt.tight_layout()
plt.savefig('output/comparision_all.png')
plt.show()