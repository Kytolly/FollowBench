import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from math import pi

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

def generate_dummy_data():
    """生成测试数据"""
    models = ['CogVideo', 'LTX', 'SVDXT', 'WanI2V', 'WanVACE']
    metrics = list(METRIC_NAME_MAP.keys())
    data = {m: {} for m in metrics}
    
    for m in metrics:
        for model in models:
            # 随机生成模拟数据
            if m == 'FVD': val = np.random.uniform(100, 500)
            elif m in ['AQ', 'IQ']: val = np.random.uniform(4, 7)
            elif m == 'TA': val = {'ADE': np.random.uniform(0.1, 0.5)} # 模拟嵌套结构
            else: val = np.random.uniform(0.5, 0.95)
            data[m][model] = val
    return data