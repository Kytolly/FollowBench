import gradio as gr
import pandas as pd
import os

# 1. 配置列属性 (名称, 数据类型, 排序默认值)
# 格式: (Column Name, Type, Description)
COLS = [
    ("Model", "str", "Model Name"),
    ("Total Score", "number", "Overall Performance"),
    ("FVD", "number", "Frechet Video Distance (Lower is better)"),
    ("CCE", "number", "Camera Centering Error (Lower is better)"),
    ("HAA", "number", "Human Action Alignment (Lower is better)"),
    ("AC", "number", "Appearance Consistency (Higher is better)"),
    ("BSC", "number", "Background Semantic Consistency (Higher is better)"),
    ("IQ", "number", "Imaging Quality (Higher is better)"),
    ("AQ", "number", "Aesthetic Quality (Higher is better)"),
    # 添加其他指标...
]

# 定义哪些指标是“越低越好”(Lower is Better)，用于高亮显示逻辑（可选）
LOWER_IS_BETTER = ["FVD", "CCE", "HAA", "TF", "TA"]

def get_leaderboard_data():
    """读取 CSV 数据"""
    if os.path.exists("leaderboard.csv"):
        df = pd.read_csv("leaderboard.csv")
        # 确保数据精度，保留2-4位小数
        return df.round(4)
    else:
        # 如果没有文件，返回空 DataFrame 用于演示
        return pd.DataFrame(columns=[c[0] for c in COLS])

def filter_data(search_query):
    """简单的搜索过滤功能"""
    df = get_leaderboard_data()
    if search_query:
        df = df[df["Model"].str.contains(search_query, case=False)]
    return df

# --- 构建 UI ---
with gr.Blocks(title="Ego2Exo Benchmark Leaderboard") as demo:
    gr.Markdown("# 🏆 Ego2ExoFollowShot Benchmark Leaderboard")
    gr.Markdown("""
    **Ego2ExoFollowShot** is a comprehensive benchmark for evaluating Ego-to-Exo video generation.
    It assesses quality, controllability, and consistency across multiple dimensions.
    
    [GitHub Repository](https://github.com/YourUsername/Ego2ExoFollowShot) | [Dataset](https://huggingface.co/datasets/YourUsername/Ego2ExoDataset)
    """)

    with gr.Row():
        search_box = gr.Textbox(placeholder="Search for a model...", label="Search")
        refresh_btn = gr.Button("Refresh")

    # 数据表格
    leaderboard_table = gr.Dataframe(
        value=get_leaderboard_data(),
        headers=[c[0] for c in COLS],
        datatype=[c[1] for c in COLS],
        interactive=False,
        label="Leaderboard",
        # 默认按 Total Score 降序排列
        # sort_by="Total Score", 
    )

    # 绑定事件
    search_box.change(fn=filter_data, inputs=search_box, outputs=leaderboard_table)
    refresh_btn.click(fn=lambda: get_leaderboard_data(), outputs=leaderboard_table)

    gr.Markdown("### Metric Details")
    gr.Markdown("""
    - **FVD**: Frechet Video Distance. Measures distribution gap. (↓)
    - **CCE**: Camera Centering Error. Measures subject tracking. (↓)
    - **HAA**: Human Action Alignment. Measures pose consistency. (↓)
    - **AC/BSC**: Consistency metrics. (↑)
    """)

# 启动
if __name__ == "__main__":
    demo.launch()