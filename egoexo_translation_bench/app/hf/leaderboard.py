import gradio as gr
import pandas as pd
import json
from pathlib import Path

from ..record.rankboard import RankBoard
from ..record.visualization import main_visualize

def get_leaderboard_df(result_dir, config_dir):
    """生成排行榜 DataFrame"""
    json_files = list(Path(result_dir).glob("*_results.json"))
    if not json_files and (Path(config_dir) / "report_example.json").exists():
        # 兜底逻辑：读取 config/report.json 演示
        json_files = [str(Path(config_dir) / "report_example.json")]
    if not json_files:
        return pd.DataFrame(columns=["Rank", "Model", "Total Score", "Message"], 
                            data=[[0, "No Data", 0, "请运行评估生成 results.json"]])

    try:
        files_str = [str(p) for p in json_files]
        ranker = RankBoard(files_str)
        # 临时 csv 路径
        csv_path = Path(result_dir) / "temp_leaderboard.csv"
        rank_df = ranker.generate_rank(output_csv=str(csv_path))
        return rank_df
    except Exception as e:
        return pd.DataFrame(columns=["Error"], data=[[str(e)]])

def plot_radar(selected_models, result_dir, config_dir):
    """绘制雷达图"""
    if not selected_models:
        return None
        
    all_files = list(Path(result_dir).glob("*_results.json"))
    if not all_files and (Path(config_dir) / "report_example.json").exists():
        all_files = [Path(config_dir) / "report_example.json"] # 兜底

    target_files = []
    # 简单匹配模型名 (建议实际部署优化为 metadata 索引)
    for f in all_files:
        try:
            with open(f, 'r', encoding='utf-8') as file:
                data = json.load(file)
                name = data.get('meta', {}).get('model_name', 'Unknown')
                if name in selected_models:
                    target_files.append(str(f))
        except:
            continue
            
    if not target_files:
        return None

    output_img = Path(result_dir) / "temp_radar.png"
    try:
        main_visualize(target_files, output_img=str(output_img))
        return str(output_img)
    except Exception as e:
        print(f"Radar Plot Error: {e}")
        return None

def create_leaderboard_tab(result_dir, config_dir):
    with gr.TabItem("🏆 Leaderboard & Analysis"):
        gr.Markdown("### 📊 Model Rankings")
        
        # 初始化数据
        df = get_leaderboard_df(result_dir, config_dir)
        all_models = df['Model'].tolist() if 'Model' in df.columns else []

        with gr.Row():
            # 排行榜
            refresh_btn = gr.Button("🔄 Refresh Board", size="sm")
        
        leaderboard_table = gr.Dataframe(
            value=df,
            datatype=["number", "str", "number", "number", "number"],
            interactive=False,
            max_height=400
        )
        
        gr.Markdown("---")
        gr.Markdown("### 📈 Capability Analysis (Radar Chart)")
        
        # 雷达图分析
        with gr.Row():
            with gr.Column(scale=1):
                model_selector = gr.Dropdown(
                    choices=all_models,
                    value=all_models[:3] if len(all_models) >= 3 else all_models,
                    multiselect=True,
                    label="Select Models to Compare",
                    info="Choose models to visualize their strengths."
                )
                plot_btn = gr.Button("🎨 Plot Radar", variant="primary")
            
            with gr.Column(scale=2):
                radar_output = gr.Image(label="Radar Chart", type="filepath")

        # 事件绑定
        refresh_btn.click(
            fn=lambda: get_leaderboard_df(result_dir, config_dir), 
            outputs=leaderboard_table
        )
        plot_btn.click(
            fn=lambda m: plot_radar(m, result_dir, config_dir), 
            inputs=model_selector, 
            outputs=radar_output
        )