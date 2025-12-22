import gradio as gr
from pathlib import Path
import random

def get_example_videos(example_videos_dir):
    """获取示例视频列表"""
    videos = list(Path(example_videos_dir).glob("*.mp4"))
    return videos[:6] if videos else []

def create_gallery_tab(assets_dir):
    with gr.TabItem("🎬 Case Gallery"):
        gr.Markdown("### Generated Samples")
        gr.Markdown("Explore the generation quality of Ego2Exo models.")
        
        videos = get_example_videos(assets_dir)
        
        if not videos:
            gr.Warning(f"No videos found in {assets_dir}. Please populate assets folder.")
            gr.Markdown("**No videos found.** Please upload video samples to `assets/`.")
        else:
            with gr.Row():
                for vid in videos:
                    with gr.Column():
                        gr.Video(value=str(vid), label=vid.name)
            
            with gr.Accordion("See More", open=False):
                gr.Markdown("Wait for more baseline results...")