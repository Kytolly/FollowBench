import gradio as gr

from ...configs import CONFIG
from .leaderboard import create_leaderboard_tab
from .gallery import create_gallery_tab
from .submission import create_submission_tab
from .questionaire import create_questionaire_tab
from .about import create_about_tab
from .citation import create_citation_tab

from . import (
    RESULT_DIR,
    ASSETS_DIR,
    CONFIG_DIR,
    HF_TOKEN,
    SUBMISSION_REPO,
)

def main():
    with gr.Blocks(title="Ego2Exo Benchmark") as demo:
        gr.Markdown("# 🏃 Ego2Exo under Follow Camera Benchmark")
        with gr.Tabs():
            create_leaderboard_tab(RESULT_DIR, CONFIG_DIR) # 排行榜
            create_gallery_tab(ASSETS_DIR) # 视频画廊
            create_submission_tab(HF_TOKEN, SUBMISSION_REPO, RESULT_DIR) # 提交入口
            create_questionaire_tab() # 问卷
            create_about_tab() # 关于
            create_citation_tab() # 引用
    return demo

if __name__ == "__main__":
    demo = main()
    demo.launch(server_name="127.0.0.1", server_port=7860, theme=gr.themes.Soft())