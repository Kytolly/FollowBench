import gradio as gr

from src.utils.hf import handle_submit

def create_submission_tab(hf_token, submission_repo, output_dir):
    with gr.TabItem("📤 Submission"):
        gr.Markdown("### Submit Results")
        gr.Markdown("Upload your `report.json` to join the leaderboard.")
        
        with gr.Row():
            with gr.Column():
                t_name = gr.Textbox(label="Team Name")
                m_name = gr.Textbox(label="Model Name")
                f_up = gr.File(label="Upload report.json", file_types=[".json"])
                sub_btn = gr.Button("Submit", variant="primary")
            
            with gr.Column():
                # 使用 lambda 包装以传入配置参数
                logs = gr.Textbox(label="Submission Logs", lines=5)
        
        sub_btn.click(
            fn=lambda t, m, f: handle_submit(t, m, f, hf_token, submission_repo, output_dir),
            inputs=[t_name, m_name, f_up],
            outputs=logs
        )