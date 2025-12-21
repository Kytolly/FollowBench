import gradio as gr
import os
import pandas as pd
import shutil
from pathlib import Path
from huggingface_hub import HfApi

def handle_submit(team, model, file_obj, hf_token, repo_id, output_dir):
    if file_obj is None:
        return "⚠️ Please upload a valid JSON file."
    
    # 本地保存
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    safe_name = "".join([c for c in model if c.isalnum() or c in "-_"])
    filename = f"{timestamp}_{safe_name}_report.json"
    
    save_path = Path(output_dir) / filename
    try:
        shutil.copy(file_obj.name, save_path)
        msg = f"✅ Saved locally: {filename}"
    except Exception as e:
        return f"❌ Local Save Failed: {e}"

    # 云端推送
    if hf_token and repo_id:
        try:
            api = HfApi(token=hf_token)
            api.upload_file(
                path_or_fileobj=save_path,
                path_in_repo=f"submissions/{filename}",
                repo_id=repo_id,
                repo_type="dataset",
                commit_message=f"Submission: {model} by {team}"
            )
            msg += "\n🚀 Pushed to Hugging Face Dataset!"
        except Exception as e:
            msg += f"\n Cloud Upload Failed: {e}"
    else:
        msg += "\n(Cloud upload skipped: HF_TOKEN not set)"
        
    return msg

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