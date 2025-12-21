import gradio as gr
import json
import random
import os
import time
import uuid
from pathlib import Path
from huggingface_hub import HfApi, upload_file
from filelock import FileLock

from src.questionaire.feedback import FeedbackSaver
from src.questionaire.blind_study import BlindStudyLoader
from src.questionaire import (
    STYLE_PATH,
    FEEDBACK_PATH,
)

saver = FeedbackSaver(FEEDBACK_PATH)
loader = BlindStudyLoader()

def load_style_config():
    if STYLE_PATH.exists():
        with open(STYLE_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {"meta": {}, "questions": []}

def create_questionaire_tab():
    style = load_style_config()
    questions = style.get("questions", [])
    
    with gr.TabItem("🙋 User Study (Blind Test)"):
        gr.Markdown("### ⚔️ Blind A/B Testing")
        gr.Markdown("请观看第一人称输入 (Ego Input)，对比下方两个视频 (Video A vs Video B)，并选出更好的一方。**注意：A 和 B 的顺序是随机的。**")
        
        # --- 视频展示区 ---
        with gr.Row():
            with gr.Column():
                gr.Markdown("#### 👁️ Ego Input")
                ego_video = gr.Video(label="Input", interactive=False, height=300)
            
            with gr.Column():
                gr.Markdown("#### 🅰️ Video A")
                video_a = gr.Video(label="Model A", interactive=False, height=300)
                
            with gr.Column():
                gr.Markdown("#### 🅱️ Video B")
                video_b = gr.Video(label="Model B", interactive=False, height=300)

        # 隐藏状态：存储当前 A/B 到底对应哪个模型
        state_meta = gr.State()

        gr.Markdown("---")
        
        # --- 动态问卷区 ---
        input_radios = []
        q_ids = []
        
        for q in questions:
            q_ids.append(q['id'])
            with gr.Group():
                gr.Markdown(f"**{q['text']}**")
                if 'criteria' in q:
                    gr.Markdown(f"_{q['criteria']}_")
                
                # 选项处理
                choices = []
                for opt in q['options']:
                    # 兼容 style.json 中的 dict 或 string
                    if isinstance(opt, dict):
                        choices.append((opt['label'], opt['value']))
                    else:
                        choices.append(str(opt))
                
                radio = gr.Radio(
                    choices=choices,
                    label=q.get('prompt', 'Choose one'),
                    interactive=True
                )
                input_radios.append(radio)
        
        # --- 按钮区 ---
        with gr.Row():
            skip_btn = gr.Button("🎲 Skip / Next Case")
            submit_btn = gr.Button("✅ Submit & Next", variant="primary")
        
        status_log = gr.Textbox(label="System Log", lines=1, interactive=False)

        # ================= 逻辑函数 =================
        
        def next_case():
            """加载下一个随机 Case"""
            ego, va, vb, meta = loader.get_blind_pair()
            if not meta:
                return None, None, None, None, "⚠️ No valid cases found in assets/Test."
            
            # 清空所有单选框
            updates = [gr.update(value=None) for _ in input_radios]
            return (ego, va, vb, meta, "🆕 New case loaded.") + tuple(updates)

        def submit_and_next(meta, *answers):
            """处理提交并加载下一个"""
            if not meta:
                return (None, None, None, None, "⚠️ Error: No active case loaded.") + tuple([gr.update()]*len(answers))
            
            if None in answers:
                # 还有未完成的问题
                return (gr.update(), gr.update(), gr.update(), meta, "❌ Please answer all questions before submitting.") + tuple([gr.update()]*len(answers))

            # 1. 解析答案
            # 这里的 logic 是为了生成 analyzer.py 能用的数据格式
            # 我们需要保存: model_a, model_b, winner
            # mapping: {"a": "ModelName1", "b": "ModelName2"}
            mapping = meta["mapping"]
            
            records = []
            session_uuid = str(uuid.uuid4())
            
            for q_id, choice_val in zip(q_ids, answers):
                # choice_val 应该是 'a', 'b', 或 'tie'
                if choice_val == 'a':
                    winner = mapping['a']
                    loser = mapping['b']
                    result_flag = 'model_a' # 对 analyzer 来说, winner 是 model_a
                elif choice_val == 'b':
                    winner = mapping['b']
                    loser = mapping['a']
                    result_flag = 'model_b'
                else:
                    winner = 'tie'
                    loser = 'tie'
                    result_flag = 'tie'
                
                record = {
                    "uuid": session_uuid,
                    "case_id": meta["case_id"],
                    "question_id": q_id,
                    "model_a": mapping['a'], # 记录实际模型名
                    "model_b": mapping['b'],
                    "choice": choice_val,    # 用户选了左还是右
                    "winner": winner,        # 胜出的模型名
                    "timestamp": time.time()
                }
                records.append(record)
            
            # 2. 保存
            msg = saver.save_vote(records)
            
            # 3. 加载下一个
            ego, va, vb, new_meta = loader.get_blind_pair()
            
            # 重置 UI
            updates = [gr.update(value=None) for _ in input_radios]
            
            return (ego, va, vb, new_meta, f"{msg} Loading next...") + tuple(updates)

        # ================= 事件绑定 =================
        
        # 页面加载时自动来一个
        # 注意：Gradio Tab 加载时触发需要用 load，这里用 demo.load 或者由用户点击 Start，
        # 为了简单，我们绑定 Skip 按钮作为 Start，且让 State 初始为空
        
        outputs_list = [ego_video, video_a, video_b, state_meta, status_log] + input_radios
        
        skip_btn.click(
            fn=next_case,
            inputs=[],
            outputs=outputs_list
        )
        
        submit_btn.click(
            fn=submit_and_next,
            inputs=[state_meta] + input_radios,
            outputs=outputs_list
        )