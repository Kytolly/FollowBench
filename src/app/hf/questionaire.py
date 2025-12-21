import gradio as gr
import json
from functools import partial

from src.utils.hf import handle_next, handle_submit
from src.questionaire.loader import DatasetLoader
from src.questionaire.blind_study import BlindStudyEngine
from src.questionaire.collector import FeedbackCollector
from src.questionaire import (
    STYLE_PATH,
    FEEDBACK_PATH,
    ASSETS_DIR,
    HF_TOKEN,
    FEEDBACK_REPO
)

loader = DatasetLoader(ASSETS_DIR) # 数据加载器
engine = BlindStudyEngine(loader) # 业务逻辑引擎
collector = FeedbackCollector(FEEDBACK_PATH, HF_TOKEN, FEEDBACK_REPO) # 数据收集器

def load_questions():
    """读取 UI 配置文件"""
    if STYLE_PATH.exists():
        with open(STYLE_PATH, 'r', encoding='utf-8') as f:
            return json.load(f).get("questions", [])
    return []

def create_questionaire_tab():
    questions = load_questions()
    
    with gr.TabItem("🙋 User Study (Blind Test)"):
        gr.Markdown("### ⚔️ Blind A/B Testing System")
        
        # --- 1. 视频展示区域 ---
        with gr.Row():
            with gr.Column():
                gr.Markdown("#### 👁️ Input (Ego)")
                ego_video = gr.Video(label="Ego View", interactive=False, height=320)
            with gr.Column():
                gr.Markdown("#### 🅰️ Video A")
                video_a = gr.Video(label="Option A", interactive=False, height=320)
            with gr.Column():
                gr.Markdown("#### 🅱️ Video B")
                video_b = gr.Video(label="Option B", interactive=False, height=320)

        # 隐藏状态：存储本轮的加密元数据
        state_meta = gr.State()

        # --- 2. 动态问题区域 ---
        input_components = [] # 存储 Radio 组件对象
        question_ids = []     # 存储问题 ID 字符串
        
        for q in questions:
            question_ids.append(q['id'])
            with gr.Group():
                gr.Markdown(f"**{q['text']}**")
                choices = []
                for opt in q['options']:
                    if isinstance(opt, dict):
                        choices.append((opt['label'], opt['value']))
                    else:
                        choices.append(str(opt))
                
                radio = gr.Radio(choices=choices, label=q.get('prompt', ''), interactive=True)
                input_components.append(radio)

        # --- 3. 控制按钮区域 ---
        with gr.Row():
            skip_btn = gr.Button("🎲 Skip / Start", scale=1)
            submit_btn = gr.Button("✅ Submit & Next", variant="primary", scale=2)
        
        status_log = gr.Textbox(label="Status", interactive=False, lines=1)

        # ================= 事件绑定 (Controller Binding) =================
        
        # 定义所有受影响的 UI 输出
        # 注意：顺序必须与 hf.py 中函数的 return 顺序一致
        # (ego, va, vb, meta, log, *radios)
        ui_outputs = [ego_video, video_a, video_b, state_meta, status_log] + input_components

        # A. 绑定 Skip/Next 事件
        # 使用 partial 将 engine 和 input_components 注入到 handle_next
        # 这样 Gradio 调用时只需要触发，不需要传参
        bound_next = partial(handle_next, engine=engine, radio_list=input_components)
        
        skip_btn.click(
            fn=bound_next,
            inputs=[],
            outputs=ui_outputs
        )

        # B. 绑定 Submit 事件
        # 使用 partial 注入固定依赖 (engine, collector, question_ids, input_components)
        # Gradio 会自动传入 inputs 列表中定义的值 (state_meta + radio values) 给剩下的参数
        bound_submit = partial(
            handle_submit, 
            engine=engine, 
            collector=collector, 
            question_ids=question_ids, 
            radio_list=input_components
        )
        
        submit_btn.click(
            fn=bound_submit,
            inputs=[state_meta] + input_components, # 这里的值会传给 meta 和 *answers
            outputs=ui_outputs
        )