import gradio as gr

from src.questionaire.blind_study import BlindStudyEngine
from src.questionaire.collector import FeedbackCollector

def handle_next(engine: BlindStudyEngine, input_components):
    """加载下一题"""
    ego, va, vb, meta = engine.next_round()
    if not meta:
        return (None, None, None, None, "⚠️ No cases available.") + tuple([None]*len(input_components))
    
    # 清空所有 Radio 的选值
    cleared_radios = [gr.update(value=None) for _ in input_components]
    return (ego, va, vb, meta, "🆕 Loaded new case.") + tuple(cleared_radios)

def handle_submit(meta, question_ids, input_components, engine: BlindStudyEngine, collector: FeedbackCollector, *answers):
    """提交并加载下一题"""
    if not meta:
        return (gr.update(), gr.update(), gr.update(), meta, "⚠️ Please start first.") + tuple([gr.update()]*len(answers))
    
    if None in answers:
        return (gr.update(), gr.update(), gr.update(), meta, "❌ Please answer all questions.") + tuple([gr.update()]*len(answers))

    # 1. 使用 Engine 解析数据 (解耦核心)
    records = []
    for q_id, ans in zip(question_ids, answers):
        record = engine.parse_result(meta, q_id, ans)
        records.append(record)
    
    # 2. 使用 Collector 保存数据
    msg = collector.save_batch(records)
    
    # 3. 自动进入下一轮
    ego, va, vb, new_meta = engine.next_round()
    cleared_radios = [gr.update(value=None) for _ in input_components]
    
    return (ego, va, vb, new_meta, f"{msg} -> Next loaded.") + tuple(cleared_radios)

import gradio as gr

def handle_next(engine: BlindStudyEngine, radio_list):
    """
    加载下一题
    Args:
        engine: BlindStudyEngine 实例 (通过 partial 注入)
        radio_list: UI 中的 Radio 组件列表 (用于生成 update，通过 partial 注入)
    """
    ego, va, vb, meta = engine.next_round()
    
    # 构造 Radio 的重置更新 (清空选项)
    cleared_radios = [gr.update(value=None) for _ in radio_list]
    
    if not meta:
        # 如果没有 Case 了，返回空状态和错误提示
        return (None, None, None, None, "⚠️ No cases available.") + tuple(cleared_radios)
    
    return (ego, va, vb, meta, "🆕 Loaded new case.") + tuple(cleared_radios)


def handle_submit(engine: BlindStudyEngine, collector: FeedbackCollector, question_ids, radio_list, meta, *answers):
    """
    处理提交并自动加载下一题
    Args:
        engine: BlindStudyEngine 实例 (固定依赖)
        collector: FeedbackCollector 实例 (固定依赖)
        question_ids: 问题ID列表 (固定依赖)
        radio_list: Radio组件列表 (用于重置UI)
        meta: [Gradio Input] 当前隐藏状态 (gr.State)
        *answers: [Gradio Input] 所有 Radio 的用户选择值
    """
    # 1. 校验状态
    # 构造空更新用于错误返回
    no_updates = [gr.update() for _ in radio_list]
    
    if not meta:
        return (gr.update(), gr.update(), gr.update(), meta, "⚠️ Please start first.") + tuple(no_updates)
    
    if None in answers:
        return (gr.update(), gr.update(), gr.update(), meta, "❌ Please answer all questions.") + tuple(no_updates)

    # 2. 解析数据 (调用业务逻辑层)
    records = []
    # zip 确保问题ID和答案一一对应
    for q_id, ans in zip(question_ids, answers):
        record = engine.parse_result(meta, q_id, ans)
        records.append(record)
    
    # 3. 保存数据 (调用数据层)
    msg = collector.save_batch(records)
    
    # 4. 自动加载下一轮 (复用逻辑)
    # 调用 engine 获取新数据
    ego, va, vb, new_meta = engine.next_round()
    
    if not new_meta:
        return (None, None, None, None, f"{msg} (No more cases).") + tuple([gr.update(value=None) for _ in radio_list])

    # 5. 返回所有更新
    cleared_radios = [gr.update(value=None) for _ in radio_list]
    
    return (ego, va, vb, new_meta, f"{msg} -> Next loaded.") + tuple(cleared_radios)