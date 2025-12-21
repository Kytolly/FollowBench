import random
import time
import uuid

class BlindStudyEngine:
    def __init__(self, loader):
        self.loader = loader

    def next_round(self):
        """
        生成下一轮测试数据
        Returns:
            ego_path, video_left, video_right, session_meta (隐藏状态)
        """
        case = self.loader.get_random_case()
        if not case:
            return None, None, None, None

        # 选取两个模型进行对比 (目前支持两两对比，未来可扩展多选)
        models = list(case["candidates"].keys())
        if len(models) < 2:
            return None, None, None, None
            
        # 这里简单取前两个，也可以随机取两个
        m1, m2 = models[0], models[1]
        path1 = case["candidates"][m1]
        path2 = case["candidates"][m2]

        # --- 核心逻辑：随机交换 ---
        if random.random() > 0.5:
            # A=m1, B=m2
            video_left, video_right = path1, path2
            mapping = {"a": m1, "b": m2}
        else:
            # A=m2, B=m1
            video_left, video_right = path2, path1
            mapping = {"a": m2, "b": m1}

        # 生成本次会话的元数据 (存入 gr.State)
        meta = {
            "session_id": str(uuid.uuid4()),
            "case_id": case["id"],
            "timestamp": time.time(),
            "mapping": mapping  # 记录真实的模型对应关系
        }

        return case["input"], video_left, video_right, meta

    def parse_result(self, meta, question_id, user_choice):
        """
        解析用户选择，还原为真实的胜负关系
        Args:
            meta: next_round 返回的隐藏状态
            user_choice: 'a', 'b', or 'tie'
        Returns:
            dict: 用于存入数据库的记录
        """
        mapping = meta["mapping"]
        
        if user_choice == 'a':
            winner = mapping['a']
            result_type = 'model_a_win'
        elif user_choice == 'b':
            winner = mapping['b']
            result_type = 'model_b_win'
        else:
            winner = 'tie'
            result_type = 'tie'

        return {
            "uuid": meta["session_id"],
            "case_id": meta["case_id"],
            "question_id": question_id,
            "model_a": mapping['a'], # 始终记录 Model A 是谁
            "model_b": mapping['b'], # 始终记录 Model B 是谁
            "user_choice": user_choice, # 用户选了左还是右
            "winner": winner,        # 真实的赢家模型名
            "timestamp": time.time()
        }