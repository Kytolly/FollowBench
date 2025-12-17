import os
import json
import torch
import cv2
from torchvision import transforms
from PIL import Image

class SubmissionLoader:
    def __init__(self, submission_path, load_size=256, clip_len=16):
        """
        Args:
            submission_path: 用户提交的文件夹路径 或 JSON 映射文件
            load_size: 视频 resize 大小 (需与 GT 一致)
            clip_len: 视频长度
        """
        self.submission_path = submission_path
        self.load_size = load_size
        self.clip_len = clip_len
        self.mapping = {}
        
        self.transform = transforms.Compose([
            transforms.Resize((load_size, load_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        # 1. 解析提交内容
        if os.path.isdir(submission_path):
            # 文件夹模式：假设文件名包含 video_id
            print(f"Loading submission from directory: {submission_path}")
            for fname in os.listdir(submission_path):
                if fname.endswith(('.mp4', '.avi', '.mov')):
                    # 简单假设: filename "1001.mp4" -> id "1001"
                    # 或者 "result_1001.mp4" -> "1001"
                    # 这里做简单处理: 去掉扩展名即为 ID (用户需遵守此规范)
                    vid_id = os.path.splitext(fname)[0]
                    self.mapping[vid_id] = os.path.join(submission_path, fname)
        elif submission_path.endswith('.json'):
            # JSON 模式：精确映射
            print(f"Loading submission from JSON: {submission_path}")
            with open(submission_path, 'r') as f:
                self.mapping = json.load(f)
        else:
            raise ValueError("Submission path must be a directory or .json file")

    def get_generated_video(self, video_id):
        """根据 ID 获取生成视频 Tensor"""
        if video_id not in self.mapping:
            # 尝试模糊匹配 (例如 ID="1-3", 文件名="1-3_gen.mp4")
            # 这里暂只支持精确匹配
            return None
            
        video_path = self.mapping[video_id]
        if not os.path.exists(video_path):
            return None
            
        return self._read_video(video_path)

    def _read_video(self, path):
        cap = cv2.VideoCapture(path)
        frames = []
        for _ in range(self.clip_len):
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            frames.append(self.transform(frame))
        cap.release()
        
        if not frames: return None
        
        # Padding
        if len(frames) < self.clip_len:
            frames += [frames[-1]] * (self.clip_len - len(frames))
            
        return torch.stack(frames) # [T, C, H, W]