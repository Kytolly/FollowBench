import json
import torch
import cv2
import logging
import numpy as np
from pathlib import Path
from torchvision import transforms
from PIL import Image

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

REQUIRED_META_KEYS = {'team_name', 'model_name', "modal", "mode", "contact"}
TOTAL_CASES_NUM = 5 # 总共测试用例
STANDARD_RESOLUTION = (256, 256) # (H, W)
STANDARD_CLIP_LEN = 300 # 帧数要求
STANDARD_CLIP_FPS = 60 # 帧率要求
ALLOWED_EXTENSIONS = {'.mp4', '.avi', '.mov'}

class Submission:
    def __init__(   
            self, 
            source_path, # 解压后生成视频源文件夹 
            submission_path, # 文件映射说明文件
        ):
        self.meta_info = {}
        self.mapping = {}
        self._load(source_path, submission_path)

    def _load(self, source_path, submission_path):
        """包含 meta 和 results"""
        logger.info(f"Loading submission map from JSON: {submission_path}")
        if not Path(submission_path).exists(): # 文件映射说明文件
            raise FileNotFoundError(f"Path not found at {submission_path}.")
        logger.info(f"Loading submission from directory: {source_path}")
        self.source = Path(source_path) 
        if not self.source.exists():
            raise FileNotFoundError(f"Source path not found at {source_path}.")
        with open(Path(submission_path), 'r') as f:
            data = json.load(f)
        f.close()
        if 'results' not in data and 'meta' not in data:
            raise ValueError("JSON must contain 'meta' and 'results' keys.")
        self.meta_info = data['meta']
        self.mapping = data['results']
        
    def valid(self):
        """判断映射后的生成视频文件是否合法"""
        report = []
        is_valid = True
        valid_videos = 0
        total_videos = len(self.mapping)
        
        # 验证映射长度
        if total_videos != TOTAL_CASES_NUM:
            report.append(f"❌ [ID: {id}] File not found: {self.mapping[id]}")
            is_valid = False
            
        # 验证 meta
        missing_meta = REQUIRED_META_KEYS - self.meta_info.keys()
        if missing_meta:
            return False, f"Missing keys in 'meta': {missing_meta}"
        report.append(f"✅ Metadata verified: {self.meta_info['team_name']}")
        
        # 验证映射文件中的每个视频
        for id, rpath in self.mapping.items():
            self.mapping[id] = self.source / rpath
            # 检查文件是否存在
            if not self.mapping[id].exists():
                report.append(f"❌ [ID: {id}] File not found: {self.mapping[id]}")
                is_valid = False
                continue
            
            # 检查是否是支持的编码格式
            if self.mapping[id].suffix not in ALLOWED_EXTENSIONS:
                report.append(f"❌ [ID: {id}] Unsupported file format: {self.mapping[id].suffix}")
                is_valid = False
                continue
            
            # 检查视频完整性
            cap = cv2.VideoCapture(self.mapping[id])
            if not cap.isOpened():
                report.append(f"❌ [ID: {id}] Cannot open video file.")
                is_valid = False
                continue
            
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            cap.release()
            
            # 检查分辨率
            error_msgs = []
            if width != STANDARD_RESOLUTION[1] or \
               height != STANDARD_RESOLUTION[0]: 
                error_msgs.append(f"Resolution {width}x{height} != {STANDARD_RESOLUTION}")
                
            # 帧数检查
            if frame_count != STANDARD_CLIP_LEN:
                error_msgs.append(f"Too short: {frame_count} frames < {STANDARD_CLIP_LEN}")
                
            # FPS 检查
            if fps != STANDARD_CLIP_FPS:
                error_msgs.append(f"FPS {fps} != {STANDARD_CLIP_FPS}")
            
            if error_msgs:
                report.append(f"⚠️ [ID: {id}] Issues: {'; '.join(error_msgs)}")
                is_valid = False
            else:
                valid_videos += 1

        report.append(f"Summary: {valid_videos}/{total_videos} videos are valid.")
        full_report = "\n".join(report)
        return is_valid, full_report
    def get_generated_video(self, video_id: str):
        """根据 ID 获取生成视频 Tensor [C, T, H, W]"""
        video_path: Path = self.mapping[video_id]
        if not video_path.exists():
            logger.error(f"Video file missing for ID {video_id}: {video_path}")
            return None
        return self._read_video(str(video_path))

    def _read_video(self, path):
        """读取视频并转换为 Tensor"""
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            logger.error(f"Failed to open video: {path}")
            return None

        frames = []
        try:
            while len(frames) < self.clip_len:
                ret, frame = cap.read()
                if not ret: break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)
                frames.append(self.transform(frame))
        finally:
            cap.release()
        
        if not frames: 
            return None
        return torch.stack(frames) # Current: [T, C, H, W]