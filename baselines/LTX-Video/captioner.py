import torch
import cv2
from PIL import Image
import numpy as np

from transformers import AutoModelForImageTextToText, AutoProcessor
from qwen_vl_utils import process_vision_info

class VisualCaptioner:
    def __init__(self, model_id, device="cuda"):
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id, device_map=device
        )
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.device = device

    def describe_character(self, image_path):
        """让模型提取角色特征 (外貌、衣着)"""
        prompt = (
            "Describe the character's appearance (clothing, hair, gender) "
            "in a very concise phrase (under 30 words). Do not use full sentences."
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        return self._generate(messages)

    def describe_action_scene(self, video_path):
        """让模型提取视频中的动作和环境 (通过采样帧)"""
        # 采样 8 帧
        frames = self._sample_frames(video_path, num_frames=8)
        
        # 修改点：强调 Concise, focus on motion
        prompt = (
            "Describe the main action and environment concisely (under 30 words). "
            "Ignore camera angles."
        )
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": frames},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        return self._generate(messages)

    def _generate(self, messages):
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return output_text

    def _sample_frames(self, video_path, num_frames=8):
        """从视频中均匀采样帧"""
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = np.linspace(0, total_frames - 1, num_frames).astype(int)
        frames = []
        for i in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                # BGR -> RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame))
        cap.release()
        return frames