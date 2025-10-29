import torch
from torchvision import transforms
from PIL import Image
from decord import VideoReader
from torchvision.transforms import InterpolationMode
import numpy as np
import random

class ResizeByLongSide:
    def __init__(self, target_long, interpolation=InterpolationMode.BILINEAR):
        """
        Args:
            target_long (int): 缩放后的长边大小
            interpolation: 插值方式，默认双线性
        """
        self.target_long = target_long
        self.interpolation = interpolation

    def __call__(self, img):
        w, h = img.size  # PIL.Image.size = (宽, 高)
        long_side = max(w, h)
        scale = self.target_long / long_side
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        return transforms.functional.resize(img, (new_h, new_w), interpolation=self.interpolation, antialias=True)


# 读取视频并获取第一帧
video_path = 'PrePipeline/data/output/tpv-6-slice--slice-000.mp4'  # 替换为你的实际视频路径
vr = VideoReader(video_path)
frame = vr[250]  # 获取视频的第一帧

# 将该帧转换为PIL图像
img = Image.fromarray(frame.asnumpy()).convert("RGB")

# 定义转换操作
train_video_transforms = transforms.Compose(
    [
        ResizeByLongSide(1505),  # Resize long side to 1505
        transforms.CenterCrop((704, 1280)),  # Crop to (704, 1280)
        transforms.ToTensor(),  # Convert to tensor (scale to [0, 1])
    ]
)

# 应用转换操作
transformed_img = train_video_transforms(img)

# 转换后的图像的形状
print(f"Transformed image shape: {transformed_img.shape}")

# 将张量的像素值限制在 [0, 1] 范围内
transformed_img = torch.clamp(transformed_img, 0, 1)

# 将张量转回PIL图像
img_reverted = transforms.ToPILImage()(transformed_img)

# 保存反归一化后的图像
img_reverted.save('reverted_image.png')
