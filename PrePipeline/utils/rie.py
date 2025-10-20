import numpy as np
import cv2
from imgmat import FigureExtractor, FigureOutRectangeMasker, BaseImageMatrixProcesser

IMP_DICT = {
    'FE': FigureExtractor,
    'FORM': FigureOutRectangeMasker,
}            
class ReferenceImageExtractor():
    def __init__(self, input_video_path, output_path):
        self.input_video_path = input_video_path
        self.cap = cv2.VideoCapture(self.input_video_path)
        if not self.cap.isOpened():
            raise Exception(f"fail to open video file: {self.input_video_path}")
        self.output_path = output_path

    def get_total_frames(self):
        '''获取输入视频的总帧数'''
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def get_fps(self):
        '''获取输入视频的帧率'''
        return self.cap.get(cv2.CAP_PROP_FPS)

    def choose_index(self, total_frames, idx=None, time_sec=None, seed=None):
        '''选取保存的帧索引'''
        if total_frames <= 0:
            return None
        if idx is not None:
            i = int(max(0, min(total_frames - 1, int(idx))))
            return i
        if time_sec is not None:
            fps = self.get_fps() or 0
            if fps > 0:
                i = int(time_sec * fps)
                i = max(0, min(total_frames - 1, i))
                return i
        if seed is not None:
            rng = np.random.default_rng(seed)
            return int(rng.integers(0, total_frames))
        return total_frames // 2

    def get_frame_matrix(self, idx=None, time_sec=None, seed=None):
        '''抽取一张帧, 获取帧矩阵'''
        total = self.get_total_frames()
        if total == 0:
            print("video has 0 frames")
            return None

        frame_idx = self.choose_index(total, idx=idx, time_sec=time_sec, seed=seed)
        if frame_idx is None:
            print("failed to determine frame index")
            return None
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        _, frame = self.cap.read()
        return frame_idx, frame
    
    def run(self, idx=None, time_sec=None, seed=None, mode='FORM'):
        '''抽取并保存一张帧, 提取人物轮廓'''
        frame_idx, frame = self.get_frame_matrix(idx=idx, time_sec=time_sec, seed=seed)
        imp :BaseImageMatrixProcesser= IMP_DICT[mode](input_img=frame, output_path=self.output_path)
        imp.run()
        self.cap.release()
        print(f"saved {self.output_path} (frame {frame_idx})")
        