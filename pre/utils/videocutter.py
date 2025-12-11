import cv2
import os

class VideoCutter():
    def __init__(self, input_video_path, output_dir, base_name, frames_per_slice=300):
        self.cap = cv2.VideoCapture(input_video_path)
        self.std_frames_per_slice = frames_per_slice
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.base_name = base_name
        self.output_dir = output_dir
        if not self.cap.isOpened():
            raise Exception(f'fail to open video file: {input_video_path}.')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    def get_fps(self):
        '''获取输入视频的帧率'''
        return self.cap.get(cv2.CAP_PROP_FPS)

    def get_total_frames(self):
        '''获取输入视频的总帧数'''
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
    def get_resolution(self):
        '''获取输入视频的分辨率'''
        return int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    def get_duration(self):
        '''获取输入视频的时长'''
        fps = self.get_fps()
        total_frames = self.get_total_frames()
        return total_frames / fps if fps else 0.0

    def get_num_clips(self):
        '''获取切片数量'''
        total_frames = self.get_total_frames()
        if total_frames == 0:
            return 0
        cnt = total_frames // self.std_frames_per_slice
        return cnt if total_frames == self.std_frames_per_slice * cnt else cnt + 1
    
    def clip_one(self, idx, tqdm=None, total_bar=None):
        '''进行一个切片的写入操作'''
        start_frame = idx * self.std_frames_per_slice
        end_frame = min((idx + 1) * self.std_frames_per_slice, self.get_total_frames())
        frames_to_write = end_frame - start_frame

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        output_filename = f'{self.base_name}-slice-{idx:03d}.mp4'
        output_path = os.path.join(self.output_dir, output_filename)
        
        out = cv2.VideoWriter(output_path, self.fourcc, self.get_fps(), self.get_resolution())
        if not out.isOpened():
            raise Exception(f"fail to create output file: {output_path}.")

        if tqdm:
            slice_bar = tqdm(total=frames_to_write, desc=f"Slice {idx+1}/{self.get_num_clips()}", unit="frame", position=1, leave=False)
        else:
            slice_bar = None
            print(f"Processing slice {idx+1}/{self.get_num_clips()}: {output_filename} [frames {start_frame}-{end_frame}]")

        current_frame = start_frame
        frames_written = 0
        while current_frame < end_frame:
            ret, frame = self.cap.read()
            if not ret:
                raise Exception(f'fail to read the {current_frame}th frame')
            out.write(frame)
            current_frame += 1
            frames_written += 1
            if slice_bar:
                slice_bar.update(1)
            if total_bar:
                total_bar.update(1)
            else:
                if frames_written % 100 == 0 or current_frame == end_frame:
                    processed = min(current_frame, self.get_total_frames())
                    print(f"Processed {processed}/{self.get_total_frames()} frames...")

        out.release()
        if slice_bar:
            slice_bar.close()
        print(f'successfully created slice {idx+1}, written {frames_written} frames: {output_filename}')
        
    def run(self):
        '''执行切片动作 带进度条显示 需要安装 tqdm 否则退回到简单打印'''
        try:
            from tqdm import tqdm
        except Exception:
            tqdm = None

        if tqdm:
            total_bar = tqdm(total=self.get_total_frames(), desc="Total Progress", unit="frame", position=0)
        else:
            total_bar = None
            print(f"Starting processing {self.get_total_frames()} frames in {self.get_num_clips()} slices...")

        try:
            for idx in range(self.get_num_clips()):
                self.clip_one(idx, tqdm, total_bar)
        finally:
            if total_bar:
                total_bar.close()
            self.cap.release()
            print(f'entire video has been clipped into {self.get_num_clips()} slices!')
            print(f'please check {self.output_dir} to view all slices')
