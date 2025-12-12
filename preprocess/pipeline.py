from utils.imgmat import BaseImageMatrixProcesser, FigureExtractor, FigureOutRectangeMasker
from utils.rie import ReferenceImageExtractor
from utils.videocutter import VideoCutter
import cv2
import numpy as np
import os
import threading

fpvs_path = [f'/opt/liblibai-models/user-workspace2/dataset/ego/new/{i}/fpv.mp4' for i in range(32)]
tpvs_path = [f'/opt/liblibai-models/user-workspace2/dataset/ego/new/{i}/tpv.mp4' for i in range(32)]
des_fpv_slices_path = '/opt/liblibai-models/user-workspace2/dataset/ego/First_Video'
des_tpv_slices_path = '/opt/liblibai-models/user-workspace2/dataset/ego/Third_Video'
des_ref_images_path = '/opt/liblibai-models/user-workspace2/dataset/ego/Reference_Image'

fpv_workers = [
            VideoCutter(
                input_video_path=fpvs_path[i], 
                output_dir=des_fpv_slices_path, 
                base_name=f'{i+1}-1', 
                frames_per_slice=300
            )
            for i in range(32)
            ]
tpv_workers = [
            VideoCutter(
                input_video_path=tpvs_path[i], 
                output_dir=des_tpv_slices_path, 
                base_name=f'{i+1}-3', 
                frames_per_slice=300
            )
            for i in range(32)
            ]
workers = fpv_workers + tpv_workers
threads = []

if __name__ == '__main__':
    # 获取视频对，创建 64 个进程进行切片工作
    for i, w in enumerate(workers):
        t = threading.Thread(target=w.run, name=f"vc-{i}")
        threads.append(t)
        t.start()
    for t in threads:
        t.join()
    print("All video cutting workers completed.")
    
    # 获得参考图像
    for i in range(1, 33):
        for j in range(1, 121):
            video_path = f'{des_tpv_slices_path}/{i}-3_part_{j:03d}.mp4'
            refimg_path = f'{des_ref_images_path}/{i}-3_part_{j:03d}.png'
            worker = ReferenceImageExtractor(input_video_path=video_path, output_path=refimg_path)
            worker.run(seed=42, mode='FigORM')