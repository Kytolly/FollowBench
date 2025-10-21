import numpy as np
import rembg
import cv2

class BaseImageMatrixProcesser():
    def __init__(self, input_img: np.ndarray, output_path :str):
        self.input_img = input_img # 图片矩阵
        self.output_path = output_path # 保存路径
    
    def run(self):
        pass
    
class FigureExtractor(BaseImageMatrixProcesser):
    def __init__(self, input_img: np.ndarray, output_path :str, model: str='u2net_human_seg'):
        super().__init__(input_img=input_img, output_path=output_path)
        self.model = model # 使用的 rembg 模型 
        # 备选项有['u2net', 'u2netp', 'u2net_human_seg', 
        # 'u2net_cloth_seg', 'silueta', 'isnet-general-use', 
        # 'isnet-general-use', 'isnet-anime']
        
    def process_input(self):
        '''将 self.input_img 编码为 png 格式'''
        ret, buffer = cv2.imencode('.png', self.input_img)
        if ret:
            png_bytes = buffer.tobytes()
        print(png_bytes)
        return png_bytes
    
    def run(self):
        input_data = self.process_input()
        session = rembg.new_session(model_name=self.model)
        output_data = rembg.remove(input_data, session=session)
        
        with open(self.output_path, "wb") as o:
            o.write(output_data)
        o.close()

class FigureOutRectangeMasker(BaseImageMatrixProcesser):
    def __init__(self, input_img: np.ndarray, output_path: str):
        super().__init__(input_img=input_img, output_path=output_path)
    
    def _detect_center_area(self, img: np.ndarray):
        '''返回高度和原图片相同，占中心1/3-2/3区域的中心矩形框区域'''
        height, width = img.shape[:2]
        x1 = width // 3
        x2 = (2 * width) // 3
        x = int(x1)
        y = 0
        w = int(x2 - x1)
        h = int(height)
        return x, y, w, h
    
    def _detect_with_HOG(self, img: np.ndarray):
        '''使用OpenCV的人体检测器检测人物边界框'''
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        boxes, weights = hog.detectMultiScale(
            img, 
            winStride=(8, 8),
            padding=(32, 32),
            scale=1.05,
            hitThreshold=0.5
        )
        
        if len(boxes) > 0:
            try:
                best_idx = int(np.argmax(np.ravel(weights))) # weights 可能是二维数组，扁平化后取最大
            except Exception:
                best_idx = 0
            x, y, w, h = boxes[best_idx]
            return int(x), int(y), int(w), int(h)
        else:
            return None

    def _detect_main_figure(self, img: np.ndarray):
        '''检测图像中的主要人物，返回边界框 (x, y, w, h)'''
        # res = self._detect_with_HOG(img)
        # if res is not None:
        #     return res
        return self._detect_center_area(img)
    
    def _create_masked_image(self, img: np.ndarray, bbox: tuple[int, int, int, int]):
        """创建掩码图像，只保留边界框内的内容"""
        x, y, w, h = bbox
        masked_img = np.zeros_like(img) # 创建黑色背景
        masked_img[y:y+h, x:x+w] = img[y:y+h, x:x+w] # 将边界框内的内容复制到黑色背景上
        return masked_img
    
    def run(self):
        '''给定人物照片，固定一个矩形区域框住人物，其他部分填黑'''
        try:
            bbox = self._detect_main_figure(self.input_img) # 检测主要人物
            result_img = self._create_masked_image(self.input_img, bbox) # 创建掩码图像
            cv2.imwrite(self.output_path, result_img) # 保存结果
        except Exception as e:
            print(f"处理过程中出现错误: {e}")
            cv2.imwrite(self.output_path, self.input_img) # 如果出现错误，保存原始图像
