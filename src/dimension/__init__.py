from ..utils import gpu
import logging

class DimensionEvaluator():
    def __init__(self, device):
        self.device = device
        self.model = None
        
    def prepare(self):
        logging.info(f"{self.__class__.__name__} prepared!")
    
    def compute(self, **kwargs):
        raise NotImplementedError
    
    def clear(self):
        if self.model is not None:
            del self.model
            self.model = None
        gpu.clear_gpu_memory()