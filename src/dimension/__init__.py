from ..utils import gpu

class DimensionEvaluator():
    def __init__(self, device):
        self.device = device
        self.model = None
        
    def prepare(self):
        pass
    
    def compute(self, **kwargs):
        raise NotImplementedError
    
    def clear(self):
        if self.model is not None:
            del self.model
            self.model = None
        gpu.clear_gpu_memory()