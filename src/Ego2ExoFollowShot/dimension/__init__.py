from fvd import FrechetVideoDistanceEvaluator
from aq import AestheticQualityEvaluator
from iq import ImagingQualityEvaluator
from tf import TemporalFlickeringEvaluator
from ms import MotionSmoothnessEvaluator
from dd import DynamicDegreeEvaluator
from cce import CameraCenteringErrorEvaluator
from ac import AppearanceConsistencyEvaluator
from vv import ViewpointValidityEvaluator
from bsc import BackgroundSemanticConsistencyEvaluator
from haa import HumanActionAlignmentEvaluator
from ofc import OpticalFlowCorrelationEvaluator
from ta import TrajectoryAlignmentEvaluator

import utils.gpu

class DimensionEvaluator():
    def __init__(self, device):
        self.device = device
        self.model = None
        
    def prepare(self):
        pass
    
    def compute(self, *args):
        raise NotImplementedError
    
    def clear(self):
        if self.model is not None:
            del self.model
            self.model = None
        utils.gpu.clear_gpu_memory()
    
    def run(self, *args):
        self.prepare()
        res = self.compute(*args)
        self.clear()
        return res