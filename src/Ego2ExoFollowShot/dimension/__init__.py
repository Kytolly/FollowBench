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

class DimensionEvaluator():
    def __init__(self):
        pass
    
    def prepare(self):
        pass
    
    def compute(self, *kwargs):
        pass
    
    def clear(self):
        pass
    
    def run(self, *kwargs):
        self.prepare()
        self.compute(*kwargs)
        self.clear()