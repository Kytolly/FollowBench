import logging
DETAILED_FORMAT = (
    '%(asctime)s | '
    '%(levelname)-s | '
    '%(name)s | '
    '%(filename)s:%(lineno)d | '
    '%(funcName)s() | ' 
    # 'PID:%(process)d | TID:%(thread)d | '
    '%(message)s'
)
logging.basicConfig(
    level=logging.INFO,
    format=DETAILED_FORMAT,
    datefmt='%Y-%m-%d %H:%M:%S',
    force=True
)

from configs import CONFIG # 确保在测试环境
from src.utils.persistence import *
from src.dataflow.submission import Submission
from src import Bench
from src.dimension import DIMENSION_NAMES

def get_submission():
    # 构建submission
    sub_path = CONFIG['submission']['submission_path']
    source_path = CONFIG['submission']['source_path']
    submission = Submission(
        source_path=source_path,
        submission_path=sub_path
    )
    submission.validate_all()
    return submission
    
def test_pipeline():
    submission = get_submission()
    bench = Bench(
        device='cuda',
        assets_root=CONFIG['path_assets'],
    )
    bench.evaluate(
        submission=submission,
        output_dir='output/',
        metrics_list=[
            'FrechetVideoDistance', 
            'AestheticQuality', 
            'ImagingQuality', 
            ],
    )
    
if __name__ == '__main__':
    test_pipeline()