import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(filename)s:%(funcName)s:%(lineno)d - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
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
    is_valid, rep = submission.valid()
    if is_valid:
        logging.info('submmision accepted!')
        return submission
    else: 
        logging.info(rep)
        logging.warning('submission invalid!')
        return None
    
def test_pipeline():
    submission = get_submission()
    bench = Bench(
        device='cuda',
        assets_root=CONFIG['path_assets'],
    )
    bench.evaluate(
        submission=submission,
        output_dir='output/',
        metrics_list=['FrechetVideoDistance'],
    )
    
if __name__ == '__main__':
    test_pipeline()