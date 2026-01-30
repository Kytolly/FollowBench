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

from follow_bench.configs import CONFIG # 确保在测试环境
from follow_bench.utils.persistence import *
from follow_bench.dataflow.submission import Submission
from follow_bench import Bench
from follow_bench.dimension import DIMENSION_NAMES

def get_submission():
    # 3. 使用 OmegaConf 的点号访问 (Config Refactor 适配)
    # 以前: CONFIG['submission']['submission_path']
    # 现在: CONFIG.submission.submission_path
    sub_path = CONFIG.submission.submission_path
    source_path = CONFIG.submission.source_path
    
    # 确保路径存在 (因为 OmegaConf 可能把它转成了绝对路径字符串)
    logging.info(f"Loading submission from: {sub_path}")
    logging.info(f"Source video path: {source_path}")
    
    submission = Submission(
        source_path=source_path,
        submission_path=sub_path
    )
    submission.validate_all()
    return submission
    
def test_pipeline():
    logging.info("Starting pipeline test...")
    
    try:
        submission = get_submission()
    except Exception as e:
        logging.error(f"Failed to load submission: {e}")
        return

    # 初始化 Benchmark
    bench = Bench(
        device='cuda', # 确保你的环境有 CUDA，否则改为 'cpu'
        assets_root=CONFIG.path_assets, # 使用 OmegaConf 访问
    )
    
    logging.info("Bench initialized. Starting evaluation...")
    
    # 运行评测
    bench.evaluate(
        submission=submission,
        output_dir='output/',
        metrics_list=[
            'FrechetVideoDistance', 
            # 'AestheticQuality', # 如果没有安装相关权重，可以先注释掉
            # 'ImagingQuality', 
            'HumanActionAlignment' # 建议加上这个核心指标
            ],
    )
    logging.info("Evaluation finished.")
    
if __name__ == '__main__':
    test_pipeline()