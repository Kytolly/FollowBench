from . import DimensionEvaluator
from dimension.metrics import FrechetVideoDistance
import yaml

class FrechetVideoDistanceEvaluator(DimensionEvaluator):
    def compute(self, *args, **kwargs):
        return None # 不支持单视频计算

    def compute_dataset(self, gen_dir, gt_dir):
        # 读取配置 (假设 config 路径固定或通过某种方式传入，这里简化处理)
        # 实际项目中建议将 config 传入 Evaluator init
        with open('src/Ego2ExoFollowShot/config.yml', 'r') as f:
            config = yaml.safe_load(f)
            
        return FrechetVideoDistance(
            repo_path=config['FVD']['STYLEGANV_REPO_PATH'],
            real_videos_dir=str(gt_dir),
            gen_videos_dir=str(gen_dir),
            mirror=config['FVD']['MIRROR'],
            gpus=config['FVD']['GPUS'],
            resolution=config['FVD']['RESOLUTION'],
            metrics=config['FVD']['METRICS'],
        )