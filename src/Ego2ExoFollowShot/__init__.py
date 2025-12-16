import os
from pathlib import Path
import json
import logging
import importlib
import torch

from .dimension import DimensionEvaluator
from utils.video_kit import load_video_to_gpu
from utils.image_kit import load_image_to_gpu
from utils.gpu import clear_gpu_memory

DIMENSION_NAMES = [
    'FrechetVideoDistance',
    'AestheticQuality',
    'ImagingQuality',
    'TemporalFlickering',
    'MotionSmoothness',
    'DynamicDegree',
    'CameraCenteringError',
    'AppearanceConsistency',
    'ViewpointValidity',
    'BackgroundSemanticConsistency',
    'HumanActionAlignment',
    'OpticalFlowCorrelation',
    'TrajectoryAlignment',
    ]
DIMENSION_NAMES_IN_SHORT = ['fvd','aq','iq','tf','ms','dd','cce','ac','vv','bsc','haa','ofc','ta']
DIMENSION_MODULE_MAP = {}
for dn, dnis in zip(DIMENSION_NAMES, DIMENSION_NAMES_IN_SHORT):
    DIMENSION_MODULE_MAP[dn] = dnis
    
class Ego2ExoFollowShotBench():
    def __init__(self,
                 device,
                 path_assets_root='../../assets/', # 所有资源的根目录 包括输入ego,ref,exo_gt
                 ):
        self.device = device
        
        self.path_assets_root = Path(path_assets_root)
        if not self.path_assets_root.exists():
            raise FileNotFoundError(f"Path not found at {self.path_assets_root}.")
        
        self.cache = {
            'Gen': {}, # 生成视频
            'Ego': {}, # Ego视频
            'Exogt':  {}, # GT视频
            'Ref': {} # 参考图片
        }
    
    def _caching(self, matching_map: dict, path_generated_video, target_size):
        logging.info("caching all videos to GPU...")
        total = len(matching_map)
        for idx, (rpath_gen, info) in enumerate(matching_map.items()):
            path_gen = path_generated_video / rpath_gen
            if rpath_gen not in self.cache['Gen']:
                self.cache['Gen'][rpath_gen] = load_video_to_gpu(path_gen, self.device, target_size)

            path_ego = self.path_assets_root / info['Ego']
            if info['Ego'] not in self.cache['Ego']:
                self.cache['Ego'][info['Ego']] = load_video_to_gpu(path_ego, self.device, target_size)
                
            path_gt = self.path_assets_root / info['Exogt']
            if info['Exogt'] not in self.cache['Exogt']:
                self.cache['Exogt'][info['Exogt']] = load_video_to_gpu(path_gt, self.device, target_size)
            
            path_ref = self.path_assets_root / info['Ref']
            if info['Ref'] not in self.cache['Ref']:
                self.cache['Ref'][info['Ref']] = load_image_to_gpu(path_ref, self.device)
                
            if (idx + 1) % 5 == 0:
                logging.info(f"Loaded {idx + 1}/{total} video pairs to GPU.")
        logging.info("All videos cached in VRAM.")
    
    def evaluate(self,
                 path_generated_video, # 用户的生成结果目录
                 path_matching_map_json, # 相对 assets_root 路径
                 path_output, # 保持 json 格式
                 dimension_list=None,
                 target_size=(224, 224),
                 *args,
                 **kwargs):
        # 路径检查
        path_generated_video = Path(path_generated_video)
        if not path_generated_video.exists():
            raise FileNotFoundError(f"Path not found at {path_generated_video}.")
        
        path_matching_map_json = Path(path_matching_map_json)
        with open(path_matching_map_json, 'r') as f: 
            matching_map = json.load(f)
        f.close()
        
        # 缓存
        try:
            self._caching(matching_map, path_generated_video, target_size)
        except torch.cuda.OutOfMemoryError:
            logging.error("OOM during pre-loading! Try reducing video resolution or batch size.")
            torch.cuda.empty_cache()
            return
        
        # 计算阶段
        results= {}
        path_output = Path(path_output)
        for dimension in dimension_list:
            # FVD 特殊处理：跳过单视频循环，直接算全集
            if dimension == 'FrechetVideoDistance':
                logging.info("Calculating FVD for the entire dataset...")
                from dimension.fvd import FrechetVideoDistanceEvaluator
                fvd_eval = FrechetVideoDistanceEvaluator(self.device)
                # 假设 matching_map 里第一个元素的 GT 目录代表了整个 GT 目录
                first_info = list(matching_map.values())[0]
                path_gt_dir = (self.path_assets_root / first_info['Exogt']).parent
                score = fvd_eval.compute_dataset(path_generated_video, path_gt_dir)
                results['FVD'] = score
                continue
            
            try:
                # 动态加载模块
                dimension_module = importlib.import_module(f'dimension.{DIMENSION_MODULE_MAP[dimension]}')
                evaluate_class = getattr(dimension_module, f'{dimension}Evaluator')
                evaluator: DimensionEvaluator = evaluate_class(self.device)
                evaluator.prepare()
                
                for rpath_gen, info in matching_map:
                    tensor_gen = self.cache['Gen'][rpath_gen]
                    tensor_ego = self.cache['Ego'][info['Ego']]
                    tensor_gt  = self.cache['Exogt'][info['Exogt']]
                    tensor_ref = self.cache['Ref'][info['Ref']]
                    
                    res = evaluator.compute(
                            tensor_gen, tensor_ego, tensor_gt, tensor_ref,
                            video_id=rpath_gen,global_cache=self.cache
                        )
                    if rpath_gen not in results: results[rpath_gen] = {}
                    results[rpath_gen][dimension] = res
                    logging.info(f'A new result of {rpath_gen} in {dimension} is {results[rpath_gen][dimension]}!')
            
            except Exception as e:
                logging.warning(f'UnImplemented dimension {dimension}!, {e}')
                continue
            finally:
                evaluator.clear()
                self.cache = {
                    'Gen': {},
                    'Ego': {},
                    'Exogt':  {},
                    'Ref': {}
                }
        
        with open(path_output, 'w') as f:
            json.dump(results, f, indent=2)
        f.close()
    
    def help():
        pass