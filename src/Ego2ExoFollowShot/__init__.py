import os
from pathlib import Path
import json
import logging
import importlib

from dimension import DimensionEvaluator

DIMENSION_NAMES = [
    'Frechet Video Distance',
    'Aesthetic Quality',
    'Imaging Quality',
    'Temporal Flickering',
    'Motion Smoothness',
    'Dynamic Degree',
    'Camera Centering Error',
    'Appearance Consistency',
    'Viewpoint Validity',
    'Background Semantic Consistency',
    'Human Action Alignment',
    'Optical Flow Correlation',
    'Trajectory Alignment',
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

    def evaluate(self,
                 path_generated_video, # 用户的生成结果目录
                 path_matching_map_json, # 相对 assets_root 路径
                 path_output, # 保持 json 格式
                 dimension_list=None,  
                 **kwargs):
        results= {}
        
        path_generated_video = Path(path_generated_video)
        if not path_generated_video.exists():
            raise FileNotFoundError(f"Path not found at {path_generated_video}.")
        
        path_matching_map_json = Path(path_matching_map_json)
        with open(path_matching_map_json, 'r') as f:
            matching_map = json.load(f)
        f.close()
        
        path_output = Path(path_output)
        for dimension in dimension_list:
            try:
                dimension_module = importlib.import_module(f'.dimension.{dimension}')
                evaluate_class = getattr(dimension_module, f'{dimension}Evaluator')
                evaluator: DimensionEvaluator = evaluate_class()
            except Exception as e:
                logging.warning(f'UnImplemented dimension {dimension}!, {e}')
                continue
            
            for rpath_gen, info in matching_map:
                path_gen = path_generated_video / rpath_gen
                path_ego = self.path_assets_root / info['Ego']
                path_exogt = self.path_assets_root / info['Exogt']
                path_ref = self.path_assets_root / info['ref']
                res = evaluator.run(path_gen, path_ego, path_exogt, path_ref)
                if rpath_gen not in results: 
                    results[rpath_gen] = {}
                results[rpath_gen][dimension] = res
                logging.info(f'A new result of {rpath_gen} in {dimension} is {results[rpath_gen][dimension]}!')
        
        with open(path_output, 'w') as f:
            json.dump(results, f, indent=2)
        f.close()
    
    def help():
        pass