import sys
from pathlib import Path

import torch

from egoexo_translation_bench.utils.gpu import get_free_gpu_ids

def print_env():
    print("Python Version:", sys.version)
    print("PyTorch CUDA Version:", torch.version.cuda)
    print("PyTorch Version:", torch.__version__)
    free_gpus = get_free_gpu_ids()
    print(free_gpus)
    
def get_project_root():
    if getattr(sys, 'frozen', False):
        
        # 方案 A: 如果你打算把 configs 打包进 exe 内部，使用:
        return Path(sys._MEIPASS)
        
        # 方案 B: 如果你希望 configs 放在 exe 旁边供用户修改，使用:
        # return Path(sys.executable).parent
    else:
        # 正常 python 运行
        return Path(__file__).resolve().parent.parent

PROJECT_ROOT = get_project_root()