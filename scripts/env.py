import os
import sys
import torch
from src.utils.gpu import get_free_gpu_ids
import yaml

def get_env_config(mode):
    with open('config/env.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    f.close()
    return config[mode]

def print_env():
    print("Python Version:", sys.version)
    print("PyTorch CUDA Version:", torch.version.cuda)
    print("PyTorch Version:", torch.__version__)
    free_gpus = get_free_gpu_ids()
    print(free_gpus)