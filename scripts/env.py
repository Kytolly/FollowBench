import os
import sys
import torch
from src.utils.gpu import get_free_gpu_ids
import yaml
from configs import CONFIG

def print_env():
    print("Python Version:", sys.version)
    print("PyTorch CUDA Version:", torch.version.cuda)
    print("PyTorch Version:", torch.__version__)
    free_gpus = get_free_gpu_ids()
    print(free_gpus)