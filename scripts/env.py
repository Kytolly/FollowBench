import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

import pynvml
import torch
from utils.gpu import get_free_gpu_ids

print("Python Version:", sys.version)
print("PyTorch CUDA Version:", torch.version.cuda)
print("PyTorch Version:", torch.__version__)
free_gpus = get_free_gpu_ids()
print(free_gpus)