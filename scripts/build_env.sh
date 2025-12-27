#!/bin/bash
set -e # 遇到错误立即停止
conda deactivate
conda remove -n bench --all -y

# 构建 bench 环境
conda init
conda create -n bench python=3.10 -y
source activate bench

# 安装 pytorch 12.4 工具包
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
conda install -c "nvidia/label/cuda-12.4.0" cuda-toolkit -y
pip install ninja
export CUDA_HOME=$CONDA_PREFIX
nvcc --version

# 构建基础依赖
pip install -U openmim
mim install "mmcv-full==1.7.2"
pip install "mmdet>=2.28.0"
pip install "numpy<1.24"
pip install "timm>=0.4.9"
pip install "xtcocotools>=1.8"
pip install "git+https://github.com/mattloper/chumpy.git" --no-build-isolation
pip install "setuptools==60.2.0"
pip install yacs joblib scikit-image opencv-python imageio[ffmpeg] matplotlib 
pip install tensorboard smplx progress einops  munkres loguru tqdm ultralytics gdown

# 设置第三方依赖目录
mkdir -p third-party
cd third-party
git clone https://github.com/Kytolly/WHAM.git --recursive # 检查代码是否升级
cd WHAM

# 安装 DPVO
cd third-party/DPVO
if [ ! -d "thirdparty/eigen-3.4.0" ]; then
    wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip
    unzip eigen-3.4.0.zip -d thirdparty 
    rm eigen-3.4.0.zip
fi
# conda install -y -c nvidia \
#     cuda-nvcc=12.4 \
#     cuda-libraries-dev=12.4 \
#     cuda-cudart-dev=12.4 \
#     cuda-cccl=12.4
conda install -y -c conda-forge gxx=11.4
export CUDA_HOME=$CONDA_PREFIX
export TORCH_CUDA_ARCH_LIST="9.0"
export CXXFLAGS="-w"
# export CPATH=$CONDA_PREFIX/include:$CPATH
# export LIBRARY_PATH=$CONDA_PREFIX/lib:$LIBRARY_PATH
# export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
pip install torch-scatter --no-build-isolation
pip install . --no-build-isolation
cd ../..

# 下载权重
mkdir checkpoints
echo "下载预训练模型权重..."
echo "  - 下载 WHAM ViT 模型..."
gdown "https://drive.google.com/uc?id=1i7kt9RlCCCNEW2aYaDWVr-G778JkLNcB&export=download&confirm=t" -O 'checkpoints/wham_vit_w_3dpw.pth.tar'
echo "  - 下载 WHAM ViT BEDLAM 模型..."
gdown "https://drive.google.com/uc?id=19qkI-a6xuwob9_RFNSPWf1yWErwVVlks&export=download&confirm=t" -O 'checkpoints/wham_vit_bedlam_w_3dpw.pth.tar'
echo "  - 下载 HMR2A 模型..."
gdown "https://drive.google.com/uc?id=1J6l8teyZrL0zFzHhzkC7efRhU0ZJ5G9Y&export=download&confirm=t" -O 'checkpoints/hmr2a.ckpt'
echo "  - 下载 DPVO 模型..."
gdown "https://drive.google.com/uc?id=1kXTV4EYb-BI3H7J-bkR3Bc4gT9zfnHGT&export=download&confirm=t" -O 'checkpoints/dpvo.pth'
echo "  - 下载 YOLOv8 模型..."
gdown "https://drive.google.com/uc?id=1zJ0KP23tXD42D47cw1Gs7zE2BA_V_ERo&export=download&confirm=t" -O 'checkpoints/yolov8x.pt'
echo "  - 下载 ViTPose 模型..."
gdown "https://drive.google.com/uc?id=1xyF7F3I7lWtdq82xmEPVQ5zl4HaasBso&export=download&confirm=t" -O 'checkpoints/vitpose-h-multi-coco.pth'

# 安装 ViTPose
pip install -v -e third-party/ViTPose

# 安装 WHAM
pip install -e .

# 安装 flash-attn
cd ..
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.1.post1/flash_attn-2.7.1.post1+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# 安装其余依赖
pip install -r requirements.txt

echo "=== 环境构建完成! ==="