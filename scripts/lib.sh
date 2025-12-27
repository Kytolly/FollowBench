conda create -n bench python=3.10 -y
source activate bench

cd third-party

cd WHAM
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -U openmim
mim install "mmcv-full==1.7.2"
pip install "mmdet>=2.28.0"
pip install "numpy<1.24"
pip install "timm>=0.4.9"
pip install "xtcocotools>=1.8"
pip install yacs joblib scikit-image opencv-python imageio[ffmpeg] matplotlib 
pip install tensorboard smplx progress einops  munkres loguru tqdm ultralytics gdown

# install DPVO
cd third-party/DPVO
if [ ! -d "thirdparty/eigen-3.4.0" ]; then
    wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip
    unzip eigen-3.4.0.zip -d thirdparty 
    rm eigen-3.4.0.zip
fi
conda install -c "nvidia/label/cuda-12.4.0" cuda-toolkit -y
pip install ninja
export CUDA_HOME=$CONDA_PREFIX
nvcc --version

rm -rf build/ dist/ *.egg-info
export TORCH_CUDA_ARCH_LIST="9.0"
pip install . --no-build-isolation
cd ../..

# instal;l ViTPose
cd third-party/ViTPose
pip install -v -e .
cd ../..

# install WHAM
echo ">>> Installing WHAM..."
pip install -e .