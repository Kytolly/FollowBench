@echo off
REM build_env.bat - Windows环境配置脚本
REM 基于 build_env.sh 转换而来，与 build_env_windows.ps1 功能一致

echo === 开始配置 Windows 环境 ===

REM 1. 清理旧环境
echo 正在清理旧环境 bench...
call conda deactivate 2>nul
call conda remove -n bench --all -y
if errorlevel 1 (
    echo 移除环境时遇到问题或环境不存在，继续...
)

REM 2. 创建并激活环境
echo 创建 conda 环境 python=3.10...
call conda init
call conda create -n bench python=3.10 -y
call conda activate bench

REM 3. 安装 PyTorch 12.4 工具包
echo 安装 PyTorch 12.4...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

echo 安装 CUDA Toolkit...
call conda install -c "nvidia/label/cuda-12.4.0" cuda-toolkit -y

pip install ninja

REM 设置 CUDA_HOME 环境变量
set CUDA_HOME=%CONDA_PREFIX%
echo CUDA_HOME 设置为: %CUDA_HOME%

REM 检查 NVCC 版本
nvcc --version

REM 4. 构建基础依赖
echo 安装基础依赖...
pip install -U openmim
call mim install "mmcv-full==1.7.2"
pip install "mmdet>=2.28.0"
pip install "numpy<1.24"
pip install "timm>=0.4.9"
pip install "xtcocotools>=1.8"
pip install "git+https://github.com/mattloper/chumpy.git" --no-build-isolation
pip install "setuptools==60.2.0"
pip install yacs joblib scikit-image opencv-python imageio[ffmpeg] matplotlib 
pip install tensorboard smplx progress einops munkres loguru tqdm ultralytics gdown

REM 5. 设置第三方依赖目录
echo 设置第三方依赖...
if not exist "third-party" mkdir "third-party"
cd third-party

echo 克隆 WHAM 仓库...
if exist "WHAM" rmdir /s /q "WHAM"
git clone https://github.com/Kytolly/WHAM.git --recursive
cd WHAM

REM 6. 安装 DPVO
echo 配置 DPVO...
cd third-party\DPVO

if not exist "thirdparty\eigen-3.4.0" (
    echo 下载 Eigen 3.4.0...
    powershell -Command "Invoke-WebRequest -Uri 'https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip' -OutFile 'eigen-3.4.0.zip'"
    powershell -Command "Expand-Archive -Path 'eigen-3.4.0.zip' -DestinationPath 'thirdparty' -Force"
    del eigen-3.4.0.zip
)

REM Windows 编译环境设置
echo 设置编译环境变量...
set TORCH_CUDA_ARCH_LIST=9.0
set CXXFLAGS=-w

REM 在Windows上设置包含路径和库路径
set INCLUDE=%CONDA_PREFIX%\include;%INCLUDE%
set LIB=%CONDA_PREFIX%\lib;%LIB%
set PATH=%CONDA_PREFIX%\lib;%CONDA_PREFIX%\bin;%PATH%

echo 安装 torch-scatter...
pip install torch-scatter --no-build-isolation

echo 编译安装 DPVO...
pip install . --no-build-isolation

REM 回到项目根目录
cd ..\..

REM 7. 下载权重文件
echo 创建 checkpoints 目录...
if not exist "checkpoints" mkdir "checkpoints"

echo 下载预训练模型权重...

echo   - 下载 WHAM ViT 模型...
gdown "https://drive.google.com/uc?id=1i7kt9RlCCCNEW2aYaDWVr-G778JkLNcB&export=download&confirm=t" -O checkpoints/wham_vit_w_3dpw.pth.tar

echo   - 下载 WHAM ViT BEDLAM 模型...
gdown "https://drive.google.com/uc?id=19qkI-a6xuwob9_RFNSPWf1yWErwVVlks&export=download&confirm=t" -O checkpoints/wham_vit_bedlam_w_3dpw.pth.tar

echo   - 下载 HMR2A 模型...
gdown "https://drive.google.com/uc?id=1J6l8teyZrL0zFzHhzkC7efRhU0ZJ5G9Y&export=download&confirm=t" -O checkpoints/hmr2a.ckpt

echo   - 下载 DPVO 模型...
gdown "https://drive.google.com/uc?id=1kXTV4EYb-BI3H7J-bkR3Bc4gT9zfnHGT&export=download&confirm=t" -O checkpoints/dpvo.pth

echo   - 下载 YOLOv8 模型...
gdown "https://drive.google.com/uc?id=1zJ0KP23tXD42D47cw1Gs7zE2BA_V_ERo&export=download&confirm=t" -O checkpoints/yolov8x.pt

echo   - 下载 ViTPose 模型...
gdown "https://drive.google.com/uc?id=1xyF7F3I7lWtdq82xmEPVQ5zl4HaasBso&export=download&confirm=t" -O checkpoints/vitpose-h-multi-coco.pth

REM 8. 安装 ViTPose
echo 安装 ViTPose...
pip install -v -e third-party/ViTPose

REM 9. 安装 WHAM
echo 安装 WHAM...
pip install -e .

REM 回到项目根目录
cd ..

REM 10. 安装 flash-attn (Windows版本)
echo 安装 flash-attention...
pip install flash-attn --no-build-isolation
if errorlevel 1 (
    echo 警告: flash-attn 安装失败，可能需要手动安装
)

REM 11. 安装其余依赖
echo 安装项目依赖...
pip install -r requirements.txt

echo === 环境构建完成! ===
echo.
echo 使用说明:
echo 1. 确保你在 Anaconda Command Prompt 中运行此脚本
echo 2. 如果遇到编译错误，请确保安装了 Visual Studio Build Tools
echo 3. 激活环境: conda activate bench
echo 4. 如果 flash-attn 安装失败，可以稍后手动安装
echo 5. 详细说明请参考 docs/windows_setup.md

pause
