# build_env_windows.ps1 - Windows环境配置脚本
# 基于 build_env.sh 转换而来

# 设置错误时停止执行
$ErrorActionPreference = "Stop"

Write-Host "=== 开始配置 Windows 环境 ===" -ForegroundColor Green

# 1. 清理旧环境
Write-Host "正在清理旧环境 bench..." -ForegroundColor Yellow
try { 
    conda deactivate 
} catch { 
    Write-Host "当前没有激活的conda环境" -ForegroundColor Gray
}

conda remove -n bench --all -y
if ($LASTEXITCODE -ne 0) { 
    Write-Host "移除环境时遇到问题或环境不存在，继续..." -ForegroundColor Gray
}

# 2. 创建并激活环境
Write-Host "创建 conda 环境 python=3.10..." -ForegroundColor Yellow
conda init
conda create -n bench python=3.10 -y

Write-Host "激活环境 bench..." -ForegroundColor Yellow
conda activate bench

# 3. 安装 PyTorch 12.4 工具包
Write-Host "安装 PyTorch 12.4..." -ForegroundColor Yellow
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

Write-Host "安装 CUDA Toolkit..." -ForegroundColor Yellow
conda install -c "nvidia/label/cuda-12.4.0" cuda-toolkit -y

pip install ninja

# 设置 CUDA_HOME 环境变量
$env:CUDA_HOME = $env:CONDA_PREFIX
Write-Host "CUDA_HOME 设置为: $env:CUDA_HOME" -ForegroundColor Green

# 检查 NVCC 版本
nvcc --version

# 4. 构建基础依赖
Write-Host "安装基础依赖..." -ForegroundColor Yellow
pip install -U openmim
mim install "mmcv-full==1.7.2"
pip install "mmdet>=2.28.0"
pip install "numpy<1.24"
pip install "timm>=0.4.9"
pip install "xtcocotools>=1.8"
pip install "git+https://github.com/mattloper/chumpy.git" --no-build-isolation
pip install "setuptools==60.2.0"
pip install yacs joblib scikit-image opencv-python imageio[ffmpeg] matplotlib 
pip install tensorboard smplx progress einops munkres loguru tqdm ultralytics gdown

# 5. 设置第三方依赖目录
Write-Host "设置第三方依赖..." -ForegroundColor Yellow
if (-not (Test-Path "third-party")) {
    New-Item -ItemType Directory -Path "third-party"
}
Set-Location "third-party"

Write-Host "克隆 WHAM 仓库..." -ForegroundColor Yellow
if (Test-Path "WHAM") {
    Remove-Item -Recurse -Force "WHAM"
}
git clone https://github.com/Kytolly/WHAM.git --recursive

Set-Location "WHAM"

# 6. 安装 DPVO
Write-Host "配置 DPVO..." -ForegroundColor Yellow
Set-Location "third-party/DPVO"

if (-not (Test-Path "thirdparty/eigen-3.4.0")) {
    Write-Host "下载 Eigen 3.4.0..." -ForegroundColor Gray
    Invoke-WebRequest -Uri "https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip" -OutFile "eigen-3.4.0.zip"
    Expand-Archive -Path "eigen-3.4.0.zip" -DestinationPath "thirdparty" -Force
    Remove-Item "eigen-3.4.0.zip"
}

# Windows 编译环境设置
Write-Host "设置编译环境变量..." -ForegroundColor Yellow
$env:TORCH_CUDA_ARCH_LIST = "9.0"
$env:CXXFLAGS = "-w"

# 在Windows上设置包含路径和库路径
if ($env:INCLUDE) {
    $env:INCLUDE = "$env:CONDA_PREFIX\include;$env:INCLUDE"
} else {
    $env:INCLUDE = "$env:CONDA_PREFIX\include"
}

if ($env:LIB) {
    $env:LIB = "$env:CONDA_PREFIX\lib;$env:LIB"
} else {
    $env:LIB = "$env:CONDA_PREFIX\lib"
}

# 确保DLL能被找到
$env:PATH = "$env:CONDA_PREFIX\lib;$env:CONDA_PREFIX\bin;$env:PATH"

Write-Host "安装 torch-scatter..." -ForegroundColor Yellow
pip install torch-scatter --no-build-isolation

Write-Host "编译安装 DPVO..." -ForegroundColor Yellow
pip install . --no-build-isolation

# 回到项目根目录
Set-Location "..\.."

# 7. 下载权重文件
Write-Host "创建 checkpoints 目录..." -ForegroundColor Yellow
if (-not (Test-Path "checkpoints")) {
    New-Item -ItemType Directory -Path "checkpoints"
}

Write-Host "下载预训练模型权重..." -ForegroundColor Yellow

Write-Host "  - 下载 WHAM ViT 模型..." -ForegroundColor Gray
gdown "https://drive.google.com/uc?id=1i7kt9RlCCCNEW2aYaDWVr-G778JkLNcB&export=download&confirm=t" -O 'checkpoints/wham_vit_w_3dpw.pth.tar'

Write-Host "  - 下载 WHAM ViT BEDLAM 模型..." -ForegroundColor Gray
gdown "https://drive.google.com/uc?id=19qkI-a6xuwob9_RFNSPWf1yWErwVVlks&export=download&confirm=t" -O 'checkpoints/wham_vit_bedlam_w_3dpw.pth.tar'

Write-Host "  - 下载 HMR2A 模型..." -ForegroundColor Gray
gdown "https://drive.google.com/uc?id=1J6l8teyZrL0zFzHhzkC7efRhU0ZJ5G9Y&export=download&confirm=t" -O 'checkpoints/hmr2a.ckpt'

Write-Host "  - 下载 DPVO 模型..." -ForegroundColor Gray
gdown "https://drive.google.com/uc?id=1kXTV4EYb-BI3H7J-bkR3Bc4gT9zfnHGT&export=download&confirm=t" -O 'checkpoints/dpvo.pth'

Write-Host "  - 下载 YOLOv8 模型..." -ForegroundColor Gray
gdown "https://drive.google.com/uc?id=1zJ0KP23tXD42D47cw1Gs7zE2BA_V_ERo&export=download&confirm=t" -O 'checkpoints/yolov8x.pt'

Write-Host "  - 下载 ViTPose 模型..." -ForegroundColor Gray
gdown "https://drive.google.com/uc?id=1xyF7F3I7lWtdq82xmEPVQ5zl4HaasBso&export=download&confirm=t" -O 'checkpoints/vitpose-h-multi-coco.pth'

# 8. 安装 ViTPose
Write-Host "安装 ViTPose..." -ForegroundColor Yellow
pip install -v -e third-party/ViTPose

# 9. 安装 WHAM
Write-Host "安装 WHAM..." -ForegroundColor Yellow
pip install -e .

# 回到项目根目录
Set-Location ".."

# 10. 安装 flash-attn (Windows版本)
Write-Host "安装 flash-attention..." -ForegroundColor Yellow
# 注意：Windows版本的flash-attention wheel文件不同
# 这里尝试安装Windows兼容版本
try {
    # 首先尝试从PyPI安装
    pip install flash-attn --no-build-isolation
}
catch {
    Write-Host "从PyPI安装flash-attn失败，尝试其他方法..." -ForegroundColor Yellow
    # 如果失败，可以尝试其他安装方法或跳过
    Write-Host "警告: flash-attn 安装失败，可能需要手动安装" -ForegroundColor Red
}

# 11. 安装其余依赖
Write-Host "安装项目依赖..." -ForegroundColor Yellow
pip install -r requirements.txt

Write-Host "=== 环境构建完成! ===" -ForegroundColor Green
Write-Host ""
Write-Host "使用说明:" -ForegroundColor Cyan
Write-Host "1. 确保你在 Anaconda PowerShell Prompt 中运行此脚本" -ForegroundColor White
Write-Host "2. 如果遇到编译错误，请确保安装了 Visual Studio Build Tools" -ForegroundColor White
Write-Host "3. 激活环境: conda activate bench" -ForegroundColor White
Write-Host "4. 如果 flash-attn 安装失败，可以稍后手动安装" -ForegroundColor White
