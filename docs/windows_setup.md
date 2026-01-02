# Windows 环境配置指南

本文档提供了在Windows系统上配置EgoExoTranslationBench环境的详细说明。

## 前置要求

在开始之前，请确保您的系统满足以下要求：

### 1. 硬件要求
- NVIDIA GPU（支持CUDA 12.4）
- 至少16GB RAM
- 至少50GB可用磁盘空间

### 2. 软件要求
- Windows 10/11 (64位)
- Anaconda 或 Miniconda
- Git
- Visual Studio Build Tools 2019/2022 (用于编译C++扩展)
- CUDA 12.4 驱动程序

### 3. 安装前置软件

#### 安装 Anaconda
1. 从 [Anaconda官网](https://www.anaconda.com/products/distribution) 下载并安装
2. 安装时选择"Add Anaconda to PATH"选项

#### 安装 Visual Studio Build Tools
1. 从 [Microsoft官网](https://visualstudio.microsoft.com/visual-cpp-build-tools/) 下载
2. 安装时选择"C++ build tools"工作负载
3. 确保包含 MSVC v143 编译器工具

#### 安装 Git
1. 从 [Git官网](https://git-scm.com/download/win) 下载并安装
2. 安装时选择默认设置即可

#### 安装 CUDA 驱动
1. 从 [NVIDIA官网](https://developer.nvidia.com/cuda-12-4-0-download-archive) 下载CUDA 12.4
2. 按照安装向导完成安装

## 环境配置方法

我们提供了两种配置方法：

### 方法1：使用PowerShell脚本（推荐）

1. 打开 **Anaconda PowerShell Prompt**（以管理员身份运行）
2. 导航到项目目录：
   ```powershell
   cd path\to\EgoExoTranslationBench
   ```
3. 运行配置脚本：
   ```powershell
   .\scripts\build_env_windows.ps1
   ```

### 方法2：使用批处理脚本

1. 打开 **Anaconda Command Prompt**（以管理员身份运行）
2. 导航到项目目录：
   ```cmd
   cd path\to\EgoExoTranslationBench
   ```
3. 运行配置脚本：
   ```cmd
   scripts\build_env_simple.bat
   ```

## 脚本功能说明

配置脚本会自动完成以下步骤：

1. **环境清理**：删除已存在的 `bench` 环境
2. **创建环境**：创建新的 Python 3.10 conda 环境
3. **安装PyTorch**：安装支持CUDA 12.4的PyTorch
4. **安装CUDA工具**：安装CUDA toolkit和相关工具
5. **安装依赖**：安装所有必需的Python包
6. **克隆仓库**：下载WHAM等第三方库
7. **编译DPVO**：编译深度视觉里程计模块
8. **下载模型**：下载预训练模型权重
9. **安装项目**：安装ViTPose、WHAM和项目本身

## 常见问题解决

### 1. 编译错误
如果遇到C++编译错误：
- 确保安装了Visual Studio Build Tools
- 重启命令提示符
- 检查环境变量是否正确设置

### 2. CUDA相关错误
如果遇到CUDA相关问题：
- 检查NVIDIA驱动是否为最新版本
- 确认GPU支持CUDA 12.4
- 重新安装CUDA toolkit

### 3. 网络下载问题
如果下载模型失败：
- 检查网络连接
- 尝试使用VPN
- 手动下载模型文件到 `checkpoints` 目录

### 4. flash-attention安装失败
这是常见问题，可以：
- 忽略此错误，项目仍可正常运行
- 稍后手动安装：`pip install flash-attn`
- 使用预编译版本

## 验证安装

安装完成后，可以通过以下步骤验证：

1. 激活环境：
   ```cmd
   conda activate bench
   ```

2. 检查PyTorch和CUDA：
   ```python
   python -c "import torch; print(torch.cuda.is_available()); print(torch.version.cuda)"
   ```

3. 运行测试：
   ```cmd
   python -m pytest tests/
   ```

## 手动安装步骤

如果自动脚本失败，可以按照以下步骤手动安装：

### 1. 创建环境
```cmd
conda create -n bench python=3.10 -y
conda activate bench
```

### 2. 安装PyTorch
```cmd
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### 3. 安装基础依赖
```cmd
pip install -U openmim
mim install "mmcv-full==1.7.2"
pip install "mmdet>=2.28.0"
pip install "numpy<1.24"
pip install -r requirements.txt
```

### 4. 克隆和安装第三方库
```cmd
mkdir third-party
cd third-party
git clone https://github.com/Kytolly/WHAM.git --recursive
cd WHAM
pip install -e .
```

## 性能优化建议

1. **使用SSD**：将项目放在SSD上以提高I/O性能
2. **增加虚拟内存**：如果RAM不足，增加虚拟内存大小
3. **关闭杀毒软件**：编译时暂时关闭实时保护
4. **使用多核编译**：设置 `MAX_JOBS` 环境变量

## 故障排除

如果遇到问题，请：

1. 检查错误日志
2. 确认所有前置要求已满足
3. 尝试重新运行脚本
4. 查看项目的GitHub Issues
5. 联系项目维护者

## 卸载

如果需要完全卸载环境：

```cmd
conda deactivate
conda remove -n bench --all -y
```

然后删除 `third-party` 目录和下载的模型文件。
