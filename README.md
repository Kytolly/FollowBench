# Ego2ExowithMotion

News: 更新日志。

Leaderboard: 链接到 HF Space。

Installation: 详细的环境安装指南。

Data Preparation: 如何下载数据、如何使用 build_dataset_json.py。

Getting Started: 如何运行 bench.py，如何提交结果（链接到 competition/README.md）。

Model Zoo: 列出已评测的 Baselines 及其下载链接。

Citation: BibTeX 引用格式。

# TODO LIST
- [ ] 实现 Trajectory Alignment (TA): 提取生成视频和参考视频中人物中心点的轨迹 -> 计算轨迹间的距离 (ADE) -> (可选) 使用 DTW 处理时间对齐。
- [ ] 代码架构重构, 迁移逻辑到 Evaluator 类
- [ ] FVD 兼容性检查
- [ ] 实现 WanVideoDataset: 读取 train.json。 加载 Ego Video, Exo Video, Ref Image。 实现 Dense Captioning Pipeline（如前文讨论，预处理好 descriptive prompts）。
- [ ] 实现训练逻辑： 1. 实现 Spatial Concatenation: 在 Dataset 或 Model Forward 中，将 Ego Latent 和 Exo Latent 拼接。 2. 实现 Masked Loss: 确保 Loss 只在 Exo 区域计算。 3. 集成 ICLora: 编写注入 Cross-Attention 的 Adapter 代码。
- [ ] 编写推理脚本: 除了 ComfyUI 节点外，你需要一个纯 Python 的推理脚本（用于批量生成 Benchmark 视频）。 实现：Load Ego -> Encode -> Concat Noise -> Denoise (w/ Mask) -> Decode
- [ ] Baseline 跑分： 运行 CogVideo, LTX, SVDXT, WanI2V 在你的 Test Set 上生成视频。
- [ ] 可视化: 完善雷达图 轨迹可视化图
- [ ] 数学推理

## git cmomit
类型,含义,示例场景
feat,新功能 (Feature),feat(metric): implement Trajectory Alignment evaluator
fix,修补 Bug,fix(test): resolve AttributeError in lifecycle test
test,测试相关,test(flow): add unit tests for shared caching logic
refactor,代码重构 (不新增功能或修复Bug),refactor(bench): optimize metric calculation pipeline
style,"格式调整 (空格, 分号等)",style: reformat code with black
docs,文档更新,docs: update README with todo list
chore,构建/工具/依赖更新,chore: update .gitignore to exclude .vscode
perf,性能优化,perf(flow): optimize optical flow calculation with batching

## intro
Ego2ExoFollowShot Competition Submission Guide欢迎参加 Ego2ExoFollowShot 挑战赛！为了确保您的模型结果能够被正确评测，请严格遵守以下提交规范。1. 提交格式 (Submission Format)我们支持两种提交方式，推荐使用 方式 B (JSON Index)。方式 A：文件夹结构 (Folder Structure)如果您直接提交文件夹，文件名必须严格包含测试集中的 video_id。Plaintextmy_submission/
├── 1001.mp4          # 对应 ID: 1001
├── 1002_result.mp4   # 对应 ID: 1002 (允许前后缀)
└── ...
方式 B：JSON 索引 (JSON Index) 【推荐】提交一个文件夹，其中包含一个 submission.json 和对应的视频文件（视频文件名随意，只需在 JSON 中对应）。submission.json 结构：JSON{
    "meta": {
        "team_name": "MyTeam",
        "model_name": "EgoGen-V1",
        "contact": "email@example.com"
    },
    "results": {
        "1001": "videos/v1.mp4",
        "1002": "videos/v2.mp4",
        "1003": "videos/v3.mp4"
    }
}
2. 视频规范 (Video Specifications)所有提交的视频必须满足以下技术参数，否则 validate 阶段将报错：属性要求说明分辨率256x256为了公平比较 FVD，请 resize 到此分辨率帧率 (FPS)30 fps必须与 Ego 视频一致时长16 帧 (Clip)评测仅取前 16 帧格式.mp4编码建议 H.264通道RGB3通道彩色3. 评测流程 (Evaluation Pipeline)您的提交将经过以下步骤：完整性检查: 检查是否覆盖了 Test Set 中的所有 1000 个 ID。缺失的 ID 将在各项指标中记为最差分（如 CCE=1.0）。有效性检查: 检查视频是否能被 OpenCV 读取，且无损坏。指标计算:Quality: FVD, AQ, IQControl: CCE, HAA, TAConsistency: AC, BSC, TF最终得分:$$Score_{final} = 0.4 \times S_{quality} + 0.3 \times S_{control} + 0.3 \times S_{consistency}$$4. 如何自测 (Self-Validation)在提交之前，请使用我们的 CLI 工具进行自检：Bash# 1. 安装 benchmark
pip install -e .

ego2exo-submit validate --submission ./my_submission_folder/

ego2exo-submit evaluate --submission ./my_submission_folder/ --device cuda:0


制作一个放在 GitHub README 和 Hugging Face 页面上的演示视频（Demo Video）对于开源项目至关重要，尤其是像 Ego2ExoFollowShot 这种视觉生成类的 Benchmark 工作。好的演示视频应该能在 30秒到2分钟 内讲清楚：输入是什么、输出是什么、难点在哪、你如何评测。基于您的代码库和任务特性，我为您设计了一份视频脚本结构和要素清单：1. 视频核心要素清单 (The Essentials)A. 任务定义 (Task Definition) - 最重要直观展示：屏幕左侧放 Ego View (First-Person) 输入视频，右侧放 Exo View (Third-Person) 生成结果（或 Ground Truth）。输入模态展示：在画面上清晰标注输入三要素：Input Ego Video (第一人称视频)Reference Image (人物参考图，强调 ID 保持)Text Prompt (文本提示词)B. 数据集展示 (Dataset Showcase)多样性：快速剪辑展示不同场景（室内、室外）、不同动作（行走、跳舞、运动）的 Ego-Exo 视频对。处理流程：可以简要展示 video-process-kits 的工作流动画：Raw Video -> VideoCutter -> RIE (Ref Image) -> Final Dataset。C. 评测指标可视化 (Metric Visualization) - 您的核心卖点您的 Benchmark 最大的价值在于指标。不要只列名字，要用画面展示它们在测什么：Control (控制力):HAA (Human Action Alignment): 在生成的 Exo 视频上叠加 OpenPose 骨架，展示其与 Ground Truth 骨架的重合度。CCE (Camera Centering Error): 在人物周围画出 Bounding Box，展示人物是否保持在画面中心。Consistency (一致性):AC (Appearance Consistency): 将 Reference Image 和生成视频中的人物放在一起对比，通过连线或高亮展示衣着、外貌的一致性。Quality (质量):FVD: 展示两组视频，一组是高质量流畅的（低 FVD），一组是抖动模糊的（高 FVD），标注分数差异。D. 排行榜与竞争 (Leaderboard & Competition)展示一个滚动的排行榜列表（Mockup），列出 "Ours", "Baseline A", "Baseline B" 等，强调这是一个竞技平台。展示 pip install 和 python bench.py 的终端录屏，证明易用性。2. 推荐的视频脚本结构 (Storyboard)建议制作两个版本的视频：Short GIF (5-10秒): 放在 README 顶部的 Header。Full Demo (1-2分钟): 放在 Hugging Face 或 README 的 "Introduction" 部分。脚本：Full Demo (1分30秒)时间画面内容字幕/旁白0:00-0:10[Hook] 分屏展示。左边：剧烈抖动的第一人称视角。右边：稳定流畅的第三人称跟随视角。中间出现 Logo: Ego2ExoFollowShot。"Transform First-Person Chaos into Third-Person Cinema."0:10-0:25[Inputs] 屏幕分为三部分：Ego Video + Ref Image + Prompt。箭头指向一个黑盒（Model），然后输出 Exo Video。"The Task: Generate consistent third-person videos conditioned on ego-view, identity image, and text."0:25-0:40[Dataset] 快速混剪 (Montage) 您的数据集片段。展示不同的人物、衣着和环境。"A standardized benchmark dataset with diverse scenarios and rich annotations."0:40-1:00[Metrics - Control] 展示 HAA 和 CCE。视频中人物动起来，骨架（Keypoints）紧紧跟随，Bounding Box 锁定中心。右下角实时跳动 Score 数值。"Precise Evaluation: Measuring Action Alignment (HAA) and Camera Control (CCE)."1:00-1:15[Metrics - Identity] 展示 AC 和 BSC。Ref 图片飞入画面，与视频中的每一帧进行特征匹配的动画示意。"Ensuring Identity and Background Consistency across time."1:15-1:25[Usage] 终端录屏：git clone, download_data, run_bench。随后弹出一个 CSV 成绩单。"Easy to use. One command to evaluate your model."1:25-1:30[Call to Action] GitHub 和 Hugging Face 的链接二维码。"Join the leaderboard. Submit your model today!"3. 制作工具推荐可视化标注: 使用 OpenCV 或 Python 脚本在视频上画框（Bounding Box）和骨架（Skeleton），这比后期剪辑软件画的更精准，也体现技术力。您现有的 visualization.py 应该可以扩充来实现这个功能。剪辑软件: CapCut (剪映) 或 DaVinci Resolve。GIF 转换: 使用 ffmpeg 将 MP4 转为高质量 GIF (用于 GitHub README)。

## docker

步骤 1: 准备文件
确保 Dockerfile, docker-compose.yml 和您的 requirements.txt 在同一个目录（项目根目录）下。

步骤 2: 构建镜像
运行以下命令构建 Docker 镜像：

Bash

docker-compose build
(注意：由于需要编译依赖和安装 PyTorch，首次构建可能需要几分钟时间)

步骤 3: 启动容器
构建完成后，启动容器：

Bash

docker-compose up -d
步骤 4: 进入环境
进入容器内部进行开发或测试：

Bash

docker-compose exec ego2exo bash
4. 关于 flash-attn 的特别说明
您的 requirements.txt 中原本包含一行指向本地文件的 flash-attn 安装命令： flash-attn @ file:///opt/liblibai-models/...

在 Docker 构建过程中，我已经通过 sed 命令移除了这行。原因如下：

路径不可达：Docker 容器无法访问您宿主机的 /opt/liblibai-models/... 路径。

兼容性：在 Dockerfile 末尾，我添加了 pip install flash-attn 尝试从 PyPI 编译安装。如果这一步因为算力匹配问题失败（它会输出 Warning 但不会中断构建），您可以：

进入容器后手动安装。

或者下载对应的 .whl 文件放到项目目录，挂载进去后安装。

5. 常见问题排查
GPU 不可见：请确保宿主机安装了 nvidia-container-toolkit。可以通过运行 docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi 来验证。

ImportError: libGL.so.1：这是 OpenCV 的常见问题，Dockerfile 中的 libgl1-mesa-glx 已经解决了这个问题。


Usage Guide
Ego2ExoFollowShot 支持多种使用方式，您可以根据需求选择作为 Python 库集成、使用命令行工具或直接运行二进制文件。

1. 📦 作为 Python 库使用 (PyPI)
如果您希望将评测功能集成到自己的训练代码或评估脚本中，推荐使用此方式。

安装
Bash

# 从 PyPI 安装 (假设已发布)
pip install ego2exo-bench

# 或者从源码安装
git clone https://github.com/YourUsername/Ego2ExoFollowShot.git
cd Ego2ExoFollowShot
pip install -e .
代码示例
运行完整评测：

Python

from ego2exo.bench import run_evaluation

# 一行代码启动评测
run_evaluation(
    submission_path="./my_model_results",  # 您的生成结果路径
    dataroot="./assets/Ego2ExoDataset",    # Benchmark 数据集路径
    output_dir="./evaluation_results",     # 结果保存路径
    device="cuda"                          # 指定运行设备
)
单独计算某个指标（例如 FVD）：

Python

import torch
from ego2exo.dimension.fvd import FrechetVideoDistanceEvaluator

# 初始化评估器 (自动加载 I3D 模型)
evaluator = FrechetVideoDistanceEvaluator(device="cuda")
evaluator.prepare()

# 准备数据 [T, C, H, W]
gen_video = torch.rand(16, 3, 224, 224).cuda()
gt_video = torch.rand(16, 3, 224, 224).cuda()

# 计算分数
score = evaluator.compute(
    tensor_gen=gen_video,
    tensor_gt=gt_video,
    video_id="test_sample_001"
)
print(f"FVD Score: {score}")
2. 💻 命令行工具 (CLI)
适合在终端、Shell 脚本或 CI/CD 流程中快速调用。安装库后，系统会自动注册 ego2exo 命令。

基本命令
Bash

# 查看帮助
ego2exo --help
核心功能
1. 提交格式校验 (Validate) 在跑分之前，检查您的提交文件格式是否符合规范。

Bash

ego2exo validate --submission ./my_submission_folder/
2. 运行评测 (Evaluate) 运行所有指标并生成 CSV 报告。

Bash

ego2exo evaluate \
    --submission ./my_submission_folder/ \
    --dataroot ./assets/Ego2ExoDataset \
    --output ./results \
    --device cuda:0
3. 下载/准备资源 (Setup) 自动下载权重文件和数据集。

Bash

ego2exo setup --download-weights --download-data
3. 🚀 独立二进制文件 (Standalone Binary)
如果您不想配置 Python 环境（例如在纯净的生产环境或非技术人员的机器上），可以直接下载编译好的二进制文件。

下载
请前往 Releases 页面下载对应系统的版本：

🐧 Linux: ego2exo-linux-x86_64

🪟 Windows: ego2exo-win64.exe

🍎 macOS: ego2exo-macos-arm64

使用方法
二进制文件的参数与 CLI 完全一致。

Linux/macOS:

Bash

# 赋予执行权限
chmod +x ego2exo-linux-x86_64

# 运行评测
./ego2exo-linux-x86_64 evaluate --submission ./results/ --device cpu
Windows (PowerShell/CMD):

PowerShell

.\ego2exo-win64.exe evaluate --submission .\results\ --device cuda
🛠️ 高级配置 (Advanced)
您可以通过环境变量或配置文件覆盖默认行为。

环境变量：

EGO2EXO_HOME: 指定模型权重和缓存的根目录（默认 ~/.cache/ego2exo）。

HF_ENDPOINT: 如果在国内，可设置为镜像站加速下载。

示例：

Bash

export EGO2EXO_HOME="/data/shared_models"
ego2exo evaluate ...
下一步建议
为了实现上述愿景，您需要在 setup.py 或 pyproject.toml 中正确配置入口点（Entry Points），以便用户安装后能直接使用 ego2exo 命令：