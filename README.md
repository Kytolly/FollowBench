# Ego2ExowithMotion
My final year project repo

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