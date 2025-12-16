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