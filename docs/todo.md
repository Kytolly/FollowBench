## Questions

- [ ] 在benchmark中给出的切片参数：需要当作benchmark的规则被发布，例如5s,60fps。所有的模型生成的视频需要和给出的ground truth 在帧数和帧率对齐，所以对于不能对齐的baseline，需要通过自回归补帧，self-forcing 在训练中的作用是什么？

- [ ] 划分的具体方案

- [x] 根据之前和师兄的讨论结果，项目主要着手解决一个Ego-centric视频到一个（和现有数据集描述的Exo-centric概念不同的，可以看作固定相机的监控，我认为是一种）第三人称的follow camera 镜头生成。但是怎么样和现有的Ego2Exo任务区分开来（它们拥有相同的输入条件和输出条件，比如Intention driven模型，但是我们的任务明显有更多的光流变化和动作语义识别，属于某种几何上的映射和重绘）来描述这种新任务呢？我们应该引入某种对相机的行为约束吗？

  **读论文 EgoExoTranslation**

- [x] 在实现这个项目的过程中，一个重要的课题是人物身份特征的一致性。我们将同一个身份定义为穿着同一套服装的同一个人，并以id称呼。我们在讨论之后认为，同一个id绝不能在训练集和测试集中同时出现，否则我们训练的模型是依据参考图中的id进行视频生成的这一点可能会被质疑。这意味着我们需要根据id对数据进行聚类划分。然而这可能导致数据不平衡问题：在我们收集的数据中，大部分的视频（占比约60%-80%）的拍摄对象都是我和我的同伴两人。尽管我们邀请了我们的同学、家人等参与了视频录制，但以他们为主角的视频占比仍然很少。对于这个数据不平衡问题，我们没有想出适合的解决方案。

  **同一个ID也是可以的，是personalized的setting** 

- [x] 任务书中有提到Dataset Preparation一项，要求是Use synthetic or public datasets containing egocentric and third-person video pairs。我们并没有寻找到适合本课题使用的公开数据集，因此，在早期和师兄的讨论之后，我们现在采集数据的方式固定为真人实地拍摄。请问这个方式符合您的预期吗？另外，单纯人工拍摄可能比较缓慢，我们是否需要探究其他的数据获取方式，如游戏录制？此外，使用不同渠道获取的数据对模型进行训练，大概率会损害模型表现，我们是否需要冒这个风险？

  **人工可以，或者可以结合使用EgoExo4D，游戏如果比较真实的话也可以** 已经ban了for CA1


- [x] 在录制视频的过程中，ego视频是通过固定在拍摄者胸前的相机进行采集的。这导致第一人称视频中显著缺少关于人物头部动作的信息，我们在后续的训练中，是否应该着重注意这方面的潜在信息，对其进行深入挖掘？您有什么方法建议吗？

  **看你们兴趣，如果重点研究这个头部信息的话是个有意思的方向，但也不是必须要的，如果视频cover hand-object interaction很好的话**


- [ ] 参考图片为首帧？

  关于参考图片和评估指标，我们也有一些问题。现有的评估指标中，有一部分需要依赖模型输出和ground truth视频之间像素级的相似度，对模型表现的优秀程度进行衡量。然而，CA2要求模型拥有“替换”生成的视频中主角id的能力，也就是说，模型生成的视频和ground truth视频之间在像素层级上的相似度会很低，更多的是动作层面的相似。我们的问题是，在接下来的训练中，我们应不应该更改现有的工作流程，从①变为②？

  ①：输入exo视频首帧作为参考图像（与大多数本方向论文相同），保持现有的评估指标

  ②：在训练集和测试集的参考图片中混入不同来源的人物背面照作为参考图片，并去掉衡量像素级相似度的指标，只衡量人物的一致性，以及动作的相似性。

- [x] CA1要求我们能有基础的函数实现。我目前跑了一些通用模型的baseline推理，它们的效果确实如预期一样差。但是很遗憾，对于我们提出的enhancement模型（基于Wan/VACE的self-forcing和In Context lora微调）的工作还没有开始，它的正式训练可能得等到所有数据收集完毕。这已经是现有的benchmark工程剩余的几项TODO之一。我们的代码进度还是可观的，但是从实验数据上来看，实质上的突破还暂时没有。想问问Mike老师，一月初来研究院听我们汇报的时候，希望听到的进度是怎么样的？根据之前的讨论结果，我们把项目的重心放在了benchmark的构建上。我们的工作重心和您的预期之间是否存在偏差？

  **如果把benchmark本身做的很完善的话，可以先focus在benchmark，baseline方面可以先只是跑一些简单的setting，比如针对某个固定场景**

## Docs

- [ ] World Wander 论文读完
- [ ] ysy's ppt: intro，related work，future work
- [ ] xqy's ppt: method，Experiments
- [ ] 可视化雷达图，折线图柱状图，生成的视频，原视频参考图；前端展示
- [ ] 贡献分配for CA1
  1. [benchmark](https://github.com/Kytolly/Ego2ExowithMotion.git) xqy
  2. [baselines](https://github.com/Kytolly/Baselines-VDMs.git) xqy
  3. [Self-forcing](https://github.com/Kytolly/Self-Forcing.git) ysy
  4. [video-process-kits](https://github.com/Kytolly/video-process-kits.git) ysy

## Preprocess

- [ ] 数据集构建 **DDL**：1月1日前 **时长** 20h/50h 
- [ ] 按**场景**划分训练集合和测试集合：切片，随机取样，检查视频不完全重复；
- [ ] 破解OSV批处理脚本；
- [x] 自动化数据清洗管道

## Baselines

- [ ] AnyV2V 
- [ ] I2VEdit 
- [ ] TrajectoryCrafter 
- [ ] ReCamMaster 
- [ ] WorldWander
- [ ] 在WorldWander 上套selfforcing wrapper；
- [ ] 基于Ipadapter 是不是要看ground truth跑一个oracle model
- [ ] 生成 Baseline 视频跑分

## Benchmark

- [ ] 核心算法与度量
- [x] 设计数据集标准格式和提交规范
- [x] 配置管理，划分`prod|test|dev`
- [x] 自动上传脚本
- [x] 辅助构建submission辅助脚本: 
- [x] 视频校验工具：集成了视频属性检查（分辨率、FPS、完整性）的功能。
- [x] submission 规范
- [x] 单元测试 
- [x] CI/CD 
- [x] Docker 
- [x] 修复 `flash-attn` 的安装方式
- [x] 实现结果记录 json
- [x] 部署 Leaderboard 和 user study
- [x] **Hugging Face 集成**: `src/app/HuggingFace/` 
- [x] 可视化雷达图
- [ ] 轨迹可视化图
- [x] 提供了命令行入口
- [ ] 拓展lauch-app命令
- [ ] 集成测试
- [x] 砍掉prompt设计；
- [ ] 测试HAA,AC,BSC,CCE,TA
- [ ] 测试FVD,AQ,IQ,OFC,SSIM,LPIPS,TF,MS,PSNR
- [ ] 文档与展示 
- [ ] Type Hints 
- [ ] Docstrings
- [ ] **Pre-commit Hooks**:
- [ ] 显式依赖管理
- [ ] 指标对齐验证



## Details

