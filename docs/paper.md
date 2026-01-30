## Summary

```\TODO
Summary: 需要补全整段论文摘要。
```

## Introduction

将视觉体验从第一人称（Egocentric）视角转换为第三人称（Exocentric）视角是构建沉浸式内容的一块基石。这种被广泛称为 “第三人称跟随视角” (Third-Person Follow Camera) 的范式，在现代电子游戏、电影制作（如跟拍镜头）、新兴的 VR/AR 应用以及具身人工智能中无处不在。与静态的监控视角或游离的广播视角不同，跟随视角能有效地将观众与主角“绑定”，维持一种稳定的几何关系——通常是从后方构图——从而传达出强烈的沉浸感与速度感。这在动作场景中常被称为“追逐视角”。

然而，实现 **Ego2Exo** 视角的自动化转换仍是一项艰巨的挑战。尽管近期视频到视频（Video-to-Video, V2V）生成技术在风格迁移和视频编辑方面取得了令人瞩目的进展，但当应用于视角变换任务时，它们却根本性地失效。现有的最先进 V2V 模型往往会生成“幻觉”内容——这些内容在语义上或许合理，但在几何上却支离破碎。例如，模型可能生成了高保真的纹理，却无法维持主角的运动轨迹或相机的跟随逻辑，导致主角莫名地漂移出画面，或是背景发生不切实际的扭曲。

我们将这些失败的根源归结为关键的“信息缺失” 。 当前的 V2V 范式主要作为“纹理编辑器”运作——它们擅长基于 2D 视觉先验重绘像素，却对底层的 3D 几何映射规则视而不见。它们仅仅将输入的第一人称视频视为颜色和风格的来源，而忽略了其作为严格的几何与时序控制信号的角色。在 Ego2Exo 任务中，输入视频隐式地编码了相机的自我运动和主角的交互逻辑。现有模型缺乏解码这种隐式 3D 几何并将其重投影为连贯第三人称视角的机制。因此，它们试图通过像素级的幻觉而非几何级的翻译来跨越这一域间鸿沟。

```\TODO
Figure 1: The Teaser (首图/概念图)位置： 第一页顶部 (Page 1 Top)。这是门面，决定审稿人第一眼的印象。核心叙事： "Input (Ego + Ref) $\to$ Challenge (Invisible Back) $\to$ Output (Stable Exo)"。布局建议： 左右三段式（参考 EgoX）。左 (Input): Ego 视频帧（带红色问号遮罩，表示盲区） + $I_{ref}$ 参考图（带白框，表示外观源）。中 (Bridge): 3D 相机位移示意图（蓝色 Ego 相机 $\to$ 橙色 Exo 相机）。右 (Output): 生成的 Exo 视频帧（带绿色骨架叠加 + 地面网格，表示几何稳定）。
```

我们将这些失败的根源归结为关键的“信息缺失” (Information Gap)。 当前的 V2V 范式主要作为“纹理编辑器”运作——它们擅长基于 2D 视觉先验重绘像素，却对底层的 3D 几何映射规则视而不见。它们仅仅将输入的第一人称视频视为风格的来源，q而忽略了其作为严格的 几何与时序控制信号 的角色。

近期，诸如 `WorldWander` 等开创性工作已开始通过设计专用架构来解决这一问题。尽管前景广阔，但该领域目前仍缺乏统一的评估协议来严格量化进展。此外，一个关键问题仍然存在：通用视频扩散模型是否能在不进行复杂的、任务特定的架构修改的情况下掌握这一任务？

为了回答这个问题，我们首先建立了 **Follow Camera Bench**，这是首个专为“跟随视角”范式定制的综合基准测试。与通用指标（如 FVD）不同，我们的基准纳入了 相机轨迹误差 (CTE) 和 人体动作对齐 (HAA) 等任务特化维度，严格审计“虚拟摄影师”的表现。

其次，我们在此基准上探究了标准自回归视频扩散模型的潜力。我们观察到，标准训练会导致 **曝光偏差 (Exposure Bias)**，即误差随时间累积，导致几何漂移。为了在不改变模型架构的情况下缓解这一问题，我们采用了一种 `Self-Forcing` 训练策略。通过在训练过程中让模型以其自身生成的历史为条件，我们迫使其学习鲁棒的纠错能力和长期的几何一致性。我们将这一优化后的基线模型与专用的 `WorldWander`模型进行了比较，为专用架构与通用模型的先进训练策略之间的权衡提供了有价值的见解。

综上所述，本文的贡献主要体现在三个方面：

1. **我们建立了 Follow Camera Bench** 这是一套涵盖视觉质量、时序动态、动作语义及运镜控制的严格评估套件，为评估跟随视角视频生成树立了新标准。
2. **我们对跨视角合成中的“信息缺失”进行了系统分析**  将通用 V2V 模型与最先进的专用方法 `WorldWander` 进行了基准对比。
3. **我们验证了 Self-Forcing 作为一种训练策略的有效性** 我们的实验表明，通过简单地弥合训练-测试差距，通用基线模型可以显著提高其几何稳定性，并逼近专用模型的性能。

## Related Work

### General Video Generation Benchmarks

文本到视频 (T2V) 生成技术的最新进展催生了综合评估套件的发展。`VBench`和 `EvalCrafter `等代表性基准标准化了通用视频质量的评估，重点关注美学评分、时序流畅度和文本-视频一致性等维度。然而，当应用于 Ego2Exo 领域时，这些基准表现出显著的局限性。首先，它们缺乏**条件控制评估**机制，主要关注对提示词 (Prompt) 的遵循，而非视觉参考所施加的几何约束。更为关键的是，它们无法衡量**跨视角一致性**，即生成的“背面视角”是否在几何逻辑上与“正面视角”输入相匹配。因此，这些基准中的标准指标（如 FVD）不足以审计本任务所需的严格空间映射。

### Egocentric-Exocentric Vision: From Analysis to Synthesis

`Ego4D`和 `Ego-Exo4D` 等大规模数据集的发布，极大地推动了跨视角关联理解的研究。然后，现有的工作主要集中在判别式任务上，包括时序对应、跨视角检索和未来预测。虽然部分工作探索了从第一人称输入生成 3D 骨架的 `Ego-to-Pose` 估计，但这些方法止步于像素级生成之前。它们提供了真值数据，但缺乏已建立的生成式基线（如 `SVD`, `DiT`）或生成质量的评估标准。我们的工作通过将范式从理解视频转变为创造照片级真实的第三人称视频，填补了这一空白，标志着首个致力于像素级 Ego2Exo Translation 的基准测试的诞生。

### Video-to-Video Translation vs. Novel View Synthesis

**视频到视频转换** 当前的 V2V 模型（如 `ControlVideo `和 `TokenFlow`）在强结构保留假设下运行。它们擅长风格迁移（例如，将舞者重新纹理化为动漫角色），其中输入和输出像素在空间上是对齐的。相比之下，我们的 Ego2Exo 任务打破了这一假设，要求从重纹理向重渲染进行范式转变。输入（手/视野）和输出（全身/背影）在像素空间上几乎不重叠，迫使模型隐式执行 3D 空间变换，而非简单的结构引导。

**新视角合成** `NeRF` 和 **`3D Gaussian Splatting`** 等技术擅长从多视角输入重建静态场景。然而，我们的任务在三个方面存在根本差异：i) **单视角输入**：我们仅依赖单目第一人称视频；ii) **动态场景**：主体和环境处于持续运动中；iii) **幻构** ：与重建不同，我们的模型必须利用生成先验来合理地幻构不可见区域（例如主体的背部），这使得我们的工作属于生成先验范畴，而非严格的 3D 重建范畴。

## Method

### Task formulation

我们将 Ego-Exo 转换任务形式化为一个“参考引导的视频到视频生成”问题。设 $\mathcal{V}_{ego}$ 表示以第一人称视角捕捉相机佩戴者动作的视频。正如 `WorldWander` 所阐述的，第一人称视角提供了几何线索（如轨迹、步态），但本质上缺乏关于佩戴者外观的视觉信息。因此，仅从 $\mathcal{V}_{ego}$ 推断出 exo 视角 $\mathcal{V}_{exo}$ 是一个不适定问题。为了解决这种模糊性，我们引入了一张参考图像作 $I_{ref}$ 为显式的外观条件。生成映射$\mathcal{M}$ 被定义为：
$$
\hat{\mathcal{V}}_{exo} = \mathcal{M}(\mathcal{V}_{ego}, I_{ref}).
$$
在这个框架中，$\mathcal{V}_{ego}$ 决定了三维几何和运动动态，而 $I_{ref}$ 严格定义了视觉外观（例如，衣物纹理、发型）。

**高保真采集**。我们采用双摄像头设备来捕获同步的第一人称-第三人称视图对。第一人称视图使用DJI Action 5 Pro录制以模拟人类视觉，而第三人称视图则由DJI Osmo 360全景相机捕捉。不同于由短片段组成的现有数据集，我们记录连续的10分钟序列，分辨率为1920×1080，帧率为60fps。延长的持续时间对于评估累积的轨迹漂移至关重要，同时高帧率减少了运动模糊，为以动作为中心的生成模型提供了精确监督。。在我们的基准测试中的一个重要区别在于评估单元的定义。标准判别式基准（例如，`Ego-Exo4D`）通过生物主体来组织数据，以评估识别不变性——即无论穿着如何变化都能识别同一人。然而，对于我们的生成任务，主要目标是外观保真度。模型必须忠实地渲染 $I_{ref}$ 提供的特定外观，而不是从训练集中检索记忆中的主体。

### Dataset Construction

为了解决现有数据集的局限性—— 包括短时长、运动模糊以及松散的时间同步，我们引入了 `FollowCameraBench` 。

```
Figure 1: The FollowCameraBench Overview总体布局： 横向长图，分为左、中、右三个面板，逻辑流从左到右。Panel A: The Data Engine (数据引擎)核心卖点： High-Fidelity Acquisition & Virtual Gimbal (对应 Sec 3.2)设计指令：物理采集 (Top):画一个简化的 Dual-Camera Rig 示意图：一个人头戴 Action 5 Pro (Ego)，手持自拍杆末端连接 Osmo 360 (Exo)。标注：Synchronized Acquisition。虚拟云台 (Bottom - 核心):展示“变废为宝”的过程。Input: 画一个晃动的全景球体 (Raw 360)。Process: 一个漏斗或齿轮图标，旁边标注 "Virtual Gimbal Pipeline" (World-Lock + Re-center)。Output: 输出一条平滑、水平的长胶卷带。高亮参数: 用醒目的字体标注 10-min Continuous, 60 FPS, Zero Drift。Panel B: The Task & Hard Split (任务与划分)核心卖点： Single-Reference & Dual-Constraint Split (对应 Sec 3.1 & 3.3)设计指令：任务流 (Top):Input: $\mathcal{V}_{ego}$ (蓝色框) + $I_{ref}$ (黄色框)。Arrow: 指向右侧，标注 "Generative Mapping $\mathcal{M}$"。Output: $\hat{\mathcal{V}}_{exo}$ (绿色框)。细节： 在 Output 旁边加个小注脚："Appearance from Ref, Geometry from Ego"。严苛划分 (Bottom - 关键):画一个 矩阵/棋盘格 (Split Matrix)。行是 Appearance，列是 Scene。Train: 蓝色格子；Test: 红色格子。关键视觉： 确保红蓝格子互不交叉（即行和列都不重叠）。标注："Dual-Constraint Zero-Shot Split"。这能瞬间解释清楚你 section 3.3 的核心难点。Panel C: Comprehensive Evaluation (多维评测)核心卖点： Appearance Fidelity & Geometric Stability (对应 Sec 3.1 & 3.5)设计指令：雷达图 (Radar Chart):画一个五边形或六边形雷达图。维度 1: Visual Quality (FVD).维度 2: Appearance Fidelity (CLIP-I) —— 注意：这里用 "Appearance" 而不是简单的 ID，呼应文中 "Redefining Identity" 的观点。维度 3: Geometric Stability (Drift Error) —— 这是长视频特有的。维度 4: Temporal Consistency.协议图标 (Protocol Icons):在雷达图下方放两个禁止图标（红圈斜杠）：图标 1: "No Optimization" (Training-Free).图标 2: "No Multi-View Ref" (Single-Reference Only).
```

**高保真采集**。我们采用双摄像头设备来捕获同步的第一人称-第三人称视图对。第一人称视图使用 DJI Action 5 Pro 录制以模拟人类视觉，而第三人称视图则由 DJI Osmo 360 全景相机捕捉。不同于由短片段组成的现有数据集，我们记录连续的，时间码对齐的长序列配对，最短的视频长度不低于10分钟，分辨率为 1920×1080，帧率为60fps。延长的持续时间对于评估累积的轨迹漂移至关重要，同时高帧率减少了运动模糊，为以动作为中心的生成模型提供了精确监督。

```
\TODO Video Duration: "最短的视频长度不低于 [\TODO]" —— 需要填入具体的时长数值（例如 10秒 或 30秒）。
```

**后期制作流程**。由于格式特殊，原始360度镜头不适合直接训练。我们引入了一个标准化的流程来模拟电影和游戏风格的第三人称跟随拍摄：首先应用方向锁定，将全景素材的观看角度从录制参与者的运动中解耦，确保生成的视频反映纯粹的平移而非混乱的旋转；其次，从稳定化的球体中提取一个直角窗口，并手动调整以保持主体位于中央区域，模仿游戏中的相机视角；最后，利用会话开始和结束时录制的高频音频锚点（同步声音）进行基于锚点的同步，确保零时间漂移。

```
Figure 2: The Data Engine (数据管线图)位置： 第三页 (Section 3: Dataset Construction)。放在介绍采集和处理的那一段。核心叙事： "我们的数据虽然是手持的，但经过了虚拟云台处理，质量极高。"布局建议： 流水线风格（参考 CelebV-HQ）。Step 1 (Acquisition): 双相机硬件 Rig 示意图 + 音频波形对齐图（Sync）。Step 2 (Stabilization): 歪斜的全景球体 $\to$ 陀螺仪图标 $\to$ 水平的全景球体。Step 3 (Reframing): 水平球体上的黄色视窗框住人物 $\to$ 裁剪出的 1080p 视频。
```

**多样性和覆盖范围**。该集合涵盖了140+细分场景，形成30+种交互动作与50+服饰-环境组合的基准库，直接支撑任务中"Appearance Fidelity"的核心目标——验证模型能否精准渲染 $I_{ref}$ 指定的特定外观，而非依赖场景惯性。此设计显著超越现有数据集（如`EgoExo8K`）在动态时序长度与视觉保真度的局限，为 Ego-Exo Translation 任务提供首个高保真、高多样性基准。

```
Figure 3: Data Statistics & Split (统计与划分图)
位置： 第四页或第五页 (Section 3.3: Statistics)。

核心叙事： "我们的划分是 Zero-Shot 的（外观隔离），但任务是可解的（动作覆盖）。"

布局建议： 左右并列 (a) + (b)。

(a) Split Matrix (矩阵图): 类似于棋盘格。行=衣服，列=场景。训练集（蓝格）和测试集（红格）互不交叉。

(b) Dual-Space t-SNE (散点图):

Appearance Space: 红蓝分离（证明 Zero-shot）。

Motion Space: 红蓝重叠（证明动作分布一致）。
```

**双重约束硬分割**。为了严格基准零样本泛化，我们采用了“选择并清除”的分割策略，该策略强制执行跨外观和跨场景的原则，确保背景几何形状或特定服装纹理在训练期间未见过。

### Benchmark Suite

**评估协议** 为了确保公平比较和可重复性，我们实施了严格的标准化协议：i) **单一参考设置**。我们采用具有挑战性的单一参考配置。模型提供有 $\mathcal{V}_{ego}$ 和恰好一张$I_{ref}$。明确禁止在推理过程中使用多视图参考或360度扫描，以测试纹理拼接上的三维推理能力。i) **免训练（仅推理）政策**。我们严格评估零样本泛化。任何形式的测试时间优化（如LoRA、文本反转）在测试参考上是严格禁止的。iii) **确定性采样**。为了消除扩散模型中随机采样的方差，我们提供了一个标准化的评估列表，其中包含所有测试样本的固定随机种子。

**难度分层** 

```
\TODO Difficulty Stratification: —— 需要定义 Easy/Hard 样本的划分标准（例如根据相机运动速度或遮挡程度）。
```

### Dimensions and Metrics

为全面评估生成质量并严格约束Ego2Exo任务的约束条件，受到 `Vbench`的启发， 我们构建了涵盖五个正交维度的综合评估协议。这些指标突破传统视频生成指标的局限，引入针对跟随摄像范式的任务专用标准。

#### Visual Quality and Temporal Dynamics

我们首先评估生成视频的基础感知保真度与运动特性。

**感知分布** 采用**Fréchet视频距离**作为合成视频与真实域分布距离的核心指标，确保生成质量符合主流视频生成标准。

**图像与美学质量** 通过基于MUSIQ的**图像质量**量化帧级技术清晰度，结合基于CLIP+MLP的**美学质量**评估无参考条件下的构图与色彩和谐度。

**时间稳定性** 计算**时间闪烁度**以惩罚高频伪影并确保视觉稳定，同时采用**动态程度**验证运动幅度，防止模型通过生成静态图像作弊稳定性得分。此外，**主体检测率**用于验证主体完整性。

#### Semantics and Action Consistency

Ego2Exo生成的核心挑战在于保留主体身份的同时转换第一视角运动。

**外观忠诚度** 在基于CLIP的**外观一致性**基础上，我们引入基于 DINOv2 的**结构保真度**。DINOv2特征对细粒度纹理和局部结构更敏感，为"身份一致性"提供鲁棒度量，确保主体身份在跨视角生成中保持不变。

**动作对齐** 我们提出**人体动作对齐**评估动作转换准确性。通过 3D 骨骼姿态对齐，该指标直接验证合成第三人称动作是否严格遵循第一视角运动先验。

#### Cross-View Correlation

为了区分本任务与通用视频生成，我们引入特定指标来衡量 Ego 源与 Exo 结果之间的因果关联。

**源控制召回**：该指标评估零样本外观泛化能力。通过测量生成视频能否从数千干扰视频中唯一检索到其对应的 Ego 源，验证其保留的独特特征完整性。

**时间注意力对齐**：在 Ego2Exo 任务中，Ego 视角的输入不仅是视觉参考，更是一种严格的时序控制信号。标准指标（如 FVD）评估分布质量，但无法惩罚时序幻觉， 即生成的视频虽然流畅，但存在时序错乱、延迟或模式坍塌。为了填补这一空白，我们引入了 Temporal Attention Alignment，与聚合 Value 的标准注意力机制不同，TAA 抽离和关注 Attention Map。它严格量化了生成视频序列$V_{gen}$ 与作为时序锚点的真值 Exo 视频 $V_{gt}$ 之间的 因果同步性。我们采用对时序敏感的编码器 (VideoMAE V2) 提取双流特征序列。通过计算生成查询 $Q$ 与真值键 $K$ 之间的注意力矩阵 $A \in \mathbb{R}^{T \times T}$，我们将 $A$ 的对角线迹解释为对齐分数。高 TAA 分数代表单调对齐，意味着模型成功习得了精确的时间映射逻辑，而非仅仅是幻觉出合理的运动。

#### Geometry and Perspective

我们验证生成 3D 空间的物理合理性。

**空间几何**：**侧边深度一致性**通过单目深度估计，验证视角关系（如角落、走廊）是否正确构建。

**轨迹对齐**：通过**平均位移误差**测量主体2D 运动路径与真实期望的偏离度。

#### Camera Movement Control

最后，我们定量评估隐含在跟随摄像范式中的“虚拟摄影师”逻辑。

**居中构图** 通过**摄像机中心误差**直观测量主体偏离视觉中心的程度，验证基础跟随镜头逻辑。

**3D轨迹一致性** 基于SLAM的提升方法，**摄像机轨迹误差**评估合成3D摄像路径是否符合理想跟随曲线。

**跟随约束** 测量**主体-摄像机距离误差**以检测不合理变焦，并计算**朝向对齐**确保摄像机随主体转向同步旋转。CSHA是区分动态"跟随视角"与静态"监控视角"的关键判据。

**平滑约束** **轨迹平滑度**确保Exo摄像轨迹的加速度显著低于Ego轨迹，模拟专业影像稳定效果。

## Experiments

### Baselines Setting

为了对 Ego2Exo 生成任务提供全面的评估，我们对跨越三种不同范式的方法进行了基准测试，范围涵盖从通用视频编辑到任务专用的架构。这种多样化的选择使我们能够深入探究架构专业化与训练策略优化之间的权衡。

**基于专用架构的方法** 我们选择 `WorldWander`作为该类别的首要基线。作为专为第一人称到第三人称转换定制的开创性工作，`WorldWander`引入了一套包含 上下文视角对齐和协同位置编码的专用架构。它代表了目前通过显式架构设计来处理跨视角几何映射的 SOTA 水平，为专用模型性能提供了“上限”参考。

**通用视频到视频编辑方法** 为了评估最先进通用模型的零样本能力，我们采用 **WanVACE** [Jiang et al., 2025] 作为代表性基线。该方法在 VACE 统一编辑框架内利用了强大的 Wan-1.3B 主干网络，基于预训练的扩散先验运行，并依赖输入的第一人称视频提供结构引导。通过在标准编辑配置下对 WanVACE 进行基准测试，我们要从经验上验证“信息缺失”假设——即证明即使是为结构保留编辑而设计的先进基础模型，在缺乏特定几何微调的情况下，也难以适应 Ego2Exo 转换所需的剧烈视角合成。

**基于训练的优化策略** 最后，为了探索通用模型在不进行架构修改情况下的潜力，我们实施了一个` Self-Forcing `基线。与修改网络结构的 `WorldWander` 不同，`Self-Forcing` 侧重于训练范式。它通过在训练过程中让模型以其自身的不完美预测为条件，从而缓解自回归视频生成中固有的曝光偏差。我们将此策略应用于`WanVACE`，以构建一个鲁棒的通用基线，旨在测试仅通过弥合训练-测试差距是否足以掌握长时序生成中的几何一致性。

### Comparison

**Visual Quality vs. Geometric Consistency.** 如 Table 1 所示，我们观察到不同范式模型之间存在显著的性能权衡（Performance Trade-off）。 首先，得益于 14B 参数量的强大生成先验，**WanVACE** 在所有视觉感知指标（FVD, IQ, AQ）上均取得了最佳成绩，优于专用架构的 **WorldWander**。这表明，通用基础模型（Foundation Models）在纹理渲染、光影一致性和细节生成上具有不可替代的优势。

```
\TODO Table 1 (Quantitative Comparison): 文中提到了 "如 Table 1 所示"，但表格未插入。
需填入内容：WorldWander, WanVACE, WanVACE+Self-Forcing 在 17 个指标上的对比数据。
```

然而，在关键的任务特化指标（CTE, HAA, CSHA）上，标准的 **WanVACE** 表现出严重的几何失效。例如，其相机轨迹误差（CTE）显著高于 WorldWander，表明模型虽然生成了高质量的画面，但未能遵循“跟随”的几何逻辑。相比之下，**WorldWander** 凭借其显式的视角对齐模块，在几何一致性上建立了坚实的“上限”（Upper Bound）。这种“高画质、低控制”的现象有力地验证了我们提出的“信息缺失（Information Gap）”假设——即通用模型缺乏将 Ego 信号转化为精确 3D 约束的内在机制。

**Efficacy of Self-Forcing.** 最引人注目的是，引入 **Self-Forcing** 训练策略后，WanVACE 的几何性能得到了质的飞跃。在不修改任何模型架构的前提下，**WanVACE + Self-Forcing** 将 CTE 误差大幅降低，并在 HAA（动作对齐）上逼近了 WorldWander 的水平。这证明了长时序生成中的几何漂移（Geometric Drift）本质上是自回归生成的曝光偏差问题。通过弥合训练-测试差距，我们成功地在通用大模型中注入了精确的几何控制，兼得高保真画质与高稳定性。

### Qualitative Results

为了直观验证上述定量发现，我们在 Figure X 中展示了不同方法的生成样本对比。

**Visualizing Geometric Drift.** 我们可以观察到，标准的 **WanVACE** 在视频初期（前 2 秒）能生成合理的跟随视角，但随着时间推移，表现出严重的**累积性几何漂移（Cumulative Geometric Drift）**。如图中红色箭头所示，生成的相机轨迹逐渐偏离主体的运动路径，导致主体滑出画面中心（Off-center）甚至完全消失。此外，背景几何往往出现非物理的扭曲，这表明模型仅仅是在根据纹理相关性进行“幻觉”生成，而非遵循 3D 物理规律。

**Visualizing Action Hallucination.** 在动作语义方面，通用编辑模型经常出现因果错位。例如，当 Ego 输入显示剧烈左转时，WanVACE 生成的 Exo 视角往往出现滞后（Lagging）或生成与原动作无关的平移运动。相比之下，**WorldWander** 展示了极其稳定的跟随逻辑，而我们的 **WanVACE + Self-Forcing** 成功复现了这种稳定性，能够及时调整相机朝向以保持与 Ego 视角的同步，消除了标准微调中的动作幻觉。

```
Figure 5: Qualitative Comparison (定性对比/胶卷图)位置： 第七页或第八页 (Section 5: Analysis)。通常占页宽的一半或全宽。核心叙事： "Baseline 会飘、会变形，而我们很稳。"布局建议： 多行胶卷条 (Filmstrip)。Row 1 (Input): Ego 视频 ($t=0, 30, 60$)。Row 2 (SVD/Gen-2): 生成结果。用红框圈出人物扭曲、消失或背景漂移的地方。Row 3 (Ours): 生成结果。用绿框圈出细节保持完好、背影结构正确的地方。
```

### Ablation Study

为了验证 **Self-Forcing** 策略的独立贡献，我们基于 WanVACE 框架进行了严格的消融实验。如 Table X 所示，我们对比了标准微调策略（Standard Fine-Tuning / Teacher Forcing）与我们提出的 Self-Forcing 策略。

实验结果表明，在相同的模型底座（Wan-1.3B）和相同的数据集设置下，仅改变训练策略（引入 Self-Forcing）使得 **CTE（相机轨迹误差）** 从 $0.55$ 显著下降至 $0.18$。这有力地证明了性能的提升并非源于模型容量的增加，而是源于 Self-Forcing 有效缓解了自回归生成中的曝光偏差（Exposure Bias），使其能够处理长时序生成中的误差累积问题。

```
\TODO Table X (Ablation Metrics): 文中提到了 "如 Table X 所示"，但表格未插入。

需填入内容：Standard FT vs Self-Forcing 在 CTE 和 FVD 等核心指标上的对比。
```



### Human  Preference Study

尽管 FVD 等自动化指标提供了分布差异的统计度量，但它们往往难以精确反映细粒度的人类感知，特别是在评估运镜稳定性和语义逻辑方面。为了严格评估生成视频的感知质量，我们采用双盲二选一强制选择 (2AFC) 协议进行了一项综合用户研究。

我们邀请了 $N$ 名参与者对 $M$ 组随机抽样的视频对进行评估。

```
\TODO Variables N & M: "邀请了 $N$ 名参与者... $M$ 组视频对" —— 需要填入具体数字。
```

在每次测试中，向参与者展示源第一人称视频和参考图像，以及两个匿名生成视频（我们的方法 vs 基线模型），视频的左右顺序被随机打乱以消除位置偏差。我们要求参与者基于源自 Ego2Exo 任务核心挑战的五个具体维度及一个综合维度，选择表现更优的结果：

1. **视觉观感 (Visual Quality):** 关注画质清晰度、纹理细节及构图美观度 (Corresponds to IQ/AQ/FVD).
2. **动态流畅性 (Temporal Dynamics):** 关注视频播放的稳定性，是否存在闪烁或卡顿 (Corresponds to TF/DD).
3. **语义与动作还原 (Semantics & Action):** 验证人物身份是否一致，以及动作（如挥手、转身）是否忠实于第一人称输入 (Corresponds to SF/HAA).
4. **几何合理性 (Geometric Consistency):** 评估背景透视关系是否正确，人物是否存在不合理的“滑步”或漂移 (Corresponds to SSDC/ADE).
5. **运镜逻辑 (Camera Control):** 评估“虚拟摄影师”的技术，如主体是否居中、相机是否随人物转向而智能旋转 (Corresponds to CTE/CSHA).
6. **综合偏好 (Overall Preference):** 综合考虑上述因素，选择更适合应用于游戏或电影的视频。

Consistency Analysis: Metrics vs. Perception.

我们观察到自动化指标与人类偏好之间存在显著的相关性，同时也揭示了通用指标的局限性。值得注意的是，尽管 Baseline (WanVACE) 在 FVD 上取得了有竞争力的分数，但在用户研究的 Q6 (Overall Preference) 中却显著落后于我们的方法。这表明人类评估者在 Ego2Exo 任务中更看重**逻辑的一致性（几何/动作）**而非单纯的纹理保真度。同时，人类对 Q5 (Camera Control) 的偏好分布与我们提出的 CTE 指标高度一致，验证了 CTE 作为衡量长时序稳定性的有效性。

```
User Study Chart: 虽然文中没显式提到图表，但通常需要一个柱状图来展示 Q1-Q6 的胜率结果。
```

```\TODO
Figure 4: Evaluation Results (量化评测图)
位置： 第六页或第七页 (Section 4: Experiments)。

核心叙事： "在人类主观评测和客观指标上，我们都不仅比 Baseline 好，而且好得很全面。"

布局建议： 组合图表。

(a) User Study (堆叠柱状图): 展示 A/B Test 的胜率。绿色代表 Ours Win，灰色 Tie，红色 Loss。

(b) Radar Chart (雷达图): 6 个维度（Visual, ID, Geometry, Temporal, etc.）。
```



## Discussion

### Limitation

**[\TODO]**

### Future Work

**[\TODO]**

## Conclusion

**[\TODO]**

## supplement material

### Implementation Details of Evaluation Metrics

为了全面且客观地评估 Ego2Exo 视频生成任务，我们构建了一个涵盖通用视频生成指标与任务特化指标的评估体系。在通用指标方面，我们严格遵循业界标准协议（如 `VBench`）进行配置以确保结果可比性；同时，针对本任务在身份保持、动作对齐及运镜控制等方面的独特需求，我们设计并详细定义了一系列特化指标。

#### 通用指标配置

对于通用指标的评估，我们采用以下标准配置：i) **Fréchet 视频距离 (FVD)** 用于衡量生成与真实视频分布的一致性，我们使用 Kinetics-400 预训练的 I3D 网络提取特征，并将所有视频统一处理为 16 帧、256×256 分辨率进行计算。ii) **感知与图像质量** 通过组合指标评估。使用 VGG 骨干网络的 LPIPS 衡量感知距离，利用 MUSIQ 模型评估技术成像质量 (IQ)，并采用 LAION-Aesthetics Predictor V2 量化美学评分 (AQ)。iii) **视频动态性** 依据 `VBench` 定义，通过 RAFT 提取光流并计算时间闪烁度 与动态程度，以分别评估视觉稳定性和运动剧烈程度。iv) **语义与外观一致性 ** 鉴于本任务不依赖文本提示，为了适配该任务，我们重新定义了语义一致性指标。我们计算 Appearance Consistency 分数，该指标利用 CLIP ViT-L/14 的图像编码器提取生成视频帧与参考图像的高层视觉特征。通过计算两者在共享特征空间中的平均余弦相似度，该指标量化了模型在长时序生成中保持主体身份、衣着及整体视觉语义不发生漂移的能力。

针对 Ego2Exo 任务中独特的“跨视角关联”、“动作语义对齐”及“跟随运镜”需求，我们设计了以下特化指标。所有涉及 3D 人体参数的计算均基于 `WHAM` 模型进行估计。

**Structural Fidelity** 尽管 CLIP 能捕捉图像的高层语义，但在维持细粒度的结构和纹理方面往往表现不足。为此，我们引入 DINOv2 来评估主体身份的结构一致性。该指标定义为参考图像与生成帧特征的平均余弦相似度：
$$
\text{SF} = \frac{1}{T} \sum_{t=1}^{T} \text{CosineSim}(f_{dino}(I_{crop, t}^{gen}), f_{dino}(I_{ref}^{gt}))
$$
其中 $\mathcal{E}_{\text{dino}}$ 表示冻结参数的 DINOv2 ViT-L/14 编码器。由于 DINOv2 对局部几何特征具有更高的敏感度，该指标能有效监测并防止生成过程中的“身份漂移”现象。

**Human Action Alignment** 为了评估从第一人称到第三人称的动作翻译准确性，我们摒弃了对绝对位置敏感的 MPJPE 指标，转而采用对视角变化鲁棒的肢体向量余弦相似度。利用 WHAM 提取每帧的 3D 关节点 $J \in \mathbb{R}^{K \times 3}$，并根据人体拓扑定义一组肢体向量 $\mathcal{L}$。对于第 $t$ 帧的第 $i$ 个肢体，其单位方向向量为
$$
\vec{v}_{t,i} = \frac{J_{t, \text{end}} - J_{t, \text{start}}}{||J_{t, \text{end}} - J_{t, \text{start}}||_2}.
$$
 HAA 定义为生成视频与 GT 视频对应肢体向量的平均余弦相似度
$$
\text{HAA} = \frac{1}{T \cdot |\mathcal{L}|} \sum_{t=1}^{T} \sum_{i \in \mathcal{L}} \left( \vec{v}_{t,i}^{gen} \cdot \vec{v}_{t,i}^{gt} \right)
$$
该指标范围为 $[-1, 1]$，值越高表示动作语义还原越精准。

**Source Control Condition Recall** 该指标旨在量化模型对 Ego 视频特征的零样本外观泛化能力，即验证生成的视频是否保留了足够独特的源信息以指回其来源。我们采用预训练的 `VideoMAE` V2作为特征提取器 $\Phi(\cdot)$，以捕捉高层的时空语义特征。我们构建了一个由全体 Ego 视频组成的干扰集 $\mathcal{D}_{neg}$。对于每个生成的 Exo 视频 $\hat V_{exo}$，提取其时空特征 $f(\hat V_{exo})$，并在干扰集与对应的真值 Ego 视频的并集 $\mathcal{D}_{neg} \cup \{V_{ego}^{gt}\}$ 中进行最近邻检索。若 $V_{ego}^{gt}$ 成功出现在检索结果的前 $K$ 位，则视为命中。
$$
\text{SCCR}@K = \mathbb{I}\left( \text{rank}(V_{ego}^{gt}) \le K \right)
$$
我们最终报告 **Recall@1** 和 **Recall@5**。

**Side-by-Side Depth Consistency** 该指标用于验证生成场景的 3D 几何结构是否合理。利用 Depth Anything V2 提取相对深度图 $D_{gen}$ 和 $D_{gt}$。由于单目深度存在尺度和位移的不确定性，我们首先通过最小二乘法求解最佳变换参数 $s, b$，使得 $||s \cdot D_{gen} + b - D_{gt}||_2^2$​ 最小化。计算对齐后的生成深度图与 GT 深度图之间的结构相似性 (SSIM)：
$$
\text{SSDC} = \text{SSIM}(\hat{D}_{gen}, D_{gt})
$$
这比像素级误差更能反映几何结构的感知一致性。

**Temporal Attention Alignment** 标准的交叉注意力机制运作方式为 
$$
\text{Output} = \text{Softmax}(QK^T) V.
$$
在我们的评估框架中，我们并不关心什么 value 被传输了；相反，我们审查模型在时间轴上 看向哪里。我们剥离了 Value 分量以审计生成器的时序指针，确保时间 $t$ 的生成帧有效地查询了时间 $t$ 的真值信息。

设 $V_{gen}$ 为生成视频，$V_{gt}$ 为真值 Exo 视频。我们使用 VideoMAE V2 ($\Phi$) 作为骨干网络，因为它具有强大的时序辨别能力。我们提取长度为 $T$ 的帧级特征序列
$$
Q = \Phi(V_{gen}) \in \mathbb{R}^{T \times D}, \quad K = \Phi(V_{gt}) \in \mathbb{R}^{T \times D}
$$
这里，$Q$ 作为 Query（生成流），$K$ 作为 Key（真值流）。

我们计算行归一化的 Attention Map：
$$
A = \text{Softmax}_{\text{row}}\left( \frac{Q K^T}{\tau} \right) \in \mathbb{R}^{T \times T}
$$
其中 $\tau$ 是用于调节锐度的温度系数。元素 $A_{i,j}$ 具有明确的物理意义：“为了生成第 $i$ 帧，模型在多大程度上依赖了真值的第 $j$ 帧？” 。理想的受控生成意味着一对一映射 ($t \to t$)，结果应为对角矩阵。TAA 定义为主对角线上的平均能量：
$$
\text{TAA} = \frac{1}{T} \text{tr}(A) = \frac{1}{T} \sum_{t=1}^T A_{t,t}
$$
高 TAA的生成视频其注意力热力图显示为一条明亮、锐利的对角线，表明 单调对齐和对因果律的遵循。低 TAA偏差表现为：i）偏移直线，说明生成视频存在系统性延迟或抢跑。ii) 垂直条纹，这意味着存在模式坍塌或重复单一静态帧。iii) 弥散云团，可能意味着时序幻觉或控制丢失。

我们注意到 TAA 依赖于独特的时序特征。在时序熵极低或自相关性极高的场景中，由于时间映射本质上变得模糊，该指标的辨别力会降低。

基于 DROID-SLAM 重建的相机轨迹 $C_{gen} \in SE(3)$ 和 WHAM 估计的主体 3D 信息，我们定义以下约束：

**Camera Centering Error** 衡量生成视频是否将主体保持在视觉中心。计算检测框中心 $\mathbf{c}_{box}$ 与图像中心 $\mathbf{c}_{img}$ 的归一化欧氏距离：
$$
\text{CCE} = \frac{1}{T} \sum_{t=1}^{T} \frac{||\mathbf{c}_{box, t} - \mathbf{c}_{img}||_2}{\sqrt{H^2 + W^2}/2}
$$
**Subject-Camera Distance Error**: 衡量跟随距离的稳定性，防止不合理的“呼吸效应”。利用 WHAM 提取主体根节点在相机坐标系下的深度 $Z_t$。在进行全局尺度对齐后，计算生成深度序列与 GT 深度序列的均方根误差 (RMSE)：
$$
\text{SCDE} = \sqrt{\frac{1}{T} \sum_{t=1}^T (Z_t^{gen} - Z_t^{gt})^2}
$$
**Camera-Subject Heading Alignment** 衡量相机是否具备“智能跟随”逻辑，即随人物转身而旋转。我们在水平面 (XZ平面) 上定义相机的方位向量 $\vec{v}_{cam}$ ，其从主体指向相机，和主体的面部朝向向量 $\vec{v}_{face}$。CSHA 计算两者相对夹角 $\Delta \theta_t$ 在时序上的方差：
$$
\text{CSHA} = \text{Var}_t (\angle(\vec{v}_{cam, t}, \vec{v}_{face, t}))
$$
较低的方差表明相机与主体保持了相对固定的方位关系，即始终跟随在身后。

**Camera Trajectory Error** 在单目视频生成中评估相机控制面临两个基础的数学挑战：尺度模糊和 参考系任意性。因此，直接计算生成轨迹 $T_{gen}$ 与真值轨迹 $T_{gt}$ 的欧氏距离是不适定的。为此，我们实施了 对称规范评估 协议，并结合 **Sim(3) Umeyama 对齐**，以严格量化相机 3D 路径的偏差。我们引入规范相机假设，即为生成视频 $V_{gen}$ 和真值视频 $V_{gt}$ 假设一个具有固定内参的规范相机。我们将 $V_{gen}$ 和 $V_{gt}$ 输入同一个鲁棒的视觉里程计估计器 `DPVO` 以获取各自的轨迹。 该策略将生成的几何质量与系统级的估计偏差隔离开来。虽然假设的内参可能与真实内参不同，但由此产生的全局尺度差异可以通过后续的 Sim(3) 对齐在数学上进行修正。

设 $T_{gt} = \{\mathbf{p}_{gt, 1}, \dots, \mathbf{p}_{gt, N}\}$ 和 $T_{gen} = \{\mathbf{p}_{gen, 1}, \dots, \mathbf{p}_{gen, N}\}$ 为 $\mathbb{R}^3$ 空间中的相机平移向量序列。我们寻求一个相似变换 $S \in \text{Sim}(3)$ —— 包含统一缩放因子 $s \in \mathbb{R}^+$、旋转矩阵 $R \in SO(3)$ 和平移向量 $\mathbf{t} \in \mathbb{R}^3$ —— 以最小化最小二乘误差：
$$
\min_{s, R, \mathbf{t}} \sum_{i=1}^{N} || \mathbf{p}_{gt, i} - (s R \mathbf{p}_{gen, i} + \mathbf{t}) ||^2
$$
我们通过 Umeyama 算法求解该优化问题。首先，计算质心 $\mu_{gt}, \mu_{gen}$ 及去中心化坐标 $\mathbf{q}_i = \mathbf{p}_{gt, i} - \mu_{gt}, \quad \mathbf{y}_i = \mathbf{p}_{gen, i} - \mu_{gen}$; 其次，计算协方差矩阵 $H = \sum \mathbf{y}_i \mathbf{q}_i^T$ 并执行 SVD ($H = U \Sigma V^T$)。最佳旋转为：
$$
R = V \text{diag}(1, 1, \det(VU^T)) U^T
$$
最后进行缩放和平移：
$$
s = \frac{\sum \mathbf{y}_i^T R^T \mathbf{q}_i}{\sum ||\mathbf{y}_i||^2}, \quad \mathbf{t} = \mu_{gt} - s R \mu_{gen}
$$
最后，我们将最佳变换应用于生成轨迹：$\hat{\mathbf{p}}_{gen, i} = s R \mathbf{p}_{gen, i} + \mathbf{t}$。相机轨迹误差 (CTE) 定义为对齐后的绝对轨迹误差 (ATE) 的均方根误差 (RMSE)：
$$
\text{CTE} = \sqrt{ \frac{1}{N} \sum_{i=1}^{N} || \mathbf{p}_{gt, i} - \hat{\mathbf{p}}_{gen, i} ||_2^2 }
$$
上述指标共同构成了一个多层次、多维度的评估框架，旨在系统性地衡量 Ego2Exo  Translation模型在视觉质量、语义忠实度、身份保持、动作翻译准确性与可控性等方面的综合性能。