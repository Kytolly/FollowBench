## Intro

现有 V2V 模型存在“信息缺失（Information Gap）”，它们看不到几何映射规则。

- 简单展示现有 V2V 模型（Baselines）的失败案例（放一张 Teaser 图），以此引出你的 **Motivation** —— 我们需要几何感知，而不是纹理编辑。现有 V2V 模型存在“信息缺失（Information Gap）”，它们看不到几何映射规则。
- Contribution: 提出了 Benchmark，提出了 IC-LoRA + Self-forcing 方法。

| 通用游戏/影视跟随视角 | **Third-Person Follow Camera** | 最广泛接受，适合文档或设计说明 |
| --------------------- | ------------------------------ | ------------------------------ |
| 强调追逐/速度感       | **Chase Camera**               | 动作类游戏或电影场景描述       |
| 技术实现（如 Unity）  | **Follow Camera**              | 开发文档、代码注释中使用       |
| 电影拍摄              | **Tracking Shot**              | 影视制作或视频拍摄相关场景     |
| 肩后视角              | **Over-the-Shoulder**          | 需明确强调角色部分视角时       |

##  related work

通用视频生成 Benchmark (General Video Generation Benchmarks)

- **代表作**: VBench , VBench++, EvalCrafter。
- **侧重点**: 评估 T2V (Text-to-Video) 的通用质量，如美学评分、时序流畅度、文本一致性 。
- **局限性**:
  - **缺乏条件控制评估**: 它们主要关注 Prompt Following，无法评估“基于参考视频的几何约束” 。
  - **无法衡量跨视角一致性**: 它们没有指标来衡量“生成的背面”是否与“输入的正面”在几何逻辑上匹配。

与 "Ego-Exo Datasets" (如 Ego4D, Ego-Exo4D) 的区别**现有任务是“分析 (Analysis)”，你是“合成 (Synthesis)”。**Ego4D, Ego-Exo4D, LEMMA。

- **Existing Work**: 像 Ego-Exo4D 这样的大型数据集，主要关注的任务是：
  - **Correspondence**: 找到第一人称某一帧对应的第三人称是哪一帧。
  - **Retrieval**: 从海量视频中检索。
  - **Forecasting**: 预测下一步动作。
  - **Pose Estimation**: 从 Ego 视角推断身体骨架。
  - **非生成任务**: 这些 Benchmark 是判别式的 (Discriminative)，不产生像素。它们只提供 Ground Truth 数据，没有建立“生成质量”的评估标准。
  - **缺乏生成式 Baseline**: 它们没有在数据集上跑过 SVD、DiT 等生成模型，因此无法作为生成任务的直接参考。
- **Your Novelty**: 你不是在**理解**视频，而是在**创造**视频（Pixel-level Generation）。
  - 目前没有 benchmark 致力于“**直接生成**照片级真实的第三人称 RGB 视频”。大多数工作止步于生成 3D 骨架（Skeleton）或简单的 Mesh。

与 "Video-to-Video Translation" (如 ControlNet, Gen-1) 的区别 **现有任务是“风格迁移”，你是“几何重绘”。**

- **Existing Work**: 目前的 V2V（包括 Video-ControlNet, TokenFlow）都有一个隐含假设：**Structure Preservation (结构保留)**。
  - 输入视频里有一个人跳舞，输出视频里这个人还在同样的位置跳舞，只是变成了动漫风格。像素是对齐的。
- **Your Novelty**: 你的输入（手/视野）和输出（背影/全身）在像素空间上**完全不重叠**。
  - 你打破了 V2V 领域赖以生存的 "Structural Guidance" 假设。你的模型必须先在脑海中完成 3D 空间变换，再生成像素。这是一个从 **"Re-texturing" (重纹理)** 到 **"Re-rendering" (重渲染)** 的范式跳跃。

与 "Novel View Synthesis" (如 NeRF, Gaussian Splatting) 的区别 **现有任务是“重建”，你是“幻构”。**

- **Existing Work**: NeRF 或 3D Gaussian Splatting 需要**多视角**图片作为输入来重建一个静态场景。或者需要在一个静态场景中过拟合。
- **Your Novelty**:
  - **Single View Input**: 你只有**单目**的第一人称视频。
  - **Dynamic Scene**: 场景和人都在动。
  - **Hallucination**: 背后有什么？NeRF 没拍到就不知道，但你的模型必须基于常识去**猜（Hallucinate）**出来。这属于 Generative Prior 的范畴，而不是 3D Reconstruction 的范畴。

Reference-Guided & In-Context Generation (参考引导与上下文生成)

**目标**：引出你的方法论核心（IC-LoRA），指出你在动态几何映射上的创新。

**应包含的内容**：

- **ID 保持技术**：引用 IP-Adapter, Reference-Net, AnyDoor 。VACE 也提到了 "Reference-to-Video Generation (R2V)" 。
- **Visual In-Context Learning**：引用 Ali-ViLab 的 Visual ICL 相关工作 (Images Speak in Images 等)。

- **你的 Punchline (打击点/Gap)**：

  > “虽然 R2V 技术  能保持 ID，但它们通常只关注静态外观的复制。Visual ICL 虽然展示了类比能力，但尚未被应用于**长视频的时空几何一致性**任务中。我们的方法是首个将 In-Context Learning 引入 Ego-Exo 视频生成，利用 Grid 输入来隐式学习几何映射算子的工作。”

这个领域目前是一片**蓝海 (Blue Ocean)**，原因如下：

1. **太难了**: 以前的模型（GAN时代）做不到这么大幅度的视角变换还能保持长视频连贯。
2. **数据稀缺**: 直到最近（Ego-Exo4D 发布），才有足够的高质量配对数据来支撑这种 Generative 任务的训练。你是第一批利用这些数据做生成任务的人。

为了坐实“First Benchmark”的地位，你需要在 Related Work 中使用一种 **"围攻"** 的策略：

- **Paragraph 1: Ego-Centric Vision**: 承认 Ego4D 等贡献，但指出它们关注 **Perception (感知)** 和 **Retrieval (检索)**，忽略了 **Generation (生成)**。
- **Paragraph 2: Video Generation & Editing**: 承认 Sora, Gen-2, ControlNet 的强大，但指出它们依赖 **Structural Alignment**，无法处理剧烈的 **Viewpoint Change**。
- **Paragraph 3: Human Motion Synthesis**: 承认有人做 Ego2Skeleton，但指出 **Skeleton $\neq$ Photorealistic Video**。骨架没有纹理、光照和环境互动。
- **Conclusion**: 因此，我们需要 **Ego2ExoFollowShot** —— 第一个致力于直接从 Ego 视频生成照片级真实 Exo 跟随视角的 Benchmark。

| **Benchmark**     | **Focus**           | **Task Type**   | **Structure Assumption** | **Key Metrics**     | **Blind Hallucination?** |
| ----------------- | ------------------- | --------------- | ------------------------ | ------------------- | ------------------------ |
| **VBench** 11     | Quality/Consistency | T2V / I2V       | N/A                      | Quality, Smoothness | ❌ No                     |
| **Ego4D**         | Understanding       | Analysis        | Aligned                  | Accuracy, mAP       | ❌ No                     |
| **VACE-Bench** 12 | Multi-task Editing  | R2V, V2V, Edit  | **Aligned / Style**      | Similarity, MOS     | ❌ No (Mostly)            |
| **ControlVideo**  | Controllability     | V2V (Structure) | **Strictly Aligned**     | Fidelity            | ❌ No                     |
| **Ours**          | **Cross-View Gen**  | **Ego2Exo**     | **Non-Aligned**          | **Geo-Consistency** | ✅ **Yes**                |

你设计的 `VideoDataset` 同时支持 `mode="train"` 和 `mode="evaluate"`，这其实是一种**非常高级且工程化**的设计（通常被称为 **Unified Data Pipeline**）。这种设计有三个巨大的优势，是 VBench 那种简单加载比不了的：

在计算机视觉（CV）和人工智能（AI）的顶级会议（CVPR, ICCV, ECCV）中，凡是定义**“映射类（Mapping）”**或**“特定领域生成（Domain-Specific Generation）”**的新任务，其 First Benchmark 几乎**毫无例外**都采用了“训练集 + 评估集”的模式。

只有当任务是“通用能力测评”（如 VBench 测评现有的大模型）时，才会只给 Prompt。而你的 **Ego2Exo** 属于**定义一种新的映射逻辑**，必须提供训练数据来“定义”这种映射。

赛道 A：标准闭集设定 (Standard / Closed Track)

- **规则**：参赛者**只能**使用你提供的 `Ego2Exo-Train` 数据集进行训练或微调。
- **目的**：**比拼算法效率与架构优势**。
  - 在相同的数据量（比如仅 1000 个视频）下，谁的模型能学得更快、泛化得更好？
  - 这对学术界（特别是算力有限的高校实验室）非常重要，保证了公平性。

赛道 B：开放设定 (Open / Unconstrained Track)

- **规则**：参赛者可以使用**任何外部数据**（比如自己去 YouTube 爬取百万小时的 Ego 视频，或者使用私有数据），甚至可以使用更大的预训练模型。
- **目的**：**探索任务的性能上限 (Upper Bound)**。
  - 验证“大力出奇迹”在这个任务上是否奏效。
  - 鼓励工业界（Google, Meta）通过扩大数据规模来解决问题。

## Method

### 任务定义

这个任务在学术上通常被归类为 **Cross-View Video Generation (跨视角视频生成)** 或 **Controlled Video-to-Video Translation**。我们将 **Ego2ExoFollowShot 条件可控生成** 建模为一个 **多条件去噪扩散过程**。

给定第一人称视频序列 $\mathbf{X} \in \mathbb{R}^{T \times H \times W \times 3}$，参考图像 $\mathbf{I}_{ref} \in \mathbb{R}^{H \times W \times 3}$，以及文本指令 $\mathbf{p}$。目标是学习一个参数化的分布 $P_\theta$，以生成目标视频 $\mathbf{Y}$：
$$
\mathbf{Y} \sim P_\theta(\mathbf{Y} \mid \mathbf{X}, \mathbf{I}_{ref}, \mathbf{p})
$$
**$\sim$ (Sample from)**: 这表示 **生成过程**。不仅仅是一个函数映射 $Y=f(X)$，而是一个**概率采样**。这意味着对于同一个 Ego 视频，可能存在多种合理的 Exo 视角演绎（不确定性），你的模型是在学习这个概率分布。

**$P_\theta$**: 代表你的 **DiT 模型**。$\theta$ 是模型的参数（在这里特指 **Base Model + LoRA** 的参数）。

**$\mid$ (Conditioned on)**: 竖线右边是**所有已知条件**。模型必须在“看到”这些东西的前提下，才能画出左边的 $\mathbf{Y}$

$\mathbf{Y}$ (Target Exo Video) 待生成的第三人称视角视频。

- $\mathbf{Y} \in \mathbb{R}^{T \times C \times h \times w}$ 。
- **输入关系**: 在训练时，它是加了噪声的 $Y_t$；在推理时，它是纯高斯噪声 $\epsilon \sim \mathcal{N}(0, I)$。

 $\mathbf{X}$ (Ego Video Condition)提供动作和时序节奏的第一人称视频。

- $\mathbf{X} \in \mathbb{R}^{T \times C \times h \times w}$。
- **具体维度**: 与 $\mathbf{Y}$ 完全一致 
- **输入关系**: 它是“干净”的（不加噪）。

 $\mathbf{I}_{ref}$ (Reference Image / Support Exo) 提供人物 ID 和外观特征的参考图。

- **数学空间**: $\mathbf{I}_{ref} \in \mathbb{R}^{C \times h \times w}$ (Image Latent) 或 $\mathbb{R}^{1 \times C_{clip} \times D}$ (Embedding)。
- **输入关系**: 它负责“填色”，确保生成的 $\mathbf{Y}$ 穿的衣服和 $\mathbf{I}_{ref}$ 一样。

条件变量 $\mathbf{p}$ (Text Prompt) 任务指令和语义描述。

- **数学空间**: $\mathbf{p} \in \mathbb{R}^{L \times D_{text}}$。
- **具体维度**:$L$: Token Length $D_{text}$: Text Embedding Dimension (e.g., 1024 or 4096)。

注：设计 Prompt 时刻意剥离对 Video 和 Reference Image 的具体描述（例如不写“红衣服”、“正在跑步”），不仅是工程经验，更有深刻的数学原理和训练动力学考量。在数学上，我们将生成模型建模为条件概率分布 $P(\mathbf{Y} | \mathbf{X}, \mathbf{I}_{ref}, \mathbf{p})$​。为了让模型高效学习，我们希望条件变量之间是**正交（Orthogonal）**或**独立（Independent）**的。**VACE 的做法**：引入了 **Concept Decoupling** 模块 。它明确区分了“Reactive Frames”（需要改变的像素）和“Inactive Frames”（需要保留的像素）。设计了 **Video Condition Unit (VCU)**，将文本、图像、视频帧序列和掩码序列统一为一个标准化的输入范式 。通过 **Concept Decoupling（概念解耦）** 策略，明确区分需要保留的视觉信息和需要修改的控制信号，从而更好地指导模型生成 。

**原理迁移**：VACE 认为显式地分离不同模态和分布的数据对于模型收敛至关重要 。

**对你的启示**：同样地，你需要将 **Semantic Concepts (语义概念，由 Prompt 负责)** 和 **Visual Concepts (视觉概念，由 Ref/Video 负责)** 显式分离。Prompt 只负责定义“任务框架”，具体的“填充内容”必须完全交给视觉输入。

不描述 Video 和 Reference 的具体内容，是为了：

1. **强制解耦**：让 Prompt 专注于**“Operator (操作逻辑)”**，让 Visual Input 专注于**“Data (数据内容)”**。
2. **防止作弊**：逼迫模型学习复杂的 Visual-to-Visual 映射，而不是简单的 Text-to-Visual 映射。
3. **提升泛化**：实现 Zero-shot ID 能力。Prompt 越抽象，模型对不同 Reference Image 的适应性就越强。

**prompt核心职责**：

1. **Viewpoint Definition (视角定义)**：
   - *Content:* "Third-person view, back view, following shot."
   - *作用:* 告诉模型相机的虚拟位置。这是最重要的 Prompt 功能。
2. **Task Trigger (任务触发)**：
   - *Content:* "Generate a video consistent with the reference grid."
   - *作用:* 激活 LoRA 中关于 Ego2Exo 映射的特定权重。
3. **Quality Booster (画质增强)**：
   - *Content:* "4k, high quality, temporal consistent, realistic physics."
   - *作用:* 压制生成过程中的伪影和噪声。

**ref核心职责**：

- **Identity (身份)**：不仅仅是“一个男人”，而是“这个穿 Supreme 红色卫衣、留着寸头的男人”。
- **Texture (材质)**：衣服的褶皱感、头发的光泽。
- **Color Palette (色调)**：确保生成的 Exo 视频色调与参考图一致。

这是一个非常敏锐且关键的问题。虽然你和IDE看起来都在“使用第一帧”，但**答案是否定的：IDE 无法完成你的任务（EGO2EXOTPS）**。

根本原因在于：你们虽然都用了“第一帧”，但对这张“第一帧”的**依赖逻辑**和**处理方式**是完全不同的。

以下是详细的技术拆解，告诉你为什么 IDE 的路线走不通你的任务：

1. “第一帧”的定义不同：是“参考”还是“底板”？

- **IDE 的第一帧 = “底板” (Canvas / Anchor)**
  - IDE 拿到的第一帧，必须是**该场景下、该时刻的真实 Exocentric 截图** 11。
  - 它包含了：**正确的背景**、**正确的环境光照**、**人物正确的起始姿态**。
  - IDE 的工作逻辑是：如果不动，画面就是第一帧；如果动了，就根据 Ego 视频计算出的光流，把第一帧的像素**推（Warp）**到新位置 222。
- **你的第一帧 = “参考” (Reference / Identity)**
  - 你的输入是**任意一张**包含该角色的图片（或者 LoRA）。
  - 这张图的背景可能和 Ego 视频里的环境完全无关（比如你在绿幕前拍的第一帧，但 Ego 视频是在厨房）。
  - 你的工作逻辑是：提取这个人的长相特征（ID），结合 Ego 视频的动作，**重新画（Generate/Denoise）** 出每一帧新的像素。

2. 技术瓶颈：Warping vs. Generation

IDE 无法完成你的任务，核心在于它使用了 **Warping（扭曲/变形）** 技术。

- IDE 的做法：

  IDE 的核心模块是预测“光流（Optical Flow）”和“遮挡图（Occlusion Map）” 。它的输出公式是：$\tilde{z}=m\otimes\mathcal{W}(z,f)$ 4。

  这意味着：它只能搬运第一帧里已有的像素。

  - *如果你的任务是：* 第一帧是这人在卧室站着，Ego 视频是这人走到厨房。
  - *IDE 的结果：* 只要人走出第一帧的画面范围，或者转身露出了第一帧里没拍到的背面，模型就会因为“没有像素可以搬运”而产生巨大的伪影或模糊（Inpainting 并不是它的强项）。

- 你的任务需求：

  你需要的是 Hallucination（合理的幻觉/生成）。

  - 如果第一帧是正面，Ego 视频里人转过去了，你的模型必须能**凭空生成**这个人的背影。这是 IDE 的 Warping 逻辑做不到的。

3. 空间对齐的硬性要求

- IDE 的假设：IDE 假设输入的 Exocentric 第一帧与 Egocentric 视频的第 0 秒在时间空间上是严格对齐的 5555。它需要从这两者之间建立特征对应关系（CFPM 模块） 6。

  如果你随便给一张角色的照片（比如一张艺术照），它和 Ego 视频里的相机轨迹、人物位置完全对不上，CFPM 模块会失效，导致生成失败。

4. 总结：如果强行用 IDE 做你的任务会发生什么？

假设你强行把一张“自定义角色图”喂给 IDE，并给一段厨房做饭的 Ego 视频：

1. **背景崩坏：** IDE 会试图把你的角色图的背景（比如白色背景）强行“扭曲”成厨房的样子，结果会是一团浆糊。
2. **动作撕裂：** 除非你的角色图姿势和 Ego 视频起始姿势一模一样，否则 IDE 会试图把一个站着的人强行扭曲成做饭的动作，导致肢体变形。
3. **无法泛化：** IDE 严重依赖它训练过的 LEMMA 数据集场景 7。它并没有学习“如何把一个人画进新环境”，它只学习了“如何根据头动推测身子动”。

**IDE 是一个“驱动器（Animator）”，而你的模型是一个“生成器（Generator）”。**

- IDE 适合：监控摄像头丢帧了，用第一人称视角把中间几秒**补回来**。
- 你的模型适合：游戏/电影制作，只有动作捕捉数据（Ego），想生成**任何角色**演这场戏。

所以，放心大胆地去陈述你的方法，你的任务难度和通用性在生成领域是高于 IDE 的。你的 "Reference 提取" 虽然也是基于图像，但你是提取 **Feature (特征)** 用于 Condition，而 IDE 是提取 **Pixels (像素)** 用于 Warping。这是本质的区别。

这是一个非常棒的切入点！**“Follow Camera”（跟随镜头）** 这个设定不仅让你的任务看起来更高级（像游戏或电影），而且从技术原理上**彻底判了 IDE 方法的“死刑”**。

因为 IDE 的“Warping（扭曲）”逻辑必须建立在**背景几乎不动**或者**背景已知**的前提下。如果是 Follow Camera，背景会随着人物移动而不断后退、切换，这意味着每一帧都有大量的“新背景”进入画面，IDE 的光流法根本无法处理这种无限出现的未知区域。

以下是如何利用“Follow Camera”这一点来重新包装你的任务定义和命名：

一、 建议的新任务命名 (加入“Dynamic/Follow”概念)

你需要强调视角的**动态性 (Dynamic)** 和**以人为中心 (Agent-Centric)**。

选项 A：强调游戏感的“跟随视角”

**"Third-Person Follow-View Generation from Egocentric Videos"** **（基于第一人称视频的第三人称跟随视图生成）**

- **解析：** "Follow-View" 是游戏工业标准术语，直接暗示了相机是锁死在角色身后或侧后方的，与监控摄像头（Surveillance/Fixed Camera）截然不同。

选项 B：强调“动态相机”

**"Dynamic Camera Third-Person View Synthesis via Ego-Motion"** **（基于第一人称运动的动态相机第三人称视图合成）**

- **解析：** "Dynamic Camera" 强调了相机本身的运动，这意味着背景是流动的，暗示了任务必须具备强大的**Open-World Hallucination（开放世界生成）**能力。

选项 C：最严谨的学术命名

**"Ego-to-Exo Video Re-enactment with Cinematic Camera Tracking"** **（具有电影级相机追踪的 Ego-to-Exo 视频重演）**

- **解析：** "Cinematic Camera Tracking" 把你的任务拔高到了电影制作辅助工具（Pre-visualization）的层面，完全脱离了 LEMMA/IDE 的家庭监控场景。

这是一个非常关键的实验设计问题。在 `Ego2Exo` 任务中，AnimateDiff + ControlNet 需要 **骨架视频 (Pose Video)** 作为输入，但我们实际上只有第一人称视频（Ego）作为条件，第三人称视频（Exo）是我们要预测的目标（GT）。

如果我们直接把 GT Exo 的骨架提取出来给 AnimateDiff，它就“偷看”了答案（Motion），这对你的模型（必须从 Ego 视频推断 Motion）是不公平的。

为了解决这个问题，并保证论文的严谨性，我建议采取 **“Oracle Baseline（上界基线）”** 的策略，并配合一套 **Pose 提取流水线**。

------

如何解决公平性问题：重新定义 Baseline 的角色

你无法在“盲测”条件下公平地使用 AnimateDiff + ControlNet，因为它不具备“从第一人称推断第三人称动作”的能力。

**解决方案**：将 AnimateDiff 定义为 **“Oracle Motion Baseline” (动作上界基线)**。

- **操作方法**：明确告诉 Reviewers，为了评估图像生成质量和 ID 保持能力的 **上限**，我们允许 AnimateDiff 访问 Ground Truth 的动作（即从 GT Exo Video 提取 Pose）。

- **论文论点 (Argument)**：

  > "由于 AnimateDiff 无法处理跨视角运动预测，我们为其提供 Ground Truth 骨架作为输入。这构建了一个 **'Oracle' 基线**。我们的目标是证明：即使没有 GT 动作的引导，我们的 End-to-End 模型也能产生接近这个 Oracle 基线的视频质量和 ID 一致性，同时还能正确预测出动作。"

- **评价指标处理**：

  - **Motion Metrics (HAA, Trajectory)**: AnimateDiff 在这些指标上会得满分（或极高分），因为它是看着答案跑的。在表格中给它的这些分数打上 `*` 号，注明是 Reference。
  - **Quality Metrics (FVD, AC)**: 这是**真正需要对比**的地方。如果你的 FVD 和 AC 接近甚至超过 AnimateDiff，说明你的模型非常强大。

------

二、 在论文中划清界限 (The "Killer" Distinction)

你可以在论文中加入一个段落，专门讨论 **"Camera Behavior Constraints"（相机行为约束）**，以此证明 IDE 方法的不适用性。

**建议的论述逻辑：**

> **1. IDE / LEMMA 的局限：固定视锥 (Fixed Frustum)** "Existing benchmarks like LEMMA and methods like IDE  operate under a **'Fixed Surveillance Assumption'**. They utilize stationary cameras (e.g., Kinects mounted in corners).
>
> In this setting, the background is static. The task is reduced to inpainting the background revealed by the moving agent. This allows warping-based methods to succeed by simply copying background pixels from the reference frame." *(现有的基准如LEMMA和方法如IDE都在“固定监控假设”下运行。它们使用固定相机（如安装在角落的Kinect）。在这种设置下，背景是静态的。任务被简化为修补被移动代理遮挡的背景。这使得基于扭曲的方法可以通过简单地复制参考帧中的背景像素来成功。)*

> **2. 你的任务挑战：动态视锥 (Dynamic Frustum / Infinite Background)** "Our task adopts a **'Follow-Camera Paradigm'**. The virtual camera is tethered to the agent's trajectory, maintaining a relative distance and angle.
>
> This introduces the **'Infinite Background Problem'**: as the agent moves forward, the camera traverses through the scene, continuously revealing entirely new environments that never appeared in the first frame. Warping-based methods (like IDE) fundamentally fail in this scenario as they cannot retrieve pixels for non-existent areas. Therefore, our task necessitates a generative model capable of **temporal hallucination** of dynamic environments." *(我们的任务采用“跟随相机范式”。虚拟相机绑定在代理的轨迹上，保持相对的距离和角度。这引入了“无限背景问题”：随着代理向前移动，相机穿越场景，不断揭示第一帧中从未出现的全新环境。基于扭曲的方法（如IDE）在这种情况下会从根本上失效，因为它们无法获取不存在区域的像素。因此，我们的任务必须需要一个具备动态环境“时间幻觉/生成能力”的生成模型。)*

### Quadruplet Grid Input & Concept Decoupling (四元组输入与概念解耦)

**目标**：介绍你的“作弊条”——即如何通过 Input Design 解决信息缺失。这是你**最核心的创新点**之一。

- **核心内容**：
  1. **Grid Construction (网格构造)**：
     - 详细描述你的四元组结构：`[Support Ego, Support Exo; Query Ego, Target]`。
     - 解释每个象限的作用：Support Set 提供“几何映射范例”，Query Ego 提供“当前驱动信号”。
     - **Storytelling**：强调这种 Grid 形式使得模型能够进行 **Visual In-Context Learning**，即通过“看例子”学会如何把 Ego 变成 Exo，而不是死记硬背。
  2. **Prompt Engineering (概念解耦)**：
     - 引用 VACE 的 **Concept Decoupling** 思想
     - 解释你如何重写 Prompt：剔除内容描述（因为 Reference Image 已经提供了 $\mathcal{C}_{id}$），只保留结构指令（如 "Third-person follow shot"）。
     - 强调这避免了模态竞争（Modality Competition），迫使模型关注 Visual Grid。

### Architecture: In-Context LoRA (架构：IC-LoRA)

**目标**：解释模型如何处理上述输入。

- **核心内容**：
  - **Base Model**：说明你基于 DiT (如 Wan-T2V 或 LTX-Video) 2。
  - **LoRA Injection**：解释 LoRA 加在哪里（通常是 Attention 层）。
  - **Attention Mechanism**：
    - 重点解释 **Self-Attention** 在 Grid 中的行为。
    - 描述 Attention 如何跨越 Grid 的边界，让右下角的 Target 能够 query 到左下角的 Reference（获取纹理）和右上角的 Ego（获取动作）。
    - **Storytelling**：把 Attention 描述为**“Information Bridge” (信息桥梁)**，连接了 Ego 动作与 Exo 外观。

#### Training Strategy: Self-Forcing for Temporal Stability (训练策略：Self-Forcing)

**目标**：解决长视频生成中的“时序崩坏”问题。

- **核心内容**：
  - **Exposure Bias (暴露偏差)**：解释如果不加 Self-Forcing，训练时用 GT，推理时用预测值，会导致分布偏移。
  - **Self-Forcing Algorithm**：
    - 详细描述你在训练时如何按概率 $P$ 将 Support Exo 替换为**“带噪声的预测值”**或**“上一轮的输出”**。
    - **数学本质**：这是一种 **Data Augmentation in Temporal Domain (时域数据增强)**，强迫模型学会从“不完美的历史”中恢复正确的未来。
  - **Loss Function**：写出最终的 Loss 公式（通常是预测噪声 $\epsilon$ 的 MSE Loss，加上可能的 Masking 策略——只计算 Target 区域的 Loss）。

## The Ego2Exo Benchmark

### 数据收集

**同步性证明 (Synchronization Verification)**：

- Ego 和 Exo 必须是严格时间同步的。你是如何保证这一点的？时间戳同步，音频事件对齐
- **Storypoint**：展示你清洗数据的 pipeline，强调“Pixel-level Temporal Alignment”。强调这是一个 **"Paired Video Dataset"**，这种 Pixel-level 的时间对齐是训练 geometric consistency 的物理基础。

**身份的多样性与划分 (Identity Diversity & Split)** 这是 Benchmark 公平性的核心。如果测试集里的人出现在了训练集里，那就是作弊。

Ego2Exo 主要是生成**背影（Follow Shot）**。在背影视角下，分辨一个人的主要特征不是脸，而是**衣服、背包和装备**。**训练逻辑**：如果同一个人今天穿红衣服，明天穿蓝衣服。对模型来说，这就是两个完全不同的“搬运任务”。如果模型记住了“这个人穿红衣服”，那它在处理蓝衣服数据时反而会 Loss 爆炸。

只要衣着和装备发生了显著变化（Pixel-level distribution shift），它们就是不同的**训练实例**。需要注意的“泄露风险” (Leakage Risks) 虽然你可以把它们当作不同身份，但因为还是同一个人，存在两种潜在的 **Data Leakage** 风险，你需要在划分训练/测试集时格外小心：体型与步态泄露 面部特征泄露

明确指出训练集和测试集的 ID 是互斥的 (Mutually Exclusive)。

**Test Set Composition**: 说明测试集不仅仅是随机抽样，而是包含精心挑选的 (Curated) 具有代表性的片段：

- 包含了 **Unseen Subjects** (测试 ID 泛化)。
- 包含了 **Unseen Scenarios** (测试环境泛化)。
- 包含了 **Extreme Viewpoint Changes** (测试几何鲁棒性)。

**Highlight the "Static" Case**: 在描述数据集时，再次强调那个“静态子集”的存在是为了测试 **Temporal Stability**，这是一个非常好的反直觉（Counter-intuitive）设计点，审稿人会喜欢的。

在 Benchmark 数据集构建部分，你需要将 Prompt 定义为**“标准化的任务指令（Standardized Task Instructions）”**。

参考 VACE 论文中关于 Benchmark 构建的描述 ，以下是你应该如何在 Benchmark 部分界定 Prompt 的作用，以及该章节应包含的完整内容：

如何在 Benchmark 中界定 Prompt 的作用？

在 Benchmark 中，Prompt 是**“考题”**。为了公平评估不同模型（特别是 Reference-based 和 Text-based 模型），你需要提供**两种版本**的 Prompt，或者明确 Prompt 的性质。

VACE 的做法非常有参考价值：它提供了**原始 Caption** 用于定量评估，同时也提供了**针对特定任务重写的 Prompt** 。

你应该在论文中这样界定 Prompt 的作用：

- **Prompt 作为“结构引导者” (The Structural Guide)**： 对于 Ego2Exo 任务，Benchmark 提供的标准 Prompt 旨在定义**“视角关系”**（如 "Third-person follow shot"），而不是描述视觉细节。
- **Prompt 作为“模态隔离墙” (The Modality Wall)**： 在 Benchmark 的设计中，我们故意设计了**不包含**外观描述的 Prompt。这是为了**强制 (Enforce)** 被测模型必须具备从 Reference Image 提取外观的能力。如果模型只能依赖文本生成（如 T2V），它在我们的 Benchmark 上得分低是合理的，因为这证明了它缺乏多模态对齐能力。

你需要明确区分**“基线模型（Baselines）”**和**“参考引导模型（Reference-guided Models，如你的模型）”**的需求差异：

- **对于 Baselines (如 CogVideoX, Wan-T2V)**：
  - 它们**看不见** Reference Image。
  - 如果你给它们一个抽象的 Prompt（如“一个角色的第三人称视角”），它们会随机生成一个人物（比如生成了一个穿绿衣服的人，但 GT 是红衣服）。
  - **后果**：这会导致 ID 一致性指标极低，这种比较是没有意义的，因为这是信息不对等造成的，而不是模型能力差。
- **对于 你的模型 (Ours)**：
  - 它们**能看见** Reference Image。
  - 如果你给它们一个详细的 Prompt（如“一个穿红衣服的人”），模型可能会偷懒只看文本，不看 Reference Image。
  - **后果**：无法验证 In-Context Learning 的有效性。

在你的数据集 JSON 文件中，每个视频样本应该包含两个字段：

Descriptive Prompt (描述性提示词)

- **服务对象**：**Baselines (T2V/I2V without reference)**。

- **作用**：作为“公平的补偿”。既然它们看不到图，那你就用文字尽可能详细地把人物长相、衣服颜色、环境细节告诉它们。

- **内容**：包含**外观 (Appearance)** + **动作 (Action)** + **视角 (Viewpoint)**。

- **例子**：

  > "A young man with short black hair, wearing a **red hoodie and blue jeans**, is walking forward in a **modern office**. Third-person follow shot view."

Type B: Structural/Blind Prompt (结构性/盲提示词)

- **服务对象**：**Ours, VACE (Reference-based Models)**。

- **作用**：实施 **Concept Decoupling** 。强制模型忽略文本中的外观信息，必须去“看” Reference Image。

- **内容**：**剔除外观**，仅保留 **结构指令 (Structural Instruction)** + **视角 (Viewpoint)**。

- **例子**：

  > "A **third-person follow shot** of the character, maintaining **geometric consistency** with the provided first-person view input."

```json
{
    "case-1001": {
        "video_paths": {
            "ego": "test/1001/ego.mp4",
            "exo": "test/1001/exo.mp4",
            "ref": "test/1001/ref.png"
        },
        "prompts": {
            "t2v_generic": "A cinematic shot of a woman running in a park, sunny day, high quality.",
            "i2v_generic": "A woman matching the reference image is running in a park, cinematic lighting.",
            "vace_instruct": "TASK-MOTION: A woman running in a park, maintaining the motion flow from the context.",
            "ours_lora": "[EGO2EXO] [REF-ID] A woman with red hair [ACTION] is running [TARGET-VIEW] cinematic follow shot."
        },
        "negative_prompt": "shaking, blurry, distorted, low quality..."
    }
}
```

在我们刚刚设计的 `EgoExoPipelineCaptioner` 中，通过**Prompt 结构**和**Trigger Word** 实现了以下引导机制（虽然代码只写了 Text 端，但它隐含了与 Vision 端的配合逻辑）：

1. 语义对齐（Semantic Alignment）—— "软"引导

我们的 Captioner 并没有只给一个 `sks woman`，而是输出了：

> ```
> [Trigger], woman, red dress, long black hair, silver earrings...
> ```

- **为什么要这么做？** 即使有参考图，模型（尤其是基于 CLIP 的 Cross-Attention）依然需要文本来“唤醒”对应的视觉特征。
  - 当 Prompt 提到 `"red dress"` 时，Text Encoder 产生的 Embedding 会作为 **Query (Q)**。
  - 参考图经过 Visual Encoder (如 IP-Adapter 的 Image Projector) 产生的 Feature 作为 **Key (K)** 和 **Value (V)**。
  - **Q 和 K 发生点积**，模型发现参考图里有一块红色的区域匹配度很高，于是把那块纹理“抄”了过来。
  - **结论**：`VisualCaptioner` 详细描述外貌，是在**帮 Attention 机制做特征匹配（Feature Matching）**。

2. 触发词作为“视觉插槽”（Visual Slot）—— "硬"引导

你说得对，这里的 Trigger Word 不应是用于检索记忆的 ID。在 In-Context Learning 的设计中，它通常扮演 **Visual Placeholder** 的角色。

**在我们的框架中，引导逻辑如下：**

- **训练阶段 (Training)**:
  - 输入：参考图 (Ref) + 目标视频 (Target)。
  - Prompt: `img woman, walking on the street...` (这里 `img` 是特殊 token)。
  - **引导机制**：我们在 Cross-Attention 层中训练模型，强迫它在处理 `img` 这个 token 时，大幅度增加对 Reference Visual Features 的权重。
  - Loss 惩罚：如果模型画的人不像 Reference，Loss 变大。模型为了降低 Loss，学会了**“一看到 `img` 这个词，就去看参考图”**。
  - `[EGO2EXO] [REF-ID] A woman with long red hair and a white dress [ACTION] is cutting vegetables in a kitchen [TARGET-VIEW] captured in a cinematic medium shot from the side.`
- **推理阶段 (Inference)**:
  - Prompt: `img woman, running...`
  - 模型反应：读取到 `img` -> 触发训练好的反射机制 -> 调取参考图特征 -> 生成保持一致的角色。

### 数据清洗

**数据分布与难度分层 (Diversity & Difficulty Split)**：

- 如果所有数据都是“走路”，那这个 Benchmark 太简单了。
- **Storypoint**：你需要把测试集分为 `Easy` (慢速移动), `Medium` (复杂交互), `Hard` (剧烈头部运动/遮挡)。证明你的方法在 Hard 模式下依然有效，而 Baseline 崩溃。

| **子集名称**                      | **数据特征**             | **考察的核心能力**                      | **你的静态数据归属** |
| --------------------------------- | ------------------------ | --------------------------------------- | -------------------- |
| **Subset A: High Dynamics**       | 跑步、运动、快速转头     | **Motion Alignment** (跟得紧不紧？)     |                      |
| **Subset B: Complex Interaction** | 手部操作、拿取物体       | **Fine-grained Control** (细节对不对？) |                      |
| **Subset C: Stationary/Stable**   | **办公、坐姿、站立对话** | **Temporal Stability** (稳不稳？)       | **<-- 放在这里**     |

### 指标设计



### 人类评估



## Experiments

### ICL+self-forcing的实现细节

VACE论文提到，由于缺乏可比较的“All-in-One”模型，他们主要与“专有任务模型（Task-specific models）”进行比较 。

你的策略应完全复刻这一逻辑：要实现“**专用模型逻辑（Ego2Exo 映射） + 通用人物适应（Zero-shot ID Generalization）**”，你的模型必须学会**“动词”**（如何转换视角），而不是死记硬背**“名词”**（某个具体的人）。

**普通 LoRA (Overfitting)**：如果你只用一个人的数据训练，LoRA 权重里就会刻下这个人的长相。不管输入是什么，它都倾向于画这个人。

**你的目标 (In-Context/Reference LoRA)**：你要训练 LoRA 学会一个**“搬运机制” (Copy-Paste Mechanism)**。

- LoRA 的任务是：在 Grid 输入中，看到左边 Support Set 里的人穿红衣服、短头发，就把这个纹理特征**搬运**到右边的 Query Exo 里去，同时结合 Ego 视角的动作。
- **关键点**：人物的身份信息（ID）存在于**输入像素（Reference Video/Image）\**里，而不是存在于\**模型权重**里。

### Comparison with State-of-the-Art Methods (SOTA 对比)

Baseline 的“多样性”与“公平性” (Baseline Fairness)

| **Baseline 类别**                 | **代表模型**                   | **你的评价逻辑 (Storytelling)**                              |
| --------------------------------- | ------------------------------ | ------------------------------------------------------------ |
| **Type A: Image-to-Video (I2V)**  | WanI2V, SVDLX, CogVideoX       | **测试“幻构能力”**。 输入：参考图+Prompt。 结果：证明它们虽然画质好，但动起来完全不听话，无法跟随Ego视角的动作节奏。 |
| **Type B: Unified/Edit Models**   | WanVACE (Zero-shot), LTX-Video | **测试“多模态理解能力”**。 输入：Ego视频+参考图（通过VCU接口）。 结果：证明即使是VACE这样的强模型，在没有经过你设计的IC-LoRA+Self-forcing特训前，也难以理解Ego到Exo的复杂几何变换。 |
| **(可选) Type C: Control Models** | VideoComposer, ControlVideo    | **测试“结构控制能力”**。 结果：证明强结构约束在Ego2Exo任务中反而失效（因为视角不同），凸显你方法的灵活性。 |

### Qualitative Results (定性结果 / 视觉展示)分析

*审稿人第一眼看图，第二眼才看表。图要做得漂亮。*

- **Figure X (主要对比图):** 选取 3 个典型案例（例如：人物转弯、快速奔跑、复杂背景）。
  - 每一行是一个方法（Baseline A, Baseline B, Ours）。
  - **标注：** 用不同颜色的框在图中标出问题（例如：Baseline 把人画丢了用红框，你的背景很稳用绿框）。
- **Figure Y (多样性展示):** 展示你的模型可以处理不同的角色（男女老少）、不同的背景（城市、野外）。
- **Figure Z (长视频稳定性):** 如果你做了长视频生成，展示 ID 和背景在第 1 帧和第 N 帧的对比。

### 消融实验 Ablation Study

评估体系的“人类视角” (Human Evaluation)

：Ablation Study (消融实验) —— 证明你的创新点

### 用户主观评测

*对于“美感”和“稳定性”，机器指标（如 FVD）有时会失真，人类的评价至关重要。*

- **设计:** 邀请 N 名用户，每人观看 M 组视频（A vs B 盲测）。
- **问卷维度:**
  1. **Visual Quality:** 哪个画质更清晰？
  2. **Camera Stability:** 哪个运镜更平滑、更像专业跟拍？
  3. **Subject Consistency:** 哪个更像参考图里的人？
- **展示方式:** 柱状图（Bar Chart），显示选择 "Ours" 的百分比（通常希望 > 50%）。

## Discussion & Future Work

**复杂遮挡处理 (Occlusion Handling):**

- *问题:* 当主角走到障碍物（如树、墙）后面时，模型是否还能保持完美的跟随视角？还是会把人画在障碍物前面（图层错误）？
- *话术:* "虽然我们的模型在开阔场景表现优异，但在处理严重遮挡（severe occlusion）时，偶尔会出现深度歧义。"

**极端动作或姿态 (Extreme Poses):**

- *问题:* 如果参考视频里的人在做非常复杂的体操动作，生成模型是否会把四肢画扭曲？
- *话术:* "对于训练集中罕见的复杂非刚性形变（rare non-rigid deformations），模型的骨骼对齐能力有待提升。"

**长时序一致性 (Long-term Consistency):**

- *问题:* 如果生成 1 分钟以上的长视频，衣服纹理或背景风格会不会逐渐“漂移”？
- *话术:* "在极长序列生成中，维持微小的纹理细节一致性仍然是一个开放性挑战。"

**推理速度 (Inference Speed):**

- *问题:* 你的模型加了很多控制模块，是否变慢了？
- *话术:* "为了换取高可控性，我们的推理时间略有增加。未来的工作将集中在蒸馏（Distillation）和加速上。"



## 和数学相关的附录

你的 **Ego2Exo + IC-LoRA + Self-Forcing** 方案本质上是在解决一个 **高维条件概率密度估计 (High-dimensional Conditional Density Estimation)** 问题，同时通过 **低秩流形近似 (Low-Rank Manifold Approximation)** 和 **分布纠偏 (Distribution Shift Correction)** 来优化求解过程。

**你的方案 (IC-LoRA Grid)**：是一种**隐式**的 IP-Adapter。你通过拼接图片，让 Self-Attention 承担了提取特征的任务。

**显式方案 (IP-Adapter)**：如果你发现 Grid 方式效果不好（比如 ID 还是变了），你可以**显式地引入一个 IP-Adapter 模块**。

- 把 Support Exo 的第一帧作为 Reference Image 喂给 IP-Adapter。
- IP-Adapter 会把这个人的特征变成 Embedding，注入到 Cross-Attention 中。
- **LoRA 依然负责 Ego2Exo 的几何变换**，而 **IP-Adapter 负责 ID 的维持**。
- **兼容性**：这两个是可以同时训练，或者冻结 IP-Adapter 只训 LoRA 的。



```
from enum import Enum

class PromptStrategy(Enum):
    T2V_GENERIC = "text-only"      # 对应 Captioner 的 _caption_text_only
    I2V_GENERIC = "text-image"     # 对应 Captioner 的 _caption_text_image
    VACE_INSTRUCT = "vace"         # 对应 Captioner 的 _caption_vace
    OURS_LORA = "ours"             # 对应 Captioner 的 _caption_ours

# 模型 ID 到 策略 的映射表
MODEL_REGISTRY = {
    # Text-to-Video Baselines
    "CogVideo": PromptStrategy.T2V_GENERIC,
    "LTX-T2V":  PromptStrategy.T2V_GENERIC,
    
    # Image-to-Video Baselines
    "SVDTX":    PromptStrategy.I2V_GENERIC,
    "WanI2V":   PromptStrategy.I2V_GENERIC,
    "LTX-I2V":  PromptStrategy.I2V_GENERIC,
    
    # Specialized Models
    "WanVACE":  PromptStrategy.VACE_INSTRUCT,
    "EgoGen":   PromptStrategy.OURS_LORA  # 您的模型
}

def get_strategy(model_name: str) -> str:
    """根据模型名获取对应的 captioner 策略字符串"""
    return MODEL_REGISTRY.get(model_name, PromptStrategy.T2V_GENERIC).value
```

