## Fréchet Video Distance

FVD 的核心思想不是比较“像素”，而是比较“特征分布”。特征提取 (Feature Extraction):我们把所有视频（生成的和真实的）都扔进一个预训练好的 I3D (Inflated 3D ConvNet) 网络中。为什么是 I3D？因为它是在 Kinetics-400（大规模动作视频数据集）上训练的，它能同时理解空间内容（画面是什么）和时间动态（动作怎么动）。我们会取 I3D 网络倒数第二层的输出（通常是 2048 维或者 400 维的向量），这代表了视频的高层语义特征。高斯假设 (Gaussian Assumption):我们将所有“真实视频”的特征看作一团点云，假设这团点云符合多维高斯分布（正态分布）。同样，把“生成视频”的特征也看作另一团符合高斯分布的点云。弗雷歇距离 (Fréchet Distance):FVD 计算这两个高斯分布（两个“椭圆球”）之间的距离。$$FVD = ||\mu_r - \mu_g||^2 + \text{Tr}(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2})$$$\mu$: 特征的均值（中心位置）。$\Sigma$: 特征的协方差矩阵（形状和朝向）。直观理解：如果 FVD=0，说明生成视频的分布和真实视频完全重合（完美）。数值越小越好.

## Aesthetic Quality

Aesthetic Quality 的核心目标是将“美”这个主观概念量化。基本假设：由于深度学习模型（如 CLIP）已经学习了极其丰富的图像语义特征，我们只需要训练一个轻量级的回归网络 (Regressor)，将 CLIP 的特征向量映射到一个 1-10 分的人类评分标准上。公式逻辑：$$Score = MLP(Encoder(Image))$$Encoder: 通常使用 CLIP ViT-L/14。它看到的不仅是像素，还有构图、色彩、风格等高层信息。MLP: 一个简单的多层感知机（Linear -> ReLU -> Linear...），在 LAION-Aesthetics 或 AVA 数据集上训练过。这些数据集包含了成千上万张由人类打分（1分=丑陋，10分=极美）的图片。视频美学：对于视频，我们通常计算每一帧的美学得分，然后取平均值：$$VideoScore = \frac{1}{T} \sum_{t=1}^{T} Score(I_t)$$

## Imaging Quality

对于生成视频，我们通常使用 No-Reference (无参考) 的图像质量评估（NR-IQA）。因为我们没有“完美的高清原图”来逐像素对比，只能让模型根据图像本身的特征来判断“它是否像一张高质量的照片”。目前最先进的方法是 MUSIQ (Multi-scale Image Quality Transformer)。核心思想：图像质量在不同尺度上的表现不同（例如噪点在局部，构图在全局）。MUSIQ 使用一个多尺度的 Transformer，将图像切成不同大小的 Patch 输入网络，最后预测一个 MOS (Mean Opinion Score) 分数。公式逻辑：$$Score = \text{Transformer}(\text{Multi-Scale Patches}(I))$$.

## Temporal Flickering

数学原理 (The Math)Temporal Flickering (时序闪烁)核心思想：如果视频是连贯的，那么第 $t$ 帧的内容应该能通过第 $t-1$ 帧推导出来。如果推导不出来（比如像素突然跳变、噪点闪烁），那就是“Flickering”。计算逻辑：计算从 $t-1$ 到 $t$ 的光流 $F$。利用光流 $F$ 将第 $t-1$ 帧扭曲（Warp）成第 $t$ 帧的样子，得到 $I'_{t}$。计算真实第 $t$ 帧 $I_t$ 与预测帧 $I'_{t}$ 之间的误差（通常是 L2 距离）。$$E_{flicker} = || M \odot (I_t - \text{Warp}(I_{t-1}, F)) ||^2$$(注：$M$ 是遮挡掩码，为了简化，我们可以忽略遮挡或简单处理)

## Motion Smoothness

Motion Smoothness (运动平滑度)核心思想：物体的运动应该是平滑的，不应该有剧烈的加速度突变。计算逻辑：计算光流场在时间上的变化率（即加速度）。$$E_{smooth} = || F_t - F_{t-1} ||^2$$(光流本身是速度，光流的差分就是加速度)

## Dynamic Degree

数学原理 (The Math)Dynamic Degree 的设计初衷是为了防止模型“作弊”。作弊现象：有些生成模型为了提高画质（FVD/IS），倾向于生成几乎不动的“PPT视频”。这种视频画质好，但失去了视频的意义。计算逻辑：计算所有像素移动距离的平均值。$$DD = \frac{1}{T-1} \sum_{t=1}^{T-1} \left( \frac{1}{H \cdot W} \sum_{x,y} || \text{Flow}_t(x,y) ||_2 \right)$$Flow: 光流向量 $(u, v)$。Norm: 向量长度 $\sqrt{u^2 + v^2}$。直观理解：数值越大，视频里的东西动得越剧烈。

## Camera Centering Error

Camera Centering Error (相机中心误差)目标：衡量生成的主角是否位于画面中心（符合第三人称跟随视角的构图习惯）。计算：计算主角框中心 $(cx, cy)$。计算画面中心 $(W/2, H/2)$。计算两点间的欧氏距离，并归一化（除以画面半对角线长度），使其范围在 $[0, 1]$。$$Error = \frac{\sqrt{(cx - W/2)^2 + (cy - H/2)^2}}{\sqrt{(W/2)^2 + (H/2)^2}}$$0.0: 完美居中。1.0: 主角在角落里。Viewpoint Validity (视角有效性)目标：衡量生成的视频中是否包含有效的人物目标。计算：$$Validity = \frac{\text{Count}(\text{Frames with Person Detected})}{\text{Total Sampled Frames}}$$这其实就是我们之前代码里的 detection_rate。

## Appearance Consistency / Long-term ID Stability

逻辑很简单：
利用现有的检测器（Detection Model）把画面中的“人”抠出来（Crop）。
把抠出来的人扔给 DINOv2 提取特征。
计算与参考图（Reference Image）的相似度。

$$Score_{ID} = \frac{1}{N_{samples}} \sum_{i \in Samples} \frac{E(C_i) \cdot E(I_{ref})}{||E(C_i)|| \cdot ||E(I_{ref})||}$$

- $C_i$: 第 $i$ 帧检测并 Crop 出来的人物图像。
- $I_{ref}$: 参考的人物图像。
- $E(\cdot)$: DINOv2 特征提取器。
- 公式即为这两个特征向量的余弦相似度（Cosine Similarity）。

## Viewpoint Validity / Subject Detection Rate

核心思想：衡量生成模型是否“崩坏”，即是否生成了无法被识别为“人”的扭曲图像。

计算逻辑：

设 $D(I_t)$ 为检测器（如 Faster R-CNN）在第 $t$ 帧的输出置信度分数。设定阈值 $\tau$（你的代码中为 0.7）。

$$SDR = \frac{1}{T} \sum_{t=1}^{T} \mathbb{1}( \max(D(I_t)) > \tau )$$

其中 $\mathbb{1}(\cdot)$ 是示性函数，满足条件为 1，否则为 0。

## Background Semantic Consistency 

核心思想：

在第三人称跟随视角（Third-person Follow Shot）中，随着人物移动，背景像素会发生剧烈变化。该指标旨在衡量生成视频的背景在语义风格上是否保持稳定，以及是否忠实于参考图像的场景设定，而非像素级的静止。

**数学原理**：

1. **掩码生成 (Mask Generation)**: 利用检测模型生成第 $t$ 帧的人物边界框 $B_t$。构建二进制掩码 $M_t$，其中人物区域为 0，背景区域为 1。

2. 背景提取 (Background Extraction): 将原图与掩码相乘，得到仅包含背景的图像 $I_{bg, t}$。

   $$I_{bg, t} = I_t \odot M_t$$

3. 语义特征提取 (Semantic Embedding): 使用冻结的 CLIP Image Encoder $E(\cdot)$ 提取背景的特征向量。

   $$v_{gen, t} = E(I_{bg, t}^{gen}), \quad v_{ref} = E(I_{bg}^{ref})$$

4. 一致性计算 (Consistency Calculation): 计算生成背景特征与参考背景特征之间的余弦相似度。

   $$Score_{BSC} = \frac{1}{T} \sum_{t=1}^{T} \text{CosineSim}(v_{gen, t}, v_{ref}) = \frac{1}{T} \sum_{t=1}^{T} \frac{v_{gen, t} \cdot v_{ref}}{||v_{gen, t}|| \cdot ||v_{ref}||}$$

这是一个非常深刻且正确的见解。你指出了**生成任务（Generation）与重建任务（Reconstruction）**之间的核心区别。
为什么应该移除 Optical Flow Correlation (OFC)？
正如你所说，在你的任务中，Ground Truth (GT) 视频包含了“缺陷”（路面颠簸、手抖），而你希望生成模型能够超越 GT，产生像游戏视角一样平滑、稳定的运镜。
如果使用 OFC：
高分陷阱：模型必须完美复刻 GT 的每一次抖动和颠簸，才能拿到高分。
低分误杀：如果模型生成了完美的“斯坦尼康”式平滑跟随镜头，而 GT 在晃动，OFC 分数会很低（不相关甚至负相关）。

## Human Action Alignment

核心目标：

衡量生成视频中人物的骨骼姿态（Pose）与参考视频（Ground Truth）的一致性，确保动作语义未发生改变。

**计算逻辑**：

1. **骨骼提取 (Keypoint Extraction)**: 使用预训练的姿态估计模型（如 Keypoint R-CNN）提取每一帧人物的 $K$ 个关键点坐标 $P = \{p_1, ..., p_K\}$（COCO 标准为 17 个点，包括鼻子、肩膀、手肘、膝盖等）。

2. 肢体向量构建 (Limb Vector Construction): 根据人体拓扑结构，定义一组肢体连线 $L$（例如：左肩$\to$左肘，左肘$\to$左腕）。计算每个肢体的向量：

   $$v_j = p_{end} - p_{start}, \quad j \in L$$

3. 向量归一化 (Normalization): 对肢体向量进行归一化，只关注方向（姿态的角度），忽略肢体长度（胖瘦/距离）的影响。

   $$\hat{v}_j = \frac{v_j}{||v_j|| + \epsilon}$$

4. 余弦相似度 (Cosine Similarity): 计算生成视频与参考视频对应肢体向量的余弦相似度。

   $$Score_t = \frac{1}{|L|} \sum_{j \in L} (\hat{v}_{j}^{gen} \cdot \hat{v}_{j}^{ref})$$

5. 时序聚合: 对所有帧的得分取平均。

   $$HAA = \frac{1}{T} \sum_{t=1}^{T} Score_t$$

   - **范围**: $[-1, 1]$。$1.0$ 表示动作完美对齐。

## Optical Flow Correlation

核心思想：不仅仅要求视频动得平滑（Smoothness），还要求运动的方向和大小与真实物理规律（或参考视频）一致。

从 **Storytelling (讲故事/写论文)** 的角度来看，是否保留 Optical Flow Correlation (OFC)，取决于你想把你的模型定义为哪种角色。

为了帮你做决定，我为你构建了两种不同的 Storytelling 路径，请看哪一种更符合你的论文立意：

故事线 A：名为“忠实的翻译官” (The Faithful Learner)

- **立意**：我们的模型是一个强大的分布学习器 (Distribution Learner)。既然训练数据（Ground Truth）包含了相机的物理颠簸（例如路面反馈、行走时的重心起伏），那么模型如果“学会了”，就应该能复现这些物理特性。
- **逻辑**：
  - Ego 视角的震动通常意味着身体的震动。
  - 如果 Ego 在剧烈震动，而 Exo 视角纹丝不动，说明模型**忽略了输入信号**，或者发生了 **Mode Collapse (模式坍塌)**，只会生成平均化的“安全”视频。
  - **OFC 的作用**：证明模型不仅仅是记住了画面，还学会了输入视频 (Ego) 和输出视频 (Exo) 之间微妙的**物理耦合 (Physical Coupling)**。
- **结论**：**必须保留 OFC**（或者类似的指标）。这能证明你的模型“懂物理”。

故事线 B：名为“聪明的摄影师” (The Intelligent Cinematographer)

- **立意**：我们的模型不仅仅是数据的搬运工，更是数据的优化者。虽然训练数据因为采集条件的限制（如手持拍摄）存在瑕疵（抖动），但我们的模型通过学习潜在的语义，能够生成更符合人类审美（如平滑运镜）的高质量视频。
- **逻辑**：
  - GT 的抖动是“噪声”，我们想要的是“信号”（人物轨迹）。
  - 如果模型自动去除了抖动，这是一种 Feature，而不是 Bug。
  - **OFC 的作用**：它会惩罚这种优化，因此它是**有害指标**。
- **结论**：**放弃 OFC**。转而强调 **Motion Smoothness** (我们的更平滑) 和 **Trajectory Alignment** (我们的轨迹是对的，只是更稳)。

**Pareto Frontier**：几乎所有 AIGC 论文都会画一张图，横轴是 FID (Quality)，纵轴是 CLIP Score (Fidelity)，展示它们的模型如何在这条曲线上寻找最佳平衡点。

计算逻辑：

**光流计算 (Flow Calculation)**: 对于视频的每一帧 $t$，利用 Farneback 算法计算稠密光流场 $F_t \in \mathbb{R}^{H \times W \times 2}$。

全局运动提取 (Global Motion Extraction): 计算每一帧光流的平均向量，作为该时刻相机的全局运动估计。

$$v_t = (\bar{dx}_t, \bar{dy}_t) = \left( \frac{1}{H \cdot W} \sum_{x,y} F_t^{(x)}, \frac{1}{H \cdot W} \sum_{x,y} F_t^{(y)} \right)$$

时序序列构建 (Series Construction): 分别构建生成视频和参考视频的水平运动序列 $X$ 和垂直运动序列 $Y$。

$$X_{gen} = \{\bar{dx}_1^{gen}, ..., \bar{dx}_T^{gen}\}, \quad X_{ref} = \{\bar{dx}_1^{ref}, ..., \bar{dx}_T^{ref}\}$$

相关性计算 (Correlation): 使用皮尔逊相关系数 (Pearson Correlation Coefficient) 计算两者在 X 轴和 Y 轴上的线性相关性，最终分数为两者的平均值。

$$Score_{OFC} = \frac{1}{2} (\rho(X_{gen}, X_{ref}) + \rho(Y_{gen}, Y_{ref}))$$

其中 $\rho$ 为皮尔逊系数，范围 $[-1, 1]$。$1$ 表示运动完全同步，$-1$ 表示运动方向完全相反。





todo list

- [ ] Trajectory Alignment 



## Trajectory Alignment

核心思想：人物在画面中的移动路径应该符合预期。

计算逻辑：

1. 记录每一帧人物检测框的中心点 $P_t = (cx_t, cy_t)$。

2. 形成生成轨迹 $\mathcal{T}_{gen} = \{P_0, ..., P_T\}$ 和参考轨迹 $\mathcal{T}_{ref}$。

3. 计算两条曲线的距离。最常用的是 平均位移误差 (ADE - Average Displacement Error)：

   $$ADE = \frac{1}{T} \sum_{t=1}^{T} || P_{gen,t} - P_{ref,t} ||_2$$

   如果存在时间轴不对齐的情况，则使用 动态时间规整 (DTW - Dynamic Time Warping) 距离。



1. 通用指标 (VBench / SOTA)：现成可调库

这部分指标是业界通用的“及格线”，用于证明你的生成视频是“正常的视频”。你确实大部分参考了 **VBench** (CVPR 2024) 的体系。

- **可以直接调库/复用现有代码的指标：**
  - **FVD (Fréchet Video Distance)**: 源自 StyleGAN-V。这是视频生成的金标准。
  - **Aesthetic Quality**: 基于 CLIP，VBench 中有直接实现。
  - **Imaging Quality**: 基于 MUSIQ，VBench 中有直接实现。
  - **Temporal Flickering / Motion Smoothness / Dynamic Degree**: 这些是基于光流的基础计算，VBench 提供了标准代码。

区别与评价：

这部分没有创新，也不需要创新。使用它们是为了公平对比 (Fair Comparison)。如果别的论文用 FVD，你不用，Reviewer 会质疑。

------

2. 任务特定指标 (Ego2Exo Custom)：需要组合开发

这部分指标虽然用到了现成的模型（如 CLIP, DINO, Keypoint R-CNN），但**计算逻辑是为你这个 Ego-to-Exo 任务定制的**，通常无法直接调用 VBench 的一行代码来实现，需要你自己写胶水代码（就是你目前正在写的）。

| **指标名称**                     | **核心组件 (库)**   | **你的定制逻辑 (区别于通用指标)**                            |
| -------------------------------- | ------------------- | ------------------------------------------------------------ |
| **Appearance Consistency (ID)**  | DINOv2 / ArcFace    | **跨视角一致性**：通用指标通常比对同一视角的帧。你的逻辑是比对 Exo 生成人脸与 Reference 图片，验证在视角转换后 ID 是否丢失。 |
| **Subject Detection Rate (SDR)** | Faster R-CNN / YOLO | **生成稳定性**：通常用于检测视频是否崩坏。你将其作为衡量模型在大幅度视角变换下“保持人形”能力的指标。 |
| **Camera Centering Error**       | Detection Box       | **构图约束**：这是完全针对“第三人称跟随 (Third-person Follow)”任务设计的几何约束。通用视频生成不要求人物居中。 |

------

3. **核心创新指标 (Key Innovations)**：你的主要贡献

这是你回答 Reviewer "What is new here?" 的核心武器。这部分指标虽然数学原理不复杂，但**针对 Ego2Exo 任务的痛点进行了针对性设计**，属于 Metric Design 层面的创新。

**创新点 A: Background Semantic Consistency (BSC) —— 解决“动态背景”难题**

- **现有指标痛点**：传统的 Background Consistency 通常计算 **Pixel-wise MSE** (像素均方误差) 或 **LPIPS**。但这假设相机是静止的。在你的跟随任务中，背景在不断后退和变化，像素误差会极大，导致传统指标失效。
- **你的创新**：
  - 提出了 **Masked-CLIP** 方法。
  - **区别**：你不再比对像素，而是把人抠掉，只比对背景的**语义特征 (Semantic Feature)**。这允许背景发生位移（Translation）和透视变化，只要“森林还是森林”，分数就高。这是对动态运镜场景的专门优化。

**创新点 B: Human Action Alignment (HAA) —— 验证“隐式推断”能力**

- **现有指标痛点**：VBench 的 Action metric 通常是 **Text-to-Video** 的 (Prompt 说是跑步，生成的是不是跑步)。
- **你的创新**：
  - 你将其改为 **Cross-View Action Matching**。
  - **深层意义**：Ego 视频里看不到全身动作。你用这个指标证明了模型具备**推断 (Inference)** 能力 —— 能够通过 Ego 的相机晃动频率，反推出 Exo 视角的步伐和姿态。这是一个非常高级的 Evaluation。

**创新点 C: Trajectory Alignment vs. Optical Flow —— 提出“智能稳像”价值观**

- **现有指标痛点**：由 **Optical Flow Correlation (OFC)** 代表的传统指标要求生成的运动必须和 GT 严丝合缝。
- **你的创新**：
  - 你明确提出**放弃 OFC，采用 Trajectory Alignment**。
  - **区别**：你利用 **Perception-Distortion Tradeoff** 理论，主张模型不应学习手持拍摄的“抖动”，而应学习“轨迹”。
  - **价值**：你把 Trajectory Alignment (ADE) + Motion Smoothness 结合起来，定义了一种新的评估标准 —— **“Cinematic Follow Quality” (电影级跟随质量)**，即“轨迹要准，运镜要稳”。

------

### 4. 总结：如何在论文中陈述？

在你的 Experiment 或 Metric 章节，建议这样分类描述：

1. **Standard Quality Metrics (通用质量指标)**:

   > "Following standard practices in video generation benchmarks like VBench [Cite], we employ **FVD, Aesthetic Quality, and Imaging Quality** to evaluate the general perceptual quality of our generated videos."

2. **Task-Specific Consistency Metrics (任务特定一致性指标 - 创新点)**:

   > "To address the unique challenges of the Ego-to-Exo translation task, we introduce a set of specialized metrics:"
   >
   > - **Background Semantic Consistency (BSC)**: "Unlike traditional pixel-level metrics that fail under camera motion, BSC evaluates the semantic stability of the background environment using Masked-CLIP features."
   > - **Human Action Alignment (HAA)**: "Evaluates the model's ability to infer implicit body dynamics from egocentric camera motion."
   > - **Trajectory Alignment**: "Prioritizes the accuracy of the subject's global path over high-frequency camera jitters, favoring a smoother, cinematic follow-cam experience."

一句话总结：

你的大部分基础指标是现成的（这是好事，保证了权威性），但针对“动镜头背景”和“隐式动作推断”设计的 BSC 和 HAA，以及针对“稳像”选择的 Trajectory Alignment，是你区别于通用视频生成任务的核心创新。