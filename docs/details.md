## Details

### FVD

Frechet Video Distance FVD 的核心思想不是比较“像素”，而是比较“特征分布”。特征提取 (Feature Extraction):我们把所有视频（生成的和真实的）都扔进一个预训练好的 I3D (Inflated 3D ConvNet) 网络中。为什么是 I3D？因为它是在 Kinetics-400（大规模动作视频数据集）上训练的，它能同时理解空间内容（画面是什么）和时间动态（动作怎么动）。我们会取 I3D 网络倒数第二层的输出（通常是 2048 维或者 400 维的向量），这代表了视频的高层语义特征。高斯假设 (Gaussian Assumption):我们将所有“真实视频”的特征看作一团点云，假设这团点云符合多维高斯分布（正态分布）。同样，把“生成视频”的特征也看作另一团符合高斯分布的点云。弗雷歇距离 (Fréchet Distance):FVD 计算这两个高斯分布（两个“椭圆球”）之间的距离。$$FVD = ||\mu_r - \mu_g||^2 + \text{Tr}(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2})$$$\mu$: 特征的均值（中心位置）。$\Sigma$: 特征的协方差矩阵（形状和朝向）。直观理解：如果 FVD=0，说明生成视频的分布和真实视频完全重合（完美）。数值越小越好.

### IQ 

(Imaging Quality / MUSIQ)

### AQ 

Aesthetic Quality 的核心目标是将“美”这个主观概念量化。基本假设：由于深度学习模型（如 CLIP）已经学习了极其丰富的图像语义特征，我们只需要训练一个轻量级的回归网络 (Regressor)，将 CLIP 的特征向量映射到一个 1-10 分的人类评分标准上。公式逻辑：$$Score = MLP(Encoder(Image))$$Encoder: 通常使用 CLIP ViT-L/14。它看到的不仅是像素，还有构图、色彩、风格等高层信息。MLP: 一个简单的多层感知机（Linear -> ReLU -> Linear...），在 LAION-Aesthetics 或 AVA 数据集上训练过。这些数据集包含了成千上万张由人类打分（1分=丑陋，10分=极美）的图片。视频美学：对于视频，我们通常计算每一帧的美学得分，然后取平均值：$$VideoScore = \frac{1}{T} \sum_{t=1}^{T} Score(I_t)$$

### SSIM

### LPIPS

### TSD

Texture Style Distance (Gram Loss)

### SCS

Shadow Consistency Score 

### TF 

(Temporal Flickering)**Temporal Flickering**

数学原理 (The Math)Temporal Flickering (时序闪烁)核心思想：如果视频是连贯的，那么第 $t$ 帧的内容应该能通过第 $t-1$ 帧推导出来。如果推导不出来（比如像素突然跳变、噪点闪烁），那就是“Flickering”。计算逻辑：计算从 $t-1$ 到 $t$ 的光流 $F$。利用光流 $F$ 将第 $t-1$ 帧扭曲（Warp）成第 $t$ 帧的样子，得到 $I'_{t}$。计算真实第 $t$ 帧 $I_t$ 与预测帧 $I'_{t}$ 之间的误差（通常是 L2 距离）。$$E_{flicker} = || M \odot (I_t - \text{Warp}(I_{t-1}, F)) ||^2$$(注：$M$ 是遮挡掩码，为了简化，我们可以忽略遮挡或简单处理)

### MS

 (Motion Smoothness)**Motion Smoothness**

Motion Smoothness (运动平滑度)核心思想：物体的运动应该是平滑的，不应该有剧烈的加速度突变。计算逻辑：计算光流场在时间上的变化率（即加速度）。$$E_{smooth} = || F_t - F_{t-1} ||^2$$(光流本身是速度，光流的差分就是加速度)

### DD 

(Dynamic Degree)**Dynamic Degree**

数学原理 (The Math)Dynamic Degree 的设计初衷是为了防止模型“作弊”。作弊现象：有些生成模型为了提高画质（FVD/IS），倾向于生成几乎不动的“PPT视频”。这种视频画质好，但失去了视频的意义。计算逻辑：计算所有像素移动距离的平均值。$$DD = \frac{1}{T-1} \sum_{t=1}^{T-1} \left( \frac{1}{H \cdot W} \sum_{x,y} || \text{Flow}_t(x,y) ||_2 \right)$$Flow: 光流向量 $(u, v)$。Norm: 向量长度 $\sqrt{u^2 + v^2}$。直观理解：数值越大，视频里的东西动得越剧烈。

### OFC 

(Optical Flow Correlation)这是一个非常深刻且正确的见解。你指出了**生成任务（Generation）与重建任务（Reconstruction）**之间的核心区别。
为什么应该移除 Optical Flow Correlation (OFC)？
正如你所说，在你的任务中，Ground Truth (GT) 视频包含了“缺陷”（路面颠簸、手抖），而你希望生成模型能够超越 GT，产生像游戏视角一样平滑、稳定的运镜。
如果使用 OFC：
高分陷阱：模型必须完美复刻 GT 的每一次抖动和颠簸，才能拿到高分。
低分误杀：如果模型生成了完美的“斯坦尼康”式平滑跟随镜头，而 GT 在晃动，OFC 分数会很低（不相关甚至负相关）。

**Optical Flow Correlation**

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

### AC

(Appearance Consistency / Fidelity) 将 CLIP 用于 **人物外观（Subject Appearance）** 的一致性评估是业界的主流做法（如 DreamBooth, Textual Inversion 的评估）。这能衡量模型是否生成了符合描述的“概念”或“风格”。

我们不再 Mask 掉人物，而是 **Crop（裁剪）出人物**。
$$
Score_{AC\_CLIP} = \frac{1}{|\mathcal{T}|} \sum_{t \in \mathcal{T}} \text{CosSim}\left( E_{clip}(Crop(I_t, B_t)), E_{clip}(I_{ref}) \right)
$$

- $I_{ref}$: 参考图像（Ground Truth 的第一帧或指定的人物参考图）。
- $B_t$: 第 $t$ 帧检测到的人物 Bounding Box。
- $Crop(\cdot)$: 根据 Bbox 裁剪图像。
- $E_{clip}(\cdot)$: CLIP Image Encoder 提取的特征向量。

###  SF

Structural Fidelity 使用 DINOv2 捕捉局部纹理和结构，比 CLIP 更细粒度。**Appearance Consistency / Long-term ID Stability**

逻辑很简单：
利用现有的检测器（Detection Model）把画面中的“人”抠出来（Crop）。
把抠出来的人扔给 DINOv2 提取特征。
计算与参考图（Reference Image）的相似度。

$$Score_{ID} = \frac{1}{N_{samples}} \sum_{i \in Samples} \frac{E(C_i) \cdot E(I_{ref})}{||E(C_i)|| \cdot ||E(I_{ref})||}$$

- $C_i$: 第 $i$ 帧检测并 Crop 出来的人物图像。
- $I_{ref}$: 参考的人物图像。
- $E(\cdot)$: DINOv2 特征提取器。
- 公式即为这两个特征向量的余弦相似度（Cosine Similarity）。

### SDR

 (Subject Detection Rate)

（旧名 `vv.py`）完整性检查，防止生成崩坏导致检测不到人。

**Viewpoint Validity / Subject Detection Rate**

核心思想：衡量生成模型是否“崩坏”，即是否生成了无法被识别为“人”的扭曲图像。

计算逻辑：

设 $D(I_t)$ 为检测器（如 Faster R-CNN）在第 $t$ 帧的输出置信度分数。设定阈值 $\tau$（你的代码中为 0.7）。

$$SDR = \frac{1}{T} \sum_{t=1}^{T} \mathbb{1}( \max(D(I_t)) > \tau )$$

其中 $\mathbb{1}(\cdot)$ 是示性函数，满足条件为 1，否则为 0。

###  ASP 

Action Semantics Preservation (ASP) / Keystep Ego

### ASC

Action Semantic Consistency (ASC)

### TAA

Temporal Action Alignment ()

### HAA

 (Human Action Alignment)**Human Action Alignment**

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

### HOI

Hand-Object Interaction () Consistency





### MTD

Motion Trajectory Deviation ()

### STC

Spatio-Temporal Correspondence

### SSDC

Side-by-Side Depth Consistency  基于[Affine-Invariant Depth Error](https://arxiv.org/pdf/2002.00569#page=9&zoom=100,62,208)  在 Ego2Exo 生成任务中，我们希望衡量生成的视频 $V_{gen}$ 是否还原了真实场景 $V_{gt}$ 的 3D 几何结构（如房间的纵深、物体的前后遮挡关系）。然而，直接对比像素是无效的。因此，我们引入单目深度估计模型（如 Depth Anything V2 或 MiDaS）作为“代理评估者”。

记深度估计函数为 $\mathcal{F}$。我们得到两个深度图序列：
$$
D_{gen} = \mathcal{F}(V_{gen}), \quad D_{gt} = \mathcal{F}(V_{gt})
$$
大多数 SOTA 单目深度模型输出的是 相对深度 (Relative Depth)，甚至是逆深度 (Inverse Depth)。这意味着输出值 $d$ 与真实物理距离 $Z$ 之间存在一个未知的线性变换关系：
$$
D_{gen} \approx s \cdot D_{gt} + t
$$


其中 $s$ 是缩放因子 (Scale)，$t$ 是平移偏移 (Shift)。由于生成模型的焦距可能与 GT 不同，直接计算 $||D_{gen} - D_{gt}||$ 毫无意义。

最小二乘对齐 (Least Squares Alignment) **寻找最佳的 $s$ 和 $t$，使得对齐后的生成深度图与 GT 深度图差异最小，然后计算这个最小差异。**

对于视频中的每一帧 $i$，我们将深度图展平为向量 $\mathbf{d}_{gen} \in \mathbb{R}^N$ 和 $\mathbf{d}_{gt} \in \mathbb{R}^N$（$N = H \times W$）。

我们的优化目标是最小化均方误差 (MSE)：
$$
\min_{s, t} E(s, t) = \sum_{j=1}^{N} \left( (s \cdot \mathbf{d}_{gen}[j] + t) - \mathbf{d}_{gt}[j] \right)^2
$$
这是一个标准的**线性最小二乘问题**。根据偏导数为 0，我们可以得到 $s$ 和 $t$ 的闭式解 (Closed-form Solution)：

1. 计算缩放因子 $s$:
   $$
   s = \frac{\sum_{j=1}^{N} (\mathbf{d}_{gen}[j] - \bar{d}_{gen})(\mathbf{d}_{gt}[j] - \bar{d}_{gt})}{\sum_{j=1}^{N} (\mathbf{d}_{gen}[j] - \bar{d}_{gen})^2}
   $$
   即
   $$
   s = \frac{\text{Cov}(D_{gen}, D_{gt})}{\text{Var}(D_{gen})}
   $$
   
2. 计算偏移量 $t$:
   $$
   t = \bar{d}_{gt} - s \cdot \bar{d}_{gen}
   $$
   

   其中 $\bar{d}$ 表示深度图的均值。

最终指标计算 (Metric Formulation) 在获得最佳对齐参数 $s^*, t^*$ 后，我们将生成深度图进行校正 (Rectification)：
$$
\hat{D}_{gen} = s^* \cdot D_{gen} + t^*
$$
SSDC 定义为校正后的深度图与 GT 深度图的 **AbsRel (Absolute Relative Difference)** 或 **RMSE**。在您的 Benchmark 中，建议使用 **RMSE** 以惩罚大的结构性错误：
$$
SSDC = \frac{1}{T} \sum_{i=1}^{T} \sqrt{ \frac{1}{N} \sum_{j=1}^{N} || \hat{D}_{gen}^{(i)}[j] - D_{gt}^{(i)}[j] ||^2 }
$$
**数值含义**：SSDC 越低，说明生成视频的“3D 结构”与 GT 越一致。

**物理直觉**：即使生成视频变亮了、变暗了，或者焦距变了（导致整体变大变小），只要**“前面的人依然在前面，后面的墙依然在后面”**，这个指标就会很低（好）。

### CD

Chamfer Distance (SfM Consistency)

### ADE

Average Displacement Error (ADE) 计算生成轨迹与 GT 轨迹的误差。核心思想：人物在画面中的移动路径应该符合预期。

计算逻辑：

1. 记录每一帧人物检测框的中心点 $P_t = (cx_t, cy_t)$。

2. 形成生成轨迹 $\mathcal{T}_{gen} = \{P_0, ..., P_T\}$ 和参考轨迹 $\mathcal{T}_{ref}$。

3. 计算两条曲线的距离。最常用的是 平均位移误差 (ADE - Average Displacement Error)：

   $$ADE = \frac{1}{T} \sum_{t=1}^{T} || P_{gen,t} - P_{ref,t} ||_2$$

   如果存在时间轴不对齐的情况，则使用 动态时间规整 (DTW - Dynamic Time Warping) 距离。

### FDE

### TAE

Time Alignment Error & Cycle-Consistency

### CCE

 (Camera Centering Error)**Camera Centering Error**

Camera Centering Error (相机中心误差)目标：衡量生成的主角是否位于画面中心（符合第三人称跟随视角的构图习惯）。计算：计算主角框中心 $(cx, cy)$。计算画面中心 $(W/2, H/2)$。计算两点间的欧氏距离，并归一化（除以画面半对角线长度），使其范围在 $[0, 1]$。$$Error = \frac{\sqrt{(cx - W/2)^2 + (cy - H/2)^2}}{\sqrt{(W/2)^2 + (H/2)^2}}$$0.0: 完美居中。1.0: 主角在角落里。Viewpoint Validity (视角有效性)目标：衡量生成的视频中是否包含有效的人物目标。计算：$$Validity = \frac{\text{Count}(\text{Frames with Person Detected})}{\text{Total Sampled Frames}}$$这其实就是我们之前代码里的 detection_rate。

### CTE

 (Camera Trajectory Error) / Warp Error CTE 的本质就是 SLAM 领域标准的 **ATE** 指标。假设我们有两个相机位姿序列：

- **$T_{gt} = \{Q_1, Q_2, \dots, Q_N\}$**：Ground Truth 视频的真实相机轨迹。
- **$T_{gen} = \{P_1, P_2, \dots, P_N\}$**：从生成视频中估算出的相机轨迹。

其中每个位姿 $P_i \in SE(3)$ 通常由一个 $4 \times 4$ 变换矩阵表示：
$$
P_i = \begin{bmatrix} R_i & t_i \\ 0 & 1 \end{bmatrix}
$$
$R_i$ 是旋转矩阵，$t_i$ 是平移向量。

实现挑战：坐标系不统一。对于单目视频生成（Monocular Video Generation），存在两个巨大的数学鸿沟：

1. 尺度模糊 (Scale Ambiguity)：生成视频重建出来的轨迹，其单位是未知的（比如“1单位”可能代表现实中的 1米，也可能代表 1厘米）。
2. 参考系漂移 (Reference Frame Arbitrariness)：生成视频的世界坐标系原点 $(0,0,0)$ 是由第一帧决定的，与 GT 的原点完全不同。

因此，直接计算 $||Q_i - P_i||$ 是没有意义的。我们必须先进行 Sim(3) 对齐。

关键算法：Umeyama 对齐 (Sim3 Alignment)我们需要找到一个相似变换 $S \in Sim(3)$包含：

- 缩放因子 $s \in \mathbb{R}$
- 旋转矩阵 $R \in SO(3)$
- 平移向量 $t \in \mathbb{R}^3$

使得变换后的生成轨迹与 GT 轨迹的**最小二乘误差最小**。**优化目标函数**：
$$
\min_{s, R, t} \sum_{i=1}^{N} || \mathbf{p}_{gt, i} - (s R \mathbf{p}_{gen, i} + t) ||^2
$$
其中 $\mathbf{p}_{gt, i}$ 和 $\mathbf{p}_{gen, i}$ 分别是第 $i$ 帧相机的 **平移部分 (Translation Vector)**。

求解方法 (Umeyama Algorithm)：这不仅是旋转平移，还包含尺度。其闭式解（Closed-form Solution）步骤如下：

1. 去中心化 (Centering)：计算两组轨迹的质心 $\mu_{gt}, \mu_{gen}$，并构建去中心化坐标：
   $$
   \mathbf{q}_i = \mathbf{p}_{gt, i} - \mu_{gt}, \quad \mathbf{y}_i = \mathbf{p}_{gen, i} - \mu_{gen}
   $$

2. 计算协方差矩阵 (Covariance Matrix)：
   $$
   H = \sum_{i=1}^{N} \mathbf{y}_i \mathbf{q}_i^T
   $$

3. SVD 分解求解旋转 $R$：对 $H$ 进行奇异值分解 $H = U \Sigma V^T$。
   $$
   R = V \begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & \det(V U^T) \end{bmatrix} U^T
   $$
   (注：中间的对角阵是为了确保 $R$ 是旋转矩阵而不是反射矩阵)

4. 求解缩放 $s$ 和平移 $t$：
   $$
   s = \frac{\sum \mathbf{y}_i^T R^T \mathbf{q}_i}{\sum ||\mathbf{y}_i||^2}\\
   t = \mu_{gt} - s R \mu_{gen}
   $$

最终 CTE 计算公式 在求得最佳变换参数 $s, R, t$ 后，我们将生成的轨迹变换到 GT 坐标系下：
$$
\hat{\mathbf{p}}_{gen, i} = s R \mathbf{p}_{gen, i} + t
$$
CTE (即 RMSE of ATE) 计算如下：
$$
CTE = \sqrt{ \frac{1}{N} \sum_{i=1}^{N} || \mathbf{p}_{gt, i} - \hat{\mathbf{p}}_{gen, i} ||_2^2 }
$$
对于 **单目视频生成 (Monocular Video Generation)** 任务，这是一个非常经典的问题。

在没有真实相机内参 ($K$) 的情况下，想要恢复**绝对的公制轨迹**是不可能的（这是一个不适定问题）。但是，对于 **Benchmark 评测** 而言，我们不需要“绝对正确”的轨迹，我们只需要 **“相对一致”** 的评估。

解决方案：基于“规范相机假设”的对称评估 (Symmetric Canonical Evaluation)

既然生成视频没有内参，我们采用 **"Canonical Camera (规范相机)"** 策略。

1. **假设一个通用的内参**：对于所有视频（无论是生成的还是 GT），我们都强制假设它们是由一个 **FOV = 60° (或 90°)** 的标准相机拍摄的。
2. **对称评估 (Symmetry)**：
   - **不要** 直接拿生成的轨迹去和数据集里提供的真实相机 pose 对比（这不公平，因为算法存在系统误差）。
   - **而是** 对 **生成的视频 ($V_{gen}$)** 和 **真值视频 ($V_{gt}$)** 使用 **同一个** 设置了规范内参的 DPVO 模型进行推理。
3. **Sim3 消除误差**：由于我们假设的内参可能是错的（比如真实 FOV 是 30°，我们假设了 60°），这会导致估算出的轨迹在 Z 轴上被“压缩”或“拉伸”。**但是**，`metric.py` 中实现的 **Sim3 Alignment (Umeyama)** 算法包含一个 **缩放因子 $s$**。这个 $s$ 会自动修正这种由内参假设错误导致的全局尺度差异。

### SCDE

Subject-Camera Distance Error 核心数学模型：透视投影与深度解耦

首先，我们需要理解为什么不能直接用检测框大小来算距离。

在 3D 空间中，相机坐标系下的 3D 人体根节点（Root Joint / Pelvis）坐标为 $P = [X, Y, Z]^T$。

图像上的 2D 投影点 $p = [u, v]^T$ 满足透视投影公式：
$$
u = f_x \frac{X}{Z} + c_x, \quad v = f_y \frac{Y}{Z} + c_y
$$
其中 $Z$ 就是我们关注的 **Subject-Camera Distance**（深度）。

要获取这个 $Z$，我们不能仅凭 2D 图像（因为 $f_x$ 和 $X$ 未知），必须借助 **3D Human Mesh Recovery (HMR)** 模型（如 WHAM, 4DHumans, SMPL-X）。这些模型会输出弱透视投影或全透视投影下的相机参数。

步骤一：提取深度序列 (Depth Sequence Extraction)

对生成视频 $V_{gen}$ 和真实视频 $V_{gt}$ 的每一帧 $t$，运行 HMR 模型（建议使用 **WHAM** 或 **4DHumans**，因为它们对时序更鲁棒）。

模型输出的相机平移向量通常表示为 $T_{cam} = [t_x, t_y, t_z]$。

我们需要提取其中的 $Z$ 分量：

- **生成序列**: $D_{gen} = \{ z_{gen}^{(1)}, z_{gen}^{(2)}, \dots, z_{gen}^{(N)} \}$
- **GT 序列**: $D_{gt} = \{ z_{gt}^{(1)}, z_{gt}^{(2)}, \dots, z_{gt}^{(N)} \}$

> **注意**：如果模型输出的是弱透视参数 $(s, t_x, t_y)$，则深度 $z \propto 1/s$。

步骤二：尺度对齐 (Scale Alignment)

由于生成模型（尤其是 Diffusion）生成的视频可能存在**“尺度模糊” (Scale Ambiguity)**——即生成的人物看起来像是在 2 米处，但实际上可能是因为生成了一个“大比例”的人在 4 米处（焦距不同导致的错觉）。

如果不进行对齐，计算出的误差会极大且无意义。我们假设两者之间存在一个线性比例关系：$z_{gt} \approx s \cdot z_{gen}$。使用 最小二乘法 (Least Squares) 计算最佳缩放因子 $\hat{s}$：
$$
\hat{s} = \frac{\sum_{t=1}^{N} z_{gen}^{(t)} \cdot z_{gt}^{(t)}}{\sum_{t=1}^{N} (z_{gen}^{(t)})^2}
$$
然后校正生成序列：*注：通常只需对齐 Scale ($s$)，不需要对齐 Shift ($t$)，因为相机坐标系原点通常都在相机光心。*
$$
\hat{z}_{gen}^{(t)} = \hat{s} \cdot z_{gen}^{(t)}
$$
步骤三：计算误差 (Error Calculation)

现在我们有了对齐后的深度序列，可以计算指标了。

A. 绝对距离误差 (Accuracy Metric)

衡量生成视频的跟随距离是否准确（比如 GT 要求保持 2 米，你是不是也保持在 2 米附近）：
$$
SCDE_{acc} = \frac{1}{N} \sum_{t=1}^{N} | \hat{z}_{gen}^{(t)} - z_{gt}^{(t)} |
$$
B. 稳定性/呼吸效应度量 (Stability Metric)

这是您关心的“忽远忽近”。如果 GT 是平滑的（比如匀速跟随），而 Gen 有呼吸效应，那么 $\hat{z}_{gen}$ 会围绕 $z_{gt}$ 震荡。

我们可以计算深度变化率的误差 (Depth Velocity Error)：
$$
v_{gen}^{(t)} = \hat{z}_{gen}^{(t)} - \hat{z}_{gen}^{(t-1)}, \quad v_{gt}^{(t)} = z_{gt}^{(t)} - z_{gt}^{(t-1)}
$$

$$
SCDE_{stab} = \frac{1}{N-1} \sum_{t=2}^{N} | v_{gen}^{(t)} - v_{gt}^{(t)} |
$$

如果 $SCDE_{stab}$ 很大，说明生成视频的相机在 Z 轴上剧烈抖动（呼吸效应）。

为了简化 Benchmark，通常将两者结合或只取绝对误差（因为剧烈抖动必然导致绝对误差增大）。

建议的 **Subject-Camera Distance Error (SCDE)** 公式：
$$
\text{SCDE} = \underbrace{\sqrt{\frac{1}{N} \sum_{t=1}^{N} (\hat{s} z_{gen}^{(t)} - z_{gt}^{(t)})^2}}_{\text{Scale-Aligned RMSE}}
$$


### FDV

3D Follow Distance Variance (FDV)

### CSHA

Camera-Subject Heading Alignment ()1. 核心数学模型：相对方位角 (Relative Azimuth)

CSHA 的本质是计算 **“相机相对于主角的方位角” (Relative Azimuth Angle)** 的稳定性。

我们需要在每一帧 $t$ 定义两个 2D 向量（投影到地平面）：

1. **主角朝向向量 (Subject Heading Vector)** $\vec{v}_{subj}^{(t)}$
2. **相机方位向量 (Camera Position Vector)** $\vec{v}_{cam\_pos}^{(t)}$（即从主角指向相机的向量）

理想情况：在跟随视角中，这两个向量的夹角 $\Delta \theta$ 应该保持恒定（例如始终差 180°，即在正后方）。

CSHA 指标：衡量 $\Delta \theta_t$ 在时间序列上的方差 (Variance)。

**步骤一：数据准备 (Data Preparation)**

我们需要每一帧 $t$ 的 3D 数据（通常由 WHAM/SMPL 和 SLAM 获得）：

- **主角位置 (Root Position)**: $P_{subj}^{(t)} = (x_s, y_s, z_s)$
- **主角姿态 (Root Orientation)**: 旋转矩阵 $R_{subj}^{(t)} \in SO(3)$
- **相机位置 (Camera Position)**: $P_{cam}^{(t)} = (x_c, y_c, z_c)$

**步骤二：提取主角朝向 (Subject Forward Vector)**

首先，我们需要知道“主角脸朝哪儿”。

在 SMPL 模型中，假设标准 T-pose 是朝向 $+Z$ 轴（或 $+Y$，视坐标系定义而定，这里假设 $+Z$ 为前，$+Y$ 为上）。

通过旋转矩阵将“前向基向量”变换到世界坐标系：
$$
\vec{V}_{forward}^{(t)} = R_{subj}^{(t)} \cdot \begin{bmatrix} 0 \\ 0 \\ 1 \end{bmatrix}
$$
然后将其投影到水平地面（假设 $XZ$ 平面为地面，忽略高度 $Y$）：
$$
\vec{v}_{subj}^{(t)} = [V_{forward, x}^{(t)}, \quad V_{forward, z}^{(t)}]^T
$$
并进行归一化。

**步骤三：提取相机方位向量 (Camera Azimuth Vector)**

计算相机相对于主角的位置向量：
$$
\vec{D}^{(t)} = P_{cam}^{(t)} - P_{subj}^{(t)}
$$
同样投影到水平地面：
$$
\vec{v}_{cam\_pos}^{(t)} = [D_x^{(t)}, \quad D_z^{(t)}]^T
$$
并进行归一化。这个向量代表了“相机站在主角的哪个方向”。

**步骤四：计算相对夹角 (Relative Azimuth Angle)**

现在我们计算这两个向量在 2D 平面上的夹角 $\phi^{(t)}$。

为了处理周期性（-180° 到 180°），推荐使用 atan2：
$$
\theta_{subj}^{(t)} = \text{atan2}(v_{subj, z}^{(t)}, v_{subj, x}^{(t)})\\
\theta_{cam}^{(t)} = \text{atan2}(v_{cam\_pos, z}^{(t)}, v_{cam\_pos, x}^{(t)})
$$
**相对方位角 (Relative Azimuth)**:
$$
\phi^{(t)} = \text{wrap\_to\_pi}(\theta_{cam}^{(t)} - \theta_{subj}^{(t)})
$$
其中 `wrap_to_pi` 函数将角度映射回 $[-\pi, \pi]$ 区间。

**物理意义**：

- $\phi \approx \pi$ (180°) $\rightarrow$ 相机在主角正后方（标准第三人称）。
- $\phi \approx \pi/2$ (90°) $\rightarrow$ 相机在主角侧面。

步骤五：计算 CSHA 得分 (Metric Formulation)

A. 稳定性误差 (Stability Error)

如果是一个好的 Follow Camera，无论主角怎么转，$\phi^{(t)}$ 应该是常数。因此我们计算它的圆周标准差 (Circular Standard Deviation)：
$$
CSHA_{error} = \sqrt{ -2 \ln | \bar{R} | }
$$
其中 $\bar{R} = \frac{1}{T} \sum_{t=1}^{T} e^{i \phi^{(t)}}$ 是平均结果向量的长度。

(或者简单地计算 $\phi^{(t)}$ 的标准差，如果变化不剧烈的话)。

B. 锁定误差 (Locking Error) [可选]

如果你强制要求相机必须在正后方（像赛车游戏），你可以计算与目标角度 $\pi$ 的偏差：
$$
CSHA_{lock} = \frac{1}{T} \sum_{t=1}^{T} | \text{wrap\_to\_pi}(\phi^{(t)} - \pi) |
$$

### HS

Horizon Stability

### TS

**Trajectory Smoothness** (或者更精确地称为 **Camera Stabilization Score**) 的数学原理核心在于：**利用“全局运动的加速度”来量化运镜的顿挫感。**

这与物理学中衡量物体运动平稳性的逻辑一致：我们不惩罚“快”（速度），只惩罚“抖”（加速度）。以下是该指标的完整数学定义和计算步骤：

1. 核心数学模型

我们将视频看作是一个相机在 2D 平面上的运动轨迹。

- **输入**：视频帧序列 $I = \{I_1, I_2, ..., I_T\}$。
- **中间变量**：每一帧的全局位移速度向量 $\vec{v}_t = (dx_t, dy_t)$。
- **核心变量**：每一帧的全局加速度向量 $\vec{a}_t = (ax_t, ay_t)$。
- **输出**：标量分数 $S$，表示平均加速度的大小（越小越稳）。

2. 计算步骤详解

**步骤一：估计全局相机速度 (Global Camera Velocity)**

首先，我们需要剥离掉画面中人物的局部动作，只提取背景（相机）的运动。

最稳健且简单的方法是计算平均光流 (Average Optical Flow)。

设 $F_{t \to t+1}(x, y)$ 为第 $t$ 帧中像素 $(x, y)$ 到第 $t+1$ 帧的光流向量。

相机的平移速度 $\vec{v}_t$ 为全图光流的均值：
$$
\vec{v}_t = \frac{1}{H \times W} \sum_{x=1}^{W} \sum_{y=1}^{H} F_{t \to t+1}(x, y)
$$


- **代码对应**：这一步对应您 `utils/video_kit.py` 中的 `get_motion_series` 函数。
- **物理意义**：代表相机在这一帧“移动了多少像素”。

**步骤二：计算全局相机加速度 (Global Camera Acceleration)**

稳定性（Stabilization）的定义是“速度保持恒定”。如果速度忽快忽慢，或者方向忽左忽右，就是“抖动”。

因此，我们对速度求导（差分）得到加速度 $\vec{a}_t$：
$$
\vec{a}_t = \vec{v}_t - \vec{v}_{t-1}
$$


**数学细节**：这是一个二阶差分过程（Second-order Difference of Position）。

位置 $P$ $\xrightarrow{\text{diff}}$ 速度 $V$ $\xrightarrow{\text{diff}}$ 加速度 $A$。

**步骤三：计算平滑度得分 (Smoothness Metric)**

我们计算加速度向量的模长（Magnitude），并取时间序列的平均值：
$$
Score_{smooth} = \frac{1}{T-2} \sum_{t=2}^{T-1} \| \vec{a}_t \|_2
$$
展开形式为：
$$
Score_{smooth} = \frac{1}{T-2} \sum_{t=2}^{T-1} \sqrt{ (dx_t - dx_{t-1})^2 + (dy_t - dy_{t-1})^2 }
$$



### SCCR

在 Ego2Exo 任务中，Ego 视频不仅是“输入”，更是“控制条件（Control Condition）”。如果生成视频无法在海量干扰项中通过检索找回源头，说明模型“脱管”了（Lost Control），忽略了输入条件，产生了幻觉。以下为您详解 **Source Control Condition Recall (SCCR)** 的数学原理。

核心数学模型：条件概率的后验验证

SCCR 的本质是验证模型是否最大化了 **$P(Exo_{gen} | Ego_{src})$** 这一条件概率。由于我们无法直接计算高维视频的概率分布，我们采用了**基于实例的代理验证（Instance-based Proxy Verification）**：

- **假设**：对于每一个 $Ego_{src}^i$，存在唯一且确定的 Ground Truth $Exo_{gt}^i$。
- **推论**：如果生成视频 $Exo_{gen}^i$ 真正受控于 $Ego_{src}^i$，那么在特征空间中，$Exo_{gen}^i$ 与 $Exo_{gt}^i$ 的距离，应当显著小于它与任何其他 $Exo_{gt}^j (j \neq i)$ 的距离。

数学定义与计算步骤

**步骤一：特征空间的映射 (Feature Embedding)**

我们需要一个对**外观（Appearance）**和**全局语义（Global Semantics）**和对**时序**高度敏感的映射函数 $\Phi(\cdot)$（ **VideoMAE**）

设测试集共有 $N$ 个样本对。

- **Query 集合 (生成视频)**: $Q = \{q_1, q_2, \dots, q_N\}$，其中 $q_i = \Phi(Exo_{gen}^i)$。
- **Gallery 集合 (真值视频)**: $G = \{g_1, g_2, \dots, g_N\}$，其中 $g_j = \Phi(Exo_{gt}^j)$。

*注意：这里的 Gallery 包含了测试集中所有的 Ground Truth，它们互为干扰项（Distractors）。*

**步骤二：控制强度矩阵 (Control Strength Matrix)**

计算生成分布与真实分布之间的亲和度矩阵 $S \in \mathbb{R}^{N \times N}$：
$$
S_{i, j} = \text{CosSim}(q_i, g_j) = \frac{q_i \cdot g_j}{\|q_i\| \|g_j\|}
$$

- **$S_{i, i}$ (对角线)**：代表**“正向控制强度”**。即模型生成的第 $i$ 个视频与第 $i$ 个源条件的匹配度。
- **$S_{i, j}$ (非对角线)**：代表**“混淆干扰强度”**。即模型生成的第 $i$ 个视频看起来像第 $j$ 个源条件的程度。

**步骤三：排他性排序 (Exclusivity Ranking)**

对于每一个生成视频 $q_i$，我们对其在 Gallery 中的匹配分数行向量 $S_{i, :}$ 进行降序排列，得到排序索引 $Rank_i$。
$$
Rank_i = \text{argsort}_{desc}(S_{i, :}) \text{ 中真实索引 } i \text{ 的位置}
$$

- 如果 $Rank_i = 1$，说明 $S_{i, i} > S_{i, j}, \forall j \neq i$。
- **物理意义**：没有任何一个干扰视频（Distractor）比真正的源视频更像生成结果。这证明了**Source Control（源控制）是唯一且排他的**。

步骤四：SCCR 指标计算

**SCCR@K** 定义为能够在前 $K$ 个候选中找回源条件的概率：
$$
SCCR@K = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(Rank_i \le K)
$$
这个名字的数学美感在于它解释了**“为什么要检索？”**

- **Condition (条件)**：输入 $Ego^i$ 是一个约束条件（比如：红衣服、高个子、在厨房）。
- **Control (控制)**：生成模型必须严格遵守这些约束。
- **Recall (召回)**：
  - 如果模型**遵守**了控制条件 $\rightarrow$ $Exo_{gen}^i$ 就会带有“红衣服、高个子、厨房”特征 $\rightarrow$ 它会在特征空间里和 $Exo_{gt}^i$ 靠得最近 $\rightarrow$ **检索成功**。
  - 如果模型**失控**（忽略条件，生成了蓝衣服） $\rightarrow$ 它会和 $Exo_{gt}^i$ 距离变远，反而可能和 $Exo_{gt}^k$（蓝衣服的干扰项）更近 $\rightarrow$ **检索失败*

总结：SCCR 的数学本质

**SCCR** 是衡量生成模型 **条件互信息 (Conditional Mutual Information)** 的硬指标。
$$
I(Exo_{gen}; Ego_{src}) \approx SCCR \uparrow
$$
它回答了：“生成结果中的信息量，有多少是直接受控于源输入的？有多少是模型瞎编的？”如果 SCCR 低，说明模型在“瞎编”（幻觉）；如果 SCCR 高，说明模型是在“精准翻译”。

### TAA

Temporal Attention Alignment 

输入 (Ego)：不仅仅是视觉参考，它是“时序控制信号 (Temporal Control Signal)”。当 Ego 视角的双手在 $t=3s$ 拿起杯子时，Exo 视角的人物必须在 $t=3s$ 精确地做出拿起杯子的全身动作。

挑战：生成模型很容易产生“幻觉”，生成一个流畅但时间错乱的视频（例如：动作做慢了，或者动作顺序颠倒了）。

TAA 的角色：TAA 是一个“时序注意力审计员（Temporal Attention Auditor）”。它不关心视频里的像素画得好不好（那是 FID/FVD 的事），它只关心：生成视频的时间轴，是否严格遵循了 Ground Truth (即 Ego 映射后的真值) 的指挥？

我们利用 Transformer 的 Attention 机制来量化这种对齐，但剥离了内容传输（Value），仅保留寻址结构（Attention Map）

假设我们使用一个对时序敏感的编码器 $\Phi$（如 VideoMAE V2，它能区分“举手”和“放下”），将视频切分为 $T$ 个时间步的特征序列。

- **Query (生成视频流)**: $Q \in \mathbb{R}^{T \times D}$，其中 $Q_t = \Phi(Gen_t)$。
- **Key (真值视频流)**: $K \in \mathbb{R}^{T \times D}$，其中 $K_t = \Phi(GT_t)$。

计算生成视频每一帧对真值视频每一帧的“关注度”：
$$
A = \text{Softmax}_{\text{row}}\left( \frac{Q K^T}{\tau} \right) \in \mathbb{R}^{T \times T}
$$

- **$Q K^T$**: 原始相似度分数 (Logits)。
- **$\tau$**: 温度系数（Temperature），用于调节注意力的锐度。
- **$\text{Softmax}_{\text{row}}$**: 确保每一行（生成的每一帧）的注意力概率之和为 1。
- **物理意义**：矩阵元素 $A_{i,j}$ 代表 **“生成视频的第 $i$ 秒，有多大概率是对真值第 $j$ 秒的复现？”**

理想的受控生成，应该是一一对应的（$t \to t$）。这意味着注意力矩阵 $A$ 应该是一个**对角矩阵**（或接近单位矩阵）。

TAA 指标公式：

我们计算注意力矩阵主对角线上的迹（Trace）的均值：
$$
\text{TAA} = \frac{1}{T} \text{tr}(A) = \frac{1}{T} \sum_{t=1}^T A_{t,t}
$$
**解耦“寻址”与“表征”**： 传统的 Attention 是 $Output = A \cdot V$。我们扔掉了 $V$（Value）。 这暗示了我们是在**审计生成器的“指针”**。我们不在乎它搬运了什么特征（Value），我们只在乎它的指针是否**准确地指向了正确的时间戳**。

**高 TAA**：热力图显示为一条明亮、锐利的对角线。这代表**“单调对齐 (Monotonic Alignment)”**，即模型不仅学到了动作，还学到了因果律。

**低 TAA**：

- *偏移直线*：表示模型有系统性的**延迟 (Lag)**。
- *垂直条纹*：表示**模式坍塌 (Mode Collapse)**（某一帧动作重复出现）。
- *弥散云团*：表示**时序幻觉 (Hallucination)**。

如果您的测试集包含以下类型的视频，TAA 的参考价值会大打折扣：

1. **低动态视频 (Low Dynamics)**：睡觉、看书、静坐、冥想。
   - *特征*：时间熵（Temporal Entropy）极低。
2. **高频重复视频 (High Repetition)**：匀速跑步、跳绳、机械性重复劳动。
   - *特征*：自相关性（Auto-correlation）极高。
3. **纯背景运镜视频 (Ego-Motion Dominant)**：
   - 如果 Ego 视角在剧烈晃动，但 Exo 视角看着这一幕，Exo 画面里的人可能只是稍微动了一下。Ego 和 Exo 的**动作幅度不匹配**，导致特征提取器难以建立映射。

## Draft

**Background Semantic Consistency**

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

### from egoexo4d and 

[EgoX evaluation metrics](https://arxiv.org/abs/2512.08269)

**Image Criteria.** 

We measured PSNR, SSIM, LPIPS, and CLIP-I to assess how closely each generated frame matches the ground-truth distribution. 

**Object Criteria.**

 Following the object-level evaluation protocol of Ego-Exo4D [13], we assessed object-level consistency between the generated egocentric video and the ground truth. We used SAM2 [35] to segment and track objects and DINOv3 [37] to establish correspondences. For each matched object, we evaluated centerlocation error, Intersection-Over-Union(IoU), and Contour Accuracy to measure spatial alignment and boundary fidelity.

**Video Criteria.** 

We measured FVD [11] to evaluate how closely the generated video aligns with the groundtruth distribution. In addition, we assessed VBench [51]- Temporal Flickering, Motion Smoothness, and Dynamic Degree to quantify temporal stability and motion quality.