## 指标分类

🟢 [通用] (General)：视频生成领域的标准指标，用于证明模型的基础能力。

🌟 [特化] (Task-Specific)：专门为 Ego2Exo / Follow Camera 任务设计，体现你 Benchmark 的独特贡献。

| **维度 (Dimension)**        | **子维度 (Sub-Category)** | **指标名称 (Metric Name)**                      | **类型 (Type)** |
   | --------------------------- | ------------------------- | ----------------------------------------------- | --------------- |
   | **I. 视觉质量与感知保真度** | 基础画质与分布            | 🟢 FVD (Fréchet Video Distance)                  | 通用            |
   |                             |                           | 🟢 IQ (Imaging Quality / MUSIQ)                  | 通用            |
   |                             |                           | 🟢 AQ (Aesthetic Quality)                        | 通用            |
   |                             |                           | 🟢 SSIM / LPIPS                                  | 通用            |
   |                             | 真实感与环境细节          | 🟢 Texture Style Distance (Gram Loss)            | 通用            |
   |                             |                           | 🌟 Global Illumination Match (GIM)               | **特化**        |
   |                             |                           | 🌟 Shadow Consistency Score                      | **特化**        |
   | **II. 时序动态**            | 流畅度与稳定性            | 🟢 TF (Temporal Flickering)                      | 通用            |
   |                             |                           | 🟢 MS (Motion Smoothness)                        | 通用            |
   |                             | 运动模式                  | 🟢 DD (Dynamic Degree)                           | 通用            |
   |                             |                           | 🟢 OFC (Optical Flow Correlation)                | 通用            |
   | **III. 语义与动作**         | 身份与外观                | 🟢 AC (Appearance Consistency / Fidelity)        | 通用            |
   |                             |                           | 🟢 Structural Fidelity (DINOv2)                  | 通用            |
   |                             |                           | 🟢 SDR (Subject Detection Rate)                  | 通用            |
   |                             | 动作语义与逻辑            | 🌟 Action Semantics Preservation (ASP) / Keystep | **特化**        |
   |                             |                           | 🌟 Action Semantic Consistency (ASC)             | **特化**        |
   |                             |                           | 🌟 Temporal Action Alignment (TAA)               | **特化**        |
   |                             | 姿态与交互                | 🌟 HAA (Human Action Alignment)                  | **特化**        |
   |                             |                           | 🌟 Hand-Object Interaction (HOI) Consistency     | **特化**        |
   |                             | 跨视角关联                | 🌟 Semantic Re-Identification (ReID) / SMS       | **特化**        |
   |                             |                           | 🌟 Motion Trajectory Deviation (MTD)             | **特化**        |
   |                             |                           | 🌟 Spatio-Temporal Correspondence                | **特化**        |
   | **IV. 几何与透视**          | 深度与结构                | 🌟 Side-by-Side Depth Consistency (SSDC)         | **特化**        |
   |                             |                           | 🌟 Chamfer Distance (SfM Consistency)            | **特化**        |
   |                             | 时空对齐                  | 🌟 Geometric Drift (ADE) & FDE                   | **特化**        |
   |                             |                           | 🌟 Time Alignment Error & Cycle-Consistency      | **特化**        |
   | **V. 相机运镜控制**         | 构图控制                  | 🌟 CCE (Camera Centering Error)                  | **特化**        |
   |                             |                           | 🌟 CTE (Camera Trajectory Error) / Warp Error    | **特化**        |
   |                             | 跟随逻辑                  | 🌟 Subject-Camera Distance Error                 | **特化**        |
   |                             |                           | 🌟 3D Follow Distance Variance (FDV)             | **特化**        |
   |                             |                           | 🌟 Camera-Subject Heading Alignment (CSHA)       | **特化**        |
   |                             | 运镜稳定性                | 🌟 Horizon Stability (水平线稳定性)              | **特化**        |
   |                             |                           | 🌟 Trajectory Smoothness                         | **特化**        |

### 视觉质量

衡量生成视频看起来是否真实、清晰、美观，以及是否符合物理世界的材质规律。

1. 基础画质与分布: 

   🟢 FVD: 视频生成领域必需的标准。衡量生成视频分布与真实视频分布的整体距离（真实感+连贯性）

   🟢IQ (Imaging Quality / MUSIQ)： 基于多尺度 Transformer 的无参考画质评估（清晰度、噪点）

   🟢AQ (Aesthetic Quality)： 基于 CLIP+MLP 的美学评分（构图、色彩）。

   🟢 SSIM / LPIPS 经典的结构相似性与深度感知相似度指标。

2. 真实感和环境细节

   🟢 Texture Style Distance (Gram Loss) 惩罚“过度平滑”或“油画感”，强制生成高频纹理。

   🌟 Global Illumination Match (GIM) 确保环境氛围（黑夜/晴天）与 Ego 输入一致，基于 LAB 空间亮度直方图。

   🌟 Shadow Consistency Score 接地性检查，解决“人物漂浮感”，强制生成合理的阴影。

### 时序动态

衡量视频在时间轴上是否流畅，运动是否自然。

1. 流畅度与稳定性 (Smoothness & Stability)

   🟢 TF (Temporal Flickering) 检查画面闪烁和噪点跳变。

   🟢 MS (Motion Smoothness) 惩罚剧烈的加速度突变。

2. 运动模式 (Motion Patterns)

   🟢 DD (Dynamic Degree) 防作弊指标，防止生成静止的“PPT视频”。

   🟢 OFC (Optical Flow Correlation) 衡量运动模式的相关性。（注：在 Follow 任务中需辩证看待，因为我们可能需要比 GT 更稳的运动）。

### 语义和动作

1. 身份与外观 (Identity & Appearance)

   🟢 AC (Appearance Consistency / Fidelity)使用 CLIP Embedding 确保整体外观不漂移。

   🟢 Structural Fidelity (DINOv2) 使用 DINOv2 捕捉局部纹理和结构，比 CLIP 更细粒度。

   🟢 SDR (Subject Detection Rate) 完整性检查，防止生成崩坏导致检测不到人。

2. 动作语义与逻辑 (Action Semantics & Logic)

   🌟 Action Semantics Preservation (ASP) / Keystep Ego 做“步骤A”，Exo 也必须检测出“步骤A”。

   🌟 Action Semantic Consistency (ASC) 防止“语义漂移”（如切菜变洗碗）。

   🌟 Temporal Action Alignment (TAA) 确保动作顺序正确（A->B->C），不仅仅是短时平滑。

3. 姿态与交互 (Pose & Interaction)

   🌟 HAA (Human Action Alignment) 基于骨骼关键点的姿态向量相似度。

   🌟 Hand-Object Interaction (HOI) Consistency 手物接触状态的一致性，防止“隔空取物”。

4. 跨视角关联 (Cross-View Mapping)

   🌟 Source Control Condition Recall (SCCR) 它是否包含了足够的独特特征，使其能够从成千上万个干扰视频（Distractors）中，精准地指回它唯一的那个 Ego 来源？难点在于外观泛化 (Zero-shot Appearance)

   🌟 Temporal Synchronization Rate (TSR) 强调Frame-level Alignment (帧级对齐)。生成视频是否“严丝合缝”？已知它是这个人生成的，但它生成的动作是第 1 分钟的“挥手”还是第 2 分钟的“转身”？ 难点在于相似动作区分 (Fine-grained Action)

### 几何与透视

衡量生成模型是否构建了一个符合流形几何的 3D 世界。

1. 深度与结构 (Depth & Structure)

   🌟 Side-by-Side Depth Consistency (SSDC) 基于单目深度估计（Depth Anything V2）的场景几何一致性。

   🌟 Chamfer Distance (SfM Consistency) 硬核指标：通过 COLMAP 重建点云，对比“世界结构”是否歪曲。

2. 时空对齐 (Spatio-Temporal Alignment)

   🌟 Geometric Drift (ADE) & FDE 衡量轨迹的平均误差和终点累积误差。

   🌟 Time Alignment Error & Cycle-Consistency 毫秒级同步检查，确保物理接触瞬间（如击打）的时间戳对齐。

### 相机运镜控制

衡量“虚拟摄影师”的跟随水平，区别于普通视频生成。

1. 构图控制 (Composition)

   🌟 CCE (Camera Centering Error) 2D 构图指标：主角是否保持在视觉中心。

   🌟 CTE (Camera Trajectory Error) / Warp Error 3D 轨迹指标：利用 SLAM 估计相机轨迹，看是否符合 Follow 曲线。

2. 跟随逻辑 (Follow Logic)

   🌟 Subject-Camera Distance Error 基于 3D HPE (WHAM/4DHumans) 测量 $t_z$。核心指标。

   🌟 3D Follow Distance Variance (FDV) 呼吸效应：衡量跟随距离的方差，惩罚无意义的忽远忽近。

   🌟 Camera-Subject Heading Alignment (CSHA) 方位跟随：相机是否随主角转身而旋转（区别于监控视角）。

3. 运镜稳定性 (Stabilization)

   🌟 Horizon Stability (水平线稳定性) Exo 视角应消除 Ego 视角的 Roll 轴（歪头）抖动。

   🌟 Trajectory Smoothness Exo 轨迹加速度应远小于 Ego 轨迹（体现“智能防抖”）。

## TODO List

- [x] 🟢FVD: 视频生成领域必需的标准。衡量生成视频分布与真实视频分布的整体距离（真实感+连贯性）
- [x] 🟢IQ (Imaging Quality / MUSIQ)： 基于多尺度 Transformer 的无参考画质评估（清晰度、噪点）
- [x] 🟢AQ (Aesthetic Quality)： 基于 CLIP+MLP 的美学评分（构图、色彩）。
- [x] 🟢 TF (Temporal Flickering) 检查画面闪烁和噪点跳变。
- [x] 🟢 DD (Dynamic Degree) 防作弊指标，防止生成静止的“PPT视频”。
- [x] 🟢 SF Structural Fidelity (DINOv2) 使用 DINOv2 捕捉局部纹理和结构，比 CLIP 更细粒度。
- [x] 🟢 AC (Appearance Consistency / Fidelity)使用 CLIP Embedding 确保整体外观不漂移。
- [x] 🌟 HAA (Human Action Alignment) 基于骨骼关键点的姿态向量相似度。
- [x] 🌟SCCR SourceControlConditionRecall 在大规模干扰项中检索对应的 Ego 源视频，计算召回率。
- [x] 🌟TAA Temporal Attention Alignment 确保动作顺序正确（A->B->C），不仅仅是短时平滑。
- [x] 🌟 SSDC Side-by-Side Depth Consistency () 基于单目深度估计（Depth Anything V2）的场景几何一致性。
- [x] 🟢 SDR (Subject Detection Rate) 完整性检查，防止生成崩坏导致检测不到人。
- [x] 🌟ADE Geometric Drift ()计算生成轨迹与 GT 轨迹的误差。
- [x] 🌟 CCE (Camera Centering Error) 2D 构图指标：主角是否保持在视觉中心。
- [x] 🌟 CTE (Camera Trajectory Error) / Warp Error 3D 轨迹指标：利用 SLAM 估计相机轨迹，看是否符合 Follow 曲线。
- [x] 🌟SCDE Subject-Camera Distance Error 基于 3D HPE (WHAM/4DHumans) 测量 $t_z$。核心指标。
- [x] 🌟CSHA Camera-Subject Heading Alignment () 方位跟随：相机是否随主角转身而旋转（区别于监控视角）。
- [x] 🌟TS Trajectory Smoothness Exo 轨迹加速度应远小于 Ego 轨迹（体现“智能防抖”）。
- [ ] 🟢 Texture Style Distance (Gram Loss) 惩罚“过度平滑”或“油画感”，强制生成高频纹理。
- [ ] 🌟 Global Illumination Match (GIM) 确保环境氛围（黑夜/晴天）与 Ego 输入一致，基于 LAB 空间亮度直方图。
- [ ] 🌟 Shadow Consistency Score 接地性检查，解决“人物漂浮感”，强制生成合理的阴影。
- [x] 🟢 OFC (Optical Flow Correlation) 衡量运动模式的相关性。
- [ ] 🌟 Action Semantics Preservation (ASP) / Keystep Ego 做“步骤A”，Exo 也必须检测出“步骤A”。
- [ ] 🌟 Action Semantic Consistency (ASC) 防止“语义漂移”（如切菜变洗碗）。
- [ ] 🟢 SSIM / LPIPS 经典的结构相似性与深度感知相似度指标。
- [x] 🟢 MS (Motion Smoothness) 惩罚剧烈的加速度突变。
- [ ] 🌟 Hand-Object Interaction (HOI) Consistency 手物接触状态的一致性，防止“隔空取物”。
- [ ] 🌟 Motion Trajectory Deviation (MTD) 衡量“专家 vs 新手”的运动 Skill Gap。
- [ ] 🌟 Spatio-Temporal Correspondence 寻找 Ego 和 Exo 帧之间的显式对应关系。
- [ ] 🌟 Chamfer Distance (SfM Consistency) 硬核指标：通过 COLMAP 重建点云，对比“世界结构”是否歪曲。
- [ ] 🌟 FDE 衡量终点累积误差。
- [ ] 🌟 Time Alignment Error & Cycle-Consistency 毫秒级同步检查，确保物理接触瞬间（如击打）的时间戳对齐。
- [ ] 🌟 3D Follow Distance Variance (FDV) 呼吸效应：衡量跟随距离的方差，惩罚无意义的忽远忽近。
- [ ] 🌟 ❌Horizon Stability (水平线稳定性) Exo 视角应消除 Ego 视角的 Roll 轴（歪头）抖动。

## 重要性排序

| **维度 (Dimension)** | **子分类 (Sub-Category)** | **核心指标 (Selected Metric)**              | **核心价值 / 选择理由**                                      |
| -------------------- | ------------------------- | ------------------------------------------- | ------------------------------------------------------------ |
| **I. 视觉质量**      | 基础画质                  | 🟢 **FVD (Fréchet Video Distance)**          | **行业红线**。必须向社区证明生成视频的基础分布质量达到了主流水平。 |
|                      |                           | 🟢IQ (Imaging Quality / MUSIQ)：             | 基于多尺度 Transformer 的无参考画质评估（清晰度、噪点）      |
|                      |                           | 🟢AQ (Aesthetic Quality)：                   | 基于 CLIP+MLP 的美学评分（构图、色彩）。                     |
| **II. 时序动态**     | 流畅度                    | 🟢 **TF (Temporal Flickering)**              | **视觉稳定性**。确保画面没有异常闪烁，是观看体验的基础。     |
|                      | 运动模式                  | 🟢 **DD (Dynamic Degree)**                   | **防作弊**。防止模型通过生成“静态PPT”来骗取高分，确保视频真的在“动”。 |
| **III. 语义与动作**  | 身份与外观                | 🟢 **Structural Fidelity (DINOv2)**          | **强身份一致性**。相比 CLIP，DINOv2 对细粒度纹理和局部结构更敏感，确保“人没变”。 |
|                      | (及)                      | 🟢 AC (Appearance Consistency / Fidelity)    | 使用 CLIP Embedding 确保整体外观不漂移。                     |
|                      | 姿态与交互                | 🌟 **HAA (Human Action Alignment)**          | **姿态准确性**。Exo 视角的核心是看清全身动作，骨骼对齐是动作翻译准确性的直接证据。 |
|                      | 跨视角关联                | 🌟 **SCCR Source Control Condition Recall ** | 它是否包含了足够的独特特征，使其能够从成千上万个干扰视频（Distractors）中，精准地指回它唯一的那个 Ego 来源？难点在于外观泛化 (Zero-shot Appearance) |
|                      | (及)                      | 🌟 **Temporal Attention Alignment （TAA）**  | 强调Frame-level Alignment (帧级对齐)。生成视频是否“严丝合缝”？已知它是这个人生成的，但它生成的动作是第 1 分钟的“挥手”还是第 2 分钟的“转身”？ 难点在于相似动作区分 (Fine-grained Action) |
| **IV. 几何与透视**   | 深度与结构                | 🌟 **SSDC (Side-by-Side Depth Consistency)** | **空间几何**。利用单目深度模型检查场景的透视关系（如墙角、走廊）是否正确构建。 |
|                      | 时空对齐                  | 🌟 **Geometric Drift (ADE)**                 | **轨迹对齐**。衡量人物在 2D 平面上的运动路径是否偏离了 Ground Truth 的预期。 |
| **V. 相机运镜**      | 构图控制                  | 🌟 **CCE (Camera Centering Error)**          | **跟随基础**。最直观的指标，衡量“摄影师”是否时刻把主角放在画面视觉中心。 |
|                      | (及)                      | 🌟 **CTE (Camera Trajectory Error)**         | **3D 运镜**。更硬核的指标，通过 SLAM 评估相机的 3D 运动轨迹是否符合跟随曲线。 |
|                      | 跟随逻辑                  | 🌟**SCDE Subject-Camera Distance Error**     | **距离保持**。衡量相机是否像“栓了绳子”一样保持稳定距离，防止不合理的忽远忽近（呼吸效应）。 |
|                      | (及)                      | 🌟 **CSHA (Heading Alignment)**              | **方位跟随**。衡量相机是否随主角转身而旋转（绕到身后），这是区别“跟随视角”与“监控视角”的关键。 |
|                      | 运镜稳定性                | 🌟 **Trajectory Smoothness**                 | **智能防抖**。Exo 轨迹加速度应远小于 Ego 轨迹。              |

## Related work

[^1]:Towards Accurate Generative Models of Video: A New Metric & Challenges
[^2]:MUSIQ: Multi-scale Image Quality Transformer
[^3]:Learning Transferable Visual Models From Natural Language Supervision (Radford et al., ICML **2021**)
[^4]:LAION-5B: An open large-scale dataset for training next generation image-text models (Schuhmann et al., NeurIPS **2022**)
[^5]:RAFT: Recurrent All-Pairs Field Transforms for Optical Flow
[^6]:Image quality assessment: From error visibility to structural similarity
[^7]:The Unreasonable Effectiveness of Deep Features as a Perceptual Metric
[^8]:Following VBench (Huang et al., 2024), we employ Temporal Flickering (TF) and Motion Smoothness (MS) to evaluate the temporal coherence of the generated videos.