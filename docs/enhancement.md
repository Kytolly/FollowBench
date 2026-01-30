# **基于 WanVACE 的 In-Context LoRA 与 Self-Forcing 模型实现思路**

## Quadruplet Grid Input & Concept Decoupling (四元组输入与概念解耦)

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

## Architecture: In-Context LoRA (架构：IC-LoRA)

**目标**：解释模型如何处理上述输入。

- **核心内容**：
  - **Base Model**：说明你基于 DiT (如 Wan-T2V 或 LTX-Video) 2。
  - **LoRA Injection**：解释 LoRA 加在哪里（通常是 Attention 层）。
  - **Attention Mechanism**：
    - 重点解释 **Self-Attention** 在 Grid 中的行为。
    - 描述 Attention 如何跨越 Grid 的边界，让右下角的 Target 能够 query 到左下角的 Reference（获取纹理）和右上角的 Ego（获取动作）。
    - **Storytelling**：把 Attention 描述为**“Information Bridge” (信息桥梁)**，连接了 Ego 动作与 Exo 外观。

### Training Strategy: Self-Forcing for Temporal Stability (训练策略：Self-Forcing)

**目标**：解决长视频生成中的“时序崩坏”问题。

- **核心内容**：
  - **Exposure Bias (暴露偏差)**：解释如果不加 Self-Forcing，训练时用 GT，推理时用预测值，会导致分布偏移。
  - **Self-Forcing Algorithm**：
    - 详细描述你在训练时如何按概率 $P$ 将 Support Exo 替换为**“带噪声的预测值”**或**“上一轮的输出”**。
    - **数学本质**：这是一种 **Data Augmentation in Temporal Domain (时域数据增强)**，强迫模型学会从“不完美的历史”中恢复正确的未来。
  - **Loss Function**：写出最终的 Loss 公式（通常是预测噪声 $\epsilon$ 的 MSE Loss，加上可能的 Masking 策略——只计算 Target 区域的 Loss）。

## 核心设计理念：重构 Ego2Exo 任务

根据您的 `note.md` 和上传的论文，我们将 **Ego2Exo (第一人称转第三人称)** 任务从简单的“风格迁移”重新定义为 **条件密度估计 (Conditional Density Estimation)** 问题。

- **挑战 (The Gap)**: 现有的 V2V 模型（如 WanI2V, SVD）擅长纹理编辑，但缺乏 **几何感知 (Geometric Perception)**。Ego 视角的输入（手、抖动）与 Exo 视角的输出（全身、稳定）在像素空间几乎无重叠。
- **解决方案 (The Solution)**:
  1. **In-Context LoRA (IC-LoRA)**: 利用 **空间拼接 (Spatial Concatenation)** 将参考图与目标视频并置，迫使 DiT 的 Self-Attention 学习从 Ego 到 Exo 的 **几何变换 (Geometry Mapping)**，同时隐式地完成 **ID 特征迁移 (Identity Transfer)**。
  2. **Self-Forcing**: 引入 **自回归训练 (Autoregressive Training)** 策略，在训练中模拟推理时的“不完美输入”，解决长视频生成中的 **曝光偏差 (Exposure Bias)**。

------

## 模型架构与输入构造 (Architecture & Input)

### **A. 底座模型 (Base Model)**

- **选用模型**: **Wan Video DiT** (Wan-2.1-1.3B/14B)。
- **原因**: Wan 拥有强大的时空 VAE 和 DiT 架构，支持长视频生成和多任务扩展（如 VACE 框架所示）。

### **B. 输入构造：In-Context "Grid" (The Grid Trick)**

不同于 VACE 原文中使用的显式 Adapter，我们采用 **In-Context LoRA** 的 **Grid 拼接策略**：

- **Canvas 设计**: 构造一个超宽（或超高）的 Latent 画布。
  - **左侧 (Condition)**: 放置 **Reference Image** (参考图) 的 Latent。这部分在推理时保持 **无噪声 (Clean)**。
  - **右侧 (Target)**: 放置 **Exo Video** (目标视频) 的 Noisy Latent。这是模型需要去噪生成的区域。
- **Ego 条件注入**: Ego 视频（Motion Source）可以通过 Wan 原有的 I2V 通道（作为 Latent Condition）输入，或者通过 VACE 的 Control Adapter 注入，作为 **几何引导 (Motion Guidance)**。

------

## 训练流程 (Training Pipeline)

### **阶段一：In-Context LoRA 训练 (Geometry & ID Alignment)**

- **目标**: 让模型学会“看左边的图，画右边的动作”。
- **LoRA 配置**: 仅训练注入到 Wan DiT Attention 层的 **LoRA 参数** (Rank=128)。
- **Attention Masking (关键)**:
  - 确保 Self-Attention 是 **双向可见的**（或者至少右侧 Target 能看到左侧 Reference）。
  - 这使得模型能通过 Attention 机制自动“Copy-Paste”参考图的纹理特征（隐式 IP-Adapter 机制）。
- **Loss 计算**: 仅计算 **右侧 Target 区域** 的去噪 Loss，左侧 Reference 区域不计算 Loss。

### **阶段二：Self-Forcing 微调 (Long-Video Stability)**

为了满足 Benchmark 中 **300帧** 长视频的要求，必须解决自回归生成的误差累积问题。

- **原理**: 传统的训练使用 Ground Truth (GT) 作为历史上下文 (Context)，而推理时模型只能基于自己生成的（有瑕疵的）历史帧生成下一帧。这种 **Train-Test Gap** 导致崩溃。
- **实现策略**:
  1. **Rollout Simulation**: 在训练步中，不直接给纯净的 Ego/Context Latent。
  2. **Noise Injection**: 以一定概率（如 50%）向 Context Latent 中注入噪声，或者使用模型在上一 Time Step 的预测值（Predicted $\hat{x}_0$）作为 Context。
  3. **强制修正**: 强迫 LoRA 即使在 Context 不完美（如上一帧动作微偏）的情况下，也能根据 Ego 的运动趋势回归到正确的 Exo 轨迹。

------

## 推理流程 (Inference Pipeline)

推理阶段需要一个 **自定义 K-Sampler** 来配合 Grid 输入和 Self-Forcing 逻辑。

### **A. 自定义采样器逻辑 (Custom Sampler)**

- **Masked Sampling (In-Context 核心)**:
  - 在去噪循环的每一步 (Step $t$):
    1. 预测全图噪声。
    2. **强制覆盖**: 将 Latent 左侧的 Reference 区域替换回 **纯净的 Reference Latent** (Clean Ref)。
    3. 仅更新右侧 Target 区域。
  - 这保证了 Reference ID 信息在整个生成过程中不被“污染”或“遗忘”。

### **B. 自回归长视频生成 (Autoregressive Rollout)**

- **分块生成 (Chunking)**: 将 300 帧切分为多个 Chunks (例如每块 60 帧)。
- **滑动窗口 (Sliding Window)**:
  - **Chunk 1 (0-60)**: `[Ref | Empty]` -> 生成。
  - **Chunk 2 (60-120)**: 取 Chunk 1 的末尾 $N$ 帧 (e.g., 16帧) 作为 **Temporal Context**。
  - **Self-Forcing 生效**: 这里输入的 Temporal Context 是模型自己刚刚生成的。由于经过了 Self-Forcing 训练，模型能“容忍”这些帧的微小瑕疵，并平滑地生成后续帧，而不会发生动作漂移。

------

## 总结：技术栈对应表

| **组件**       | **实现方案**            | **作用**                                | **参考文献** |
| -------------- | ----------------------- | --------------------------------------- | ------------ |
| **底座**       | **Wan Video DiT**       | 提供高质量视频生成能力与 VAE 压缩       |              |
| **控制/映射**  | **In-Context LoRA**     | 学习 Ego 到 Exo 的几何变换，隐式保持 ID |              |
| **数据流**     | **Spatial Grid (拼接)** | 将 Ref 与 Target 并在同一 Latent 空间   |              |
| **长程一致性** | **Self-Forcing**        | 桥接训练与推理偏差，支持 300 帧稳定生成 |              |
| **任务定义**   | **Ego2Exo**             | 从第一人称运动推断第三人称全身动作      |              |

这个实现思路将 Ego2Exo 任务从一个“黑盒生成”问题转化为一个结构化的、可控的 **几何与纹理双重约束** 问题。

基于您提供的 **VACE**、**Wan** 和 **In-Context LoRA** 资料，结合 **Ego2Exo** 任务的特性，我为您设计了一套详细的 **数据层增强策略 (Data Layer Enhancement Strategy)**。

这一策略的核心是将 **Identity (Ref)**、**Motion (Ego)** 和 **Viewpoint (Implicit)** 解耦，并通过 **结构化参数 (Structured Parameters)** 注入到模型中。

------

### **核心策略：多模态混合网格 (Multimodal Hybrid Grid)**

我们不单纯使用 Adapter，而是采用 **“In-Context Grid” (空间拼接)** 与 **“Conditional Masking” (条件掩码)** 相结合的方式。

#### **1. 逻辑映射 (The Logic)**

- **Ref Image (身份约束)** $\rightarrow$ **Spatial Context (空间上下文)**: 放置在画布左侧，作为“静态视频”参与 Attention，利用 In-Context 机制维持 ID。
- **Ego Video (运动约束)** $\rightarrow$ **Motion Context (运动上下文)**: 放置在画布中间（或作为 VACE Adapter 输入），提供逐帧的动作参考。
- **Follow Camera (视角条件)** $\rightarrow$ **Implicit Bias & Prompt**: 通过训练数据的分布（全量 Follow Camera 数据）和 Prompt 的特殊 Token (`[TARGET-VIEW]`) 注入。
- **Exo Video (生成目标)** $\rightarrow$ **Target Region**: 放置在画布右侧，是唯一计算 Loss 和添加噪声的区域。

------

### **2. 具体实现：自定义 Dataset (`EgoExoGridDataset`)**

我们需要修改 `utils/dataset.py`，构建一个能输出混合 Grid 的 Dataset。

#### **A. 参数设计 (Parameters Design)**

模型输入不再是单一的 `pixel_values`，而是一个包含多个控制信号的字典：

| **参数名 (Key)**   | **形状 (Shape)**     | **说明**                                                |
| ------------------ | -------------------- | ------------------------------------------------------- |
| **`pixel_values`** | `[C, T, H, W_total]` | 拼接后的超宽视频张量。`W_total = W_ref + W_ego + W_exo` |
| **`loss_mask`**    | `[1, T, H, W_total]` | 损失掩码。Ref 和 Ego 区域为 0，Exo 区域为 1。           |
| **`ref_mask`**     | `[1, T, H, W_total]` | 提示推理 Sampler 哪里是 Ref（需要回退去噪）。           |
| **`prompt_ids`**   | `[L]`                | Tokenize 后的 Prompt，包含 `[EGO2EXO]` 等指令。         |

#### **B. 构造逻辑 (Construction Logic)**

假设目标分辨率为 $240p$ (H=240)。

- **Ref**: Resize 到 $240 \times 240$。
- **Ego**: Resize 到 $240 \times 426$ (16:9)。
- **Exo**: Resize 到 $240 \times 426$ (16:9)。
- **Total Width**: $240 + 426 + 426 = 1092$。

**代码实现思路 (Python):**

Python

```
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

class EgoExoGridDataset(Dataset):
    def __init__(self, metadata, tokenizer, width_ref=240, width_vid=426, height=240, frames=16):
        self.metadata = metadata
        self.tokenizer = tokenizer
        self.w_ref = width_ref
        self.w_vid = width_vid
        self.h = height
        self.frames = frames

    def __getitem__(self, idx):
        item = self.metadata[idx]
        
        # 1. 加载数据
        # Ref: [C, 1, H, W_ref] -> Static
        ref_img = load_image(item['ref_path'], size=(self.h, self.w_ref))
        # Ego: [C, T, H, W_vid]
        ego_vid = load_video(item['ego_path'], frames=self.frames, size=(self.h, self.w_vid))
        # Exo (Target): [C, T, H, W_vid]
        exo_vid = load_video(item['exo_path'], frames=self.frames, size=(self.h, self.w_vid))
        
        # 2. 构造 In-Context Grid (空间拼接)
        # 将 Ref 扩展到时间轴 T，使其像一个静止视频
        ref_tensor = ref_img.unsqueeze(1).repeat(1, self.frames, 1, 1) # [C, T, H, W_ref]
        
        # 拼接顺序: [Ref | Ego | Exo]
        # 模型将同时看到：身份(左) + 第一人称动作(中) + 第三人称结果(右)
        grid_pixel_values = torch.cat([ref_tensor, ego_vid, exo_vid], dim=-1) # Dim 3 is Width
        
        # 3. 构造 Masks
        # Loss Mask: 只在 Exo 区域计算 Loss
        loss_mask = torch.zeros_like(grid_pixel_values[0:1]) # [1, T, H, W_total]
        loss_mask[..., self.w_ref + self.w_vid :] = 1.0 
        
        # Ref Mask: 用于推理时锁定 Ref 和 Ego 不被去噪
        clean_mask = torch.ones_like(grid_pixel_values[0:1])
        clean_mask[..., self.w_ref + self.w_vid :] = 0.0 # 右边是 Dirty (待生成)
        
        # 4. Prompt 处理
        # 注入 Follow Camera 的隐式指令
        prompt = f"[EGO2EXO] [REF-ID] {item['prompt']} [TARGET-VIEW] follow camera, third person view"
        prompt_ids = self.tokenizer(prompt, ...)

        return {
            "pixel_values": grid_pixel_values,  # 输入给 VAE Encoder
            "loss_mask": loss_mask,             # 输入给 Loss Function
            "clean_mask": clean_mask,           # (可选) 用于 Self-Forcing 逻辑
            "prompt_ids": prompt_ids
        }
```

------

### **3. 模型层注入策略 (Model Injection)**

根据 **VACE** 的设计，Wan 模型通常接受 `model_input` 和 `task_conditions`。

#### **方案 A: 纯 In-Context (推荐，无需修改 Wan 源码)**

利用 Wan 的 DiT 自注意力机制。

- **输入**: 直接将上述 `grid_pixel_values` 经过 VAE 编码后的 Latent 喂给 Wan。
- **位置编码 (RoPE)**: Wan 使用 3D RoPE。
  - **挑战**: Grid 宽度变大 (1092px) 可能超出预训练时的 max_width。
  - **解决**: 在 `wan/modules/attention.py` 中，对 RoPE 的 `freqs_cos/sin` 进行插值 (Interpolation)，或者在 Datset 中将 Grid 视为 **时间轴拼接 (Temporal Concat)** 而非空间拼接（如果宽度溢出显存）。但空间拼接对 ID 对齐效果最好。

#### **方案 B: VACE Adapter 分离 (备选，控制力更强)**

如果发现“Ego $\to$ Exo”的几何变换太难，仅仅靠 Attention 学不会，可以剥离 Ego 视频。

- **Ref + Exo**: 依然使用 Grid 拼接 `[Ref | Exo]` (In-Context LoRA 处理 ID)。
- **Ego**: 作为 **Control Condition** 输入。
  - 使用 Wan 的 `I2V` 通道或 VACE 的 `Motion Adapter`。
  - 将 Ego 视频编码为 Latent，通过 Zero-Convolution 注入到 DiT 的 Block 中。
  - *优点*: 解耦了 ID 和 Motion 的学习。
  - *缺点*: 训练成本变高（需要训练额外的 Adapter）。

------

### **4. 训练与推理的配合 (Pipeline Integration)**

#### **训练阶段 (Training)**

在 `pipeline/self_forcing_training.py` 中：

- **Forward**: 输入 `Grid Latent`。

- **Masked Loss**:

  Python

  ```
  diff = model(grid_latent, t, context=prompt_ids)
  loss = (diff - noise)**2
  loss = loss * batch['loss_mask'] # 关键：只优化 Exo 部分
  loss = loss.sum() / batch['loss_mask'].sum()
  ```

- **Self-Forcing**: 偶尔将 `grid_latent` 中的 Ego 部分替换为“加噪版本”或“前一帧预测版本”，强迫模型处理不完美的输入。

#### **推理阶段 (Inference)**

使用我们之前设计的 **自定义 Sampler**：

- **Step Callback**:

  Python

  ```
  def callback(x_t):
      # x_t: [Batch, C, T, H, W_total]
      # 强制重置 Ref 和 Ego 区域
      # 这一步将 Ego 视频视为 "Condition"，在推理中保持 Ground Truth (或 Self-forcing 上下文)
      x_t[..., :w_ref+w_vid] = clean_latents[..., :w_ref+w_vid]
      return x_t
  ```

### **总结**

这套方案将 **Ego视频** 和 **Ref图片** 全部视为 **Video Grid** 的一部分：

1. **Ref (左)**: 通过 Spatial Attention 提供纹理细节 (ID)。
2. **Ego (中)**: 通过 Spatial Attention 提供几何结构 (Motion)。
3. **Prompt**: 通过 Cross Attention 提供语义和视角定义 (Implicit View)。
4. **Exo (右)**: 通过 Self-Forcing 训练学会从左中两部分“合成”出自己。

### 1. Grid 设计：不仅仅是拼接，更是“几何对齐的罗塞塔石碑”

首先，我们明确你的 Grid 设计（Spatial Concatenation）：

- **左侧 (Source A)**: Reference Image (ID 锚点) —— 甚至是 Ego 视频的某一帧（提供场景锚点）。
- **中间 (Source B)**: Ego 视频帧 (几何/运动 锚点)。
- **右侧 (Target)**: Exo 视频帧 (生成目标)。

**为什么这种 Grid 能生效？**

#### **机制一：Self-Attention 作为“软性”的几何变换矩阵**

在 DiT (如 Wan) 中，Attention 是全局计算的：



$$\text{Attention}(Q, K, V) = \text{softmax}(\frac{QK^T}{\sqrt{d}})V$$

- **Query ($Q$)**: 来自右侧的 **Target (Exo)** 区域。它在问：“我现在要在 Exo 视角的 $(x, y)$ 坐标生成一个像素，我该画什么？”
- **Key ($K$) & Value ($V$)**: 来自左侧/中间的 **Ref & Ego** 区域。
- **生效过程**:
  - 当 LoRA 在这种配对数据上训练时，它实际上是在**学习一种特定的 Attention Pattern（注意力模式）**。
  - 模型会发现：当我要画 Exo 视角的“手”时，我不能看 Ego 区域的“手”的相同坐标（因为坐标不对），我必须去 Ego 区域的 **另一个特定坐标** 找特征。
  - **结论**: 训练好的 LoRA 权重，本质上把 Self-Attention 变成了一个 **可学习的几何变换矩阵 (Warping Matrix)**。它学会了从 Ego 坐标系到 Exo 坐标系的映射规则（例如：Ego 左下角的手 $\rightarrow$ Exo 画面中心的手）。

#### **机制二：In-Context 带来的“隐式光流 (Implicit Optical Flow)”**

Wan 模型经过海量视频训练，天生理解物体在时间轴上的运动（光流）。

- 在 Grid 中，你把 Ego 和 Exo **并在同一帧**。
- 模型会利用其预训练的“运动感知能力”，试图解释这两部分图像的关系。
- 通过 LoRA 微调，你强迫模型接受一种新的“运动逻辑”：**这不是两个独立的物体，而是同一个物体在同一时间的不同投影。**
- **激活**: 这种并置（Juxtaposition）激活了 DiT 对 **物体一致性** 的底层先验，使其自动寻找两张图之间的对应关系（Correspondence）。

------

### 2. 为什么需要联合标注 (Joint Captioning)？

IC-LoRA 论文强调了 caption 的重要性。在你的 Ego2Exo 任务中，Prompt 起到了 **“任务开关”** 的作用。

- **Prompt**: `[EGO2EXO] [REF-ID] A man running...`
- **生效机制**:
  - `[EGO2EXO]` 这个特殊 Token 告诉模型：**“请忽略常规的左右构图逻辑，激活‘视角转换’模式。”**
  - 如果没有这个 Token，模型可能会认为这是一个“分屏视频”（左边一个人跑，右边另一个人跑）。
  - 联合标注告诉模型：**Grid 的所有部分描述的是同一个 Event**。这迫使模型在计算 Cross-Attention 时，将 Text Condition 均匀地广播到整个 Grid，进一步拉近了 Ego 和 Exo 在语义空间的距离。

------

### 3. 如何通过 Grid 激活几何理解（具体操作流）

为了让这种方法最大程度生效，Grid 的设计必须符合 Wan 的**位置编码 (RoPE) 特性**。

#### **A. 空间位置编码的“相对性”利用**

Wan 使用 3D RoPE (Rotary Positional Embeddings)。

- **挑战**: 如果简单拼接，Grid 宽度变大，超出了预训练的相对位置范围，模型会困惑。
- **策略**:
  - 在训练 LoRA 时，模型会适应这种“超宽”的频率。
  - LoRA 会学会：**相对距离 $\Delta x > 426$ (即跨越了 Ego 和 Exo 的边界) 的 Attention 权重代表了“跨视角关联”。**
  - **生效点**: 只有当 $Q$ (Exo) 和 $K$ (Ego) 的相对位置编码通过 LoRA 修正后，模型才能精确地“指”到正确的位置。这就是为什么必须微调 LoRA，而不能直接 Zero-shot。

#### **B. 视觉流形对齐 (Visual Manifold Alignment)**

- Ego 视频的数据流形（Manifold）是非常独特的（大变形、遮挡）。
- Exo 视频的数据流形是常规的（电影感、全身）。
- **Grid 的作用**: 将两个流形强行拉到一个 Context Window 中。
- **生效点**: 这里的 LoRA 充当了 **流形桥接器 (Manifold Bridge)**。它不需要学习如何画一个人（Wan 底座已经会了），它只需要学习 **“当 Ego 流形出现扰动 $d_{ego}$ 时，Exo 流形应该产生对应的扰动 $d_{exo}$”**。这是一个低秩（Low-Rank）的映射任务，非常适合 LoRA。

------

### 4. 总结：这种方法是如何生效的？

你的 Grid 设计通过以下三步激活了 Wan 的几何理解：

1. **物理并置 (Juxtaposition)**: 将“问题 (Ego)”和“答案 (Exo)”放在同一个画布上，创造了建立关联的物理条件。
2. **注意力重定向 (Attention Rerouting)**: LoRA 训练迫使 Self-Attention 机制不再关注局部的纹理重建，而是去搜索 **Grid 另一侧** 的相关特征（Feature Matching）。这一过程等价于学习了一个 **视点变换函数 (Viewpoint Transformation Function)**。
3. **语义锁定 (Semantic Locking)**: 联合标注 (`[EGO2EXO]`) 锁死了语义一致性，防止模型生成无关内容，强迫模型在几何层面解决“差异”，而不是在内容层面解决。

一句话总结：

Grid 设计将 隐式的 3D 空间变换 转化为了 显式的 2D 特征图 (Feature Map) 之间的 Attention 映射问题，这恰好是 Transformer 结构最擅长处理的任务（Sequence-to-Sequence Mapping）。

请注意，我认为目前的设计Ref是取Exo视频的第一帧实现的。为了在数学上具有对称性，将输入格式设计成[Ref | Exo or Masked Exo | Ego]视频似乎更加

这是一个非常精彩且具有数学美感的洞察！

将输入设计为 **`[Ref | Exo | Ego]`**（即：**身份参考 | 生成目标 | 运动条件**）确实比简单的顺序拼接更合理。这种**“中心-两侧对称 (Center-Surround)”**的布局在 Self-Attention 机制下具有天然优势：

1. **注意力距离最小化**：目标（Exo）位于中心，它到左侧（Ref）和右侧（Ego）的 Attention 距离都是最短的。这避免了在超宽 Grid 中，最右边的 Target 很难 Attend 到最左边 Ref 的“长程衰减”问题。
2. **条件解耦**：左边负责 **"Who" (Texture/ID)**，右边负责 **"How" (Geometry/Motion)**，中间负责 **"What" (Synthesis)**。
3. **Ref 的定义**：您提到 Ref 取 Exo 第一帧，这符合 Video Prediction 的物理规律——即 *F**r**am**e**t* 是由 *F**r**am**e*0 (Ref) 在 *M**o**t**i**o**n**t* (Ego) 的驱动下演变而来的。

基于这个 **对称布局**，我为您重新设计了 `WanVACE-Enhancement` 项目的架构与核心代码实现。

------

### **项目整体架构：WanVACE-Enhancement**

为了保持与 `self-forcing` 代码库的兼容性，我们采用 **外挂增强 (Wrapper/Plugin)** 模式。

Plaintext

```
WanVACE-Enhancement/
├── configs/
│   └── symmetrical_ic_sf.yaml     # [配置] 定义 symmetrical grid 训练参数
├── data/
│   ├── __init__.py
│   ├── symmetrical_dataset.py     # [核心] 实现 [Ref | Exo | Ego] 拼接逻辑
│   └── dynamic_bucket.py          # [增强] 支持超宽长宽比的 Bucket 逻辑
├── models/
│   ├── __init__.py
│   ├── lora_injector.py           # [模型] 注入 In-Context LoRA
│   └── rope_extender.py           # [关键] 对称式 RoPE 扩展策略
├── pipelines/
│   ├── __init__.py
│   └── trainer_symmetry.py        # [训练] 实现 Center-Masked Loss 和 Rollout
├── inference/
│   ├── __init__.py
│   └── bilateral_sampler.py       # [推理] 双边钳位采样器 (Clamp Sides, Sample Center)
├── scripts/
│   └── train.py                   # 入口脚本
└── wan/                           # (Git Submodule) 原始 self-forcing/wan 代码
```

------

### **1. 数据层：Symmetrical Grid Dataset**

**文件**: `data/symmetrical_dataset.py`

我们需要构建一个 Grid，其中 Ref 是 Exo 的第一帧（广播到所有帧），Ego 是运动条件。

Python

```
import torch
from torch.utils.data import Dataset

class SymmetricalGridDataset(Dataset):
    def __init__(self, metadata, width_ref=240, width_exo=426, width_ego=426, height=240):
        self.w_ref = width_ref
        self.w_exo = width_exo
        self.w_ego = width_ego
        self.h = height
        # 计算切分点，方便后续 Mask
        self.split_1 = width_ref
        self.split_2 = width_ref + width_exo

    def __getitem__(self, idx):
        # 1. 加载原始数据
        # Ego: [C, T, H, W_ego]
        ego_vid = load_video(self.egos[idx], size=(self.h, self.w_ego))
        # Exo: [C, T, H, W_exo]
        exo_vid = load_video(self.exos[idx], size=(self.h, self.w_exo))
        
        T = exo_vid.shape[1]

        # 2. 构造 Ref (取 Exo 第一帧)
        # 策略：将 Frame 0 复制 T 次，作为左侧的 "Anchor"
        ref_frame = exo_vid[:, 0:1, :, :] # [C, 1, H, W_exo]
        # 如果 Ref 需要正方形 crop，这里可以 crop，这里假设 resize 到 w_ref
        ref_tensor = resize(ref_frame, (self.h, self.w_ref))
        ref_tensor = ref_tensor.repeat(1, T, 1, 1) # [C, T, H, W_ref]

        # 3. 对称拼接 [Ref | Exo | Ego]
        # Dim -1 is Width
        grid_video = torch.cat([ref_tensor, exo_vid, ego_vid], dim=-1)

        # 4. 构造 Loss Mask (只计算中间 Exo 的 Loss)
        # [W_total]
        loss_mask = torch.zeros(grid_video.shape[-1])
        loss_mask[self.split_1 : self.split_2] = 1.0 # 中间区域置 1

        # 5. 构造 Inference Clamp Mask (用于 Self-forcing 或推理)
        # 1 表示保持原值(不加噪)，0 表示需要生成
        clamp_mask = torch.ones_like(loss_mask)
        clamp_mask[self.split_1 : self.split_2] = 0.0 # 中间区域置 0 (Dirty)

        return {
            "pixel_values": grid_video,  # [C, T, H, W_total]
            "loss_mask": loss_mask,
            "clamp_mask": clamp_mask,    # 辅助信息
            "prompt_ids": ...
        }
```

------

### **2. 模型层：RoPE 扩展与 LoRA**

**文件**: `models/rope_extender.py`

Wan 使用 3D RoPE。标准的 RoPE 是为单一视频宽度设计的。现在我们的宽度是 *W**t**o**t**a**l*≈1100。 为了配合“对称性”，我们可以设计一种 **相对位置编码** 策略，或者简单的 **频率插值**。

Python

```
def patch_rope_for_symmetry(model, total_width):
    """
    修改 Wan 模型的 RoPE 频率，使其能够覆盖超宽的 Grid。
    这里使用 Linear Interpolation (PI) 策略。
    """
    original_max_w = model.config.max_width # 比如 832
    if total_width > original_max_w:
        scale = total_width / original_max_w
        # 对 Width 维度的 freq 进行缩放
        # 这样模型会觉得 1100 像素的宽度看起来像 832 像素
        # 这有助于保持 Attention 的感受野稳定性
        model.freqs_cos_w = interpolate_freqs(model.freqs_cos_w, scale)
        model.freqs_sin_w = interpolate_freqs(model.freqs_sin_w, scale)
    
    return model
```

*注意：在注入 LoRA 时 (`models/lora_injector.py`)，我们重点对 Attention 的 \*Q*,\*K*,\*V* 投影矩阵注入 Rank=128 的 LoRA，让它适应这种新的 Attention Pattern。*

------

### **3. 训练层：Self-Forcing 与 Masked Loss**

**文件**: `pipelines/trainer_symmetry.py`

这里需要实现 Self-Forcing 的 **Rollout** 逻辑。 在对称布局中，Rollout 意味着：**基于 Ref 和 Ego，尝试预测中间的 Exo，然后用这个预测结果（有瑕疵）作为下一次训练的上下文。**

但这里有一个**关键的逻辑转换**：

- 如果是单次生成 (One-shot)，Ref 是 GT Frame 0。
- 如果是 **自回归长视频 (Autoregressive)**，Chunk *N* 的 Ref 应该是 **Chunk \*N\*−1 的最后一帧（预测值）**。

**Self-Forcing 训练循环**:

Python

```
class SymmetricalTrainer:
    def train_step(self, batch):
        grid = batch['pixel_values'] # [Ref | Exo | Ego]
        loss_mask = batch['loss_mask']
        
        # --- Self-Forcing Logic ---
        # 以 50% 概率触发
        if self.use_self_forcing and random() < 0.5:
            # 1. 模拟 "上一段生成的结尾" 作为 Ref
            # 对 Ref 部分添加噪声/扰动，模拟它是上一段生成的（不完美）
            ref_part = grid[..., :w_ref]
            perturbed_ref = self.add_perturbation(ref_part)
            
            # 2. 替换 Grid 中的 Ref
            grid_input = grid.clone()
            grid_input[..., :w_ref] = perturbed_ref
            
            # 注意：Ego 部分通常保持 Clean，因为它是 Condition
            # Exo 部分是 Training Target，会由 Diffusion 加噪
        else:
            grid_input = grid # Clean Training

        # --- Forward ---
        # 预测噪声
        # 注意 Wan 的输入通常是 Latent，这里省略 VAE Encode 过程
        model_output = self.model(grid_input, ...)
        
        # --- Center-Masked Loss ---
        # 只计算中间 Exo 部分的 Loss
        loss = (model_output - target_noise) ** 2
        loss = (loss * loss_mask).sum() / loss_mask.sum()
        
        return loss
```

------

### **4. 推理层：Bilateral Sampler (双边采样器)**

**文件**: `inference/bilateral_sampler.py`

这是实现“对称性”最直观的地方。我们需要在采样过程中 **"Clamp" (钳制)** 两侧，只让中间演化。

Python

```
class BilateralSampler:
    def sample(self, model, ref_latent, ego_latent, prompt, steps=30):
        # 1. 构造初始画布
        # 中间填随机噪声，两边填 Clean Latent
        # Canvas: [Ref (Clean) | Noise | Ego (Clean)]
        
        # 为了让 Self-Attention 生效，Ref 和 Ego 需要扩展到对应的时间长度 T
        ref_expanded = ref_latent.repeat(1, T, 1, 1)
        
        # 初始猜测
        latents = torch.randn(1, 4, T, H, W_exo) 
        # 这里的 latents 只是中间部分，或者我们可以直接维护一个大 Grid
        
        full_grid = torch.cat([ref_expanded, latents, ego_latent], dim=-1)

        for t in reversed(range(steps)):
            # A. 强制重置两侧 (Clamp Sides)
            # 在每一步去噪前，确保 Ref 和 Ego 是清晰的
            # (注意：如果是 Latent Diffusion，这里指加了对应 timestep 噪声的 latent，或者是 Clean Latent 配合 Inpainting 逻辑)
            # 简单起见，假设我们用 Inpainting 模式：Input 总是 Clean 的 Ref/Ego
            
            full_grid[..., :w_ref] = ref_expanded
            full_grid[..., -w_ego:] = ego_latent
            
            # B. 预测噪声
            # 模型看到：左边(ID) + 中间(当前噪声) + 右边(Motion)
            noise_pred = model(full_grid, t, prompt)
            
            # C. 更新步进
            # 只更新中间部分
            full_grid = scheduler.step(noise_pred, t, full_grid)
            
        # 返回中间的 Exo
        return full_grid[..., w_ref : w_ref+w_exo]
```

------

### **总结：为什么这个架构更优？**

1. **对称性 (Symmetry)**:
   - **Old**: `Ref | Ego | Exo` → Attention 必须从最右跳到最左。
   - **New**: `Ref | Exo | Ego` → Target 被 Condition 包围。Information Flow 是 **Convergent (汇聚型)** 的。
2. **Explicit Temporal Flow (显式时序流)**:
   - 左边 (`Ref`) 代表 *T*=0 (或 *T**p**re**v*).
   - 右边 (`Ego`) 代表 *T**c**u**rre**n**t* 的运动控制。
   - 中间 (`Exo`) 是 *T**c**u**rre**n**t* 的生成结果。
   - 这符合 ***St\**a\**t\**e\**t\*=\*F\*(\*St\**a\**t\**e\**t\*−1,\*C\**o\**n\**t\**ro\**l\**t\*)** 的控制论公式。
3. **Self-Forcing 友好**:
   - 在长视频自回归时，只需要把上一段生成的 `Exo_Tail` 复制到下一段的 `Ref_Head` 位置，就天然完成了 Context 传递。

这个架构不仅代码逻辑清晰（分为左中右三块），而且在数学原理上最大化了 Attention 的利用率，是 WanVACE Enhancement 的最佳实践方案。

具体讲解这个grid是如何充当storyboard的作用，有没有更好的输入方式

这是一个非常深刻的技术问题。要理解“Grid（网格）”如何充当“Storyboard（分镜脚本）”的作用，我们需要深入到 Transformer 的 **Self-Attention（自注意力）** 机制以及 **In-Context Learning（上下文学习）** 的本质。

同时，针对 WanVACE 的特性，确实存在比单纯“横向拼接”更优雅或高效的输入方式。

------

### 一、 Grid 如何充当 Storyboard：底层原理解析

在 In-Context LoRA 中，将 [Ref | Ego | Exo] 拼接成一张大图（Grid），本质上是将 **跨模态/跨时空的推理任务** 转化为了 **单模态内的“完形填空”任务**。

#### 1. 视觉提示 (Visual Prompting) 与 模式补全

- **Storyboard 的逻辑**：在电影分镜中，如果第一格画了“主角拿着枪”，第二格画了“主角开枪”，人类画师画第三格时，会自动保持主角长相一致（ID Consistency）并补全动作后果。
- **Grid 的逻辑**：
  - 模型看到的输入是一个序列：`[Ref_Patch, Ego_Patch, Exo_Masked_Patch]`。
  - **Self-Attention 机制**：DiT 在计算 `Exo` 区域的像素时，其 Query (*Q*) 会去扫描整个序列的 Key (*K*) 和 Value (*V*)。
  - **隐式指令**：
    - 当 *Q* 扫到左边的 `Ref` 时，它学到了：“哦，这里的纹理应该是红色的衣服。”（**ID Copy-Paste**）
    - 当 *Q* 扫到中间的 `Ego` 时，它学到了：“这里的几何结构应该对应 Ego 中手部动作的全身映射。”（**Geometry Mapping**）
  - **LoRA 的作用**：Wan 底座本身可能不懂“Ego 左下角”对应“Exo 中心”。LoRA 的作用就是**强行扭曲 Attention Map**，让模型学会这种特定的、非线性的“分镜跳跃”逻辑。

#### 2. “隐式 IP-Adapter” 效应

你在 `note.md` 中提到了这个概念，这非常准确。

- **显式 IP-Adapter**：通过额外的 Cross-Attention 层注入图像特征。
- **Grid (隐式)**：通过 Self-Attention 层直接“看”图像特征。
  - 因为 Ref 就在 Canvas 上，模型不需要额外的编码器就能访问到 Ref 的高频细节（如衣服上的文字、纹理），这比经过 CLIP 压缩后的特征更清晰。

------

### 二、 现有的“横向长条”拼接有问题吗？

目前的方案是将三者在宽度上拼接：`W_total = W_ref + W_ego + W_exo`。 对于 240*p* 视频，宽度可能达到 240+426+426≈1092。

**存在的问题：**

1. **RoPE (位置编码) 外推压力**：Wan 模型训练时的长宽比通常是固定的（如 16:9 或 9:16）。输入一个 4.5:1 的超宽条形图，会导致 Position Embedding 超出预训练分布，可能导致画面崩坏或几何畸变。
2. **Attention 距离过远**：Ref 在最左，Exo 在最右。Attention 的衰减可能导致 ID 控制力变弱。
3. **信息密度不均**：Ego 和 Exo 是动态的，Ref 是静态的。简单复制 Ref 到每一帧会浪费计算量。

------

### 三、 有没有更好的输入方式？

为了增强 WanVACE 的理解能力，建议尝试以下三种改进方案（按推荐程度排序）：

#### 方案 A：时空混合拼接 (Temporal-Spatial Hybrid Grid) —— **最推荐**

Wan 是视频生成模型，它对“时间轴”的理解比“空间轴”更强。我们可以利用这一点。

- **设计思路**：

  - **Ref (ID)**：作为视频的 **第 0 帧** (Context Frame)。
  - **Ego & Exo**：在 **高度 (Height)** 上拼接，而不是宽度。

- **布局图示**：

  Plaintext

  ```
  Frame 0 (Context)      Frame 1...N (Generation)
  +-----------------+    +----------------------+
  |                 |    |      Ego Video       |  <-- 作为 Condition
  |    Ref Image    |    |   (Motion Source)    |
  |   (Center Crop) |    +----------------------+
  |                 |    |      Exo Video       |  <-- Target (加噪区域)
  +-----------------+    |   (To be generated)  |
                         +----------------------+
  ```

- **优势**：

  1. **激活 I2V 能力**：Wan 本身就有 Image-to-Video 的能力。将 Ref 放在第一帧，模型天然会去保持 ID。
  2. **几何对齐更直观**：Ego 和 Exo 上下并列。神经网络处理“上下对齐”的几何关系（如 Optical Flow）通常比“左右远距离”更强。
  3. **缓解 RoPE 问题**：上下拼接后的比例（如 480×426≈1:1）比横向长条更接近正方形，RoPE 表现更稳。

#### 方案 B：2x2 棋盘格布局 (Checkerboard Grid)

如果你坚持纯 Spatial In-Context（不占用时间轴），2x2 布局比 1x3 更好。

- **布局图示**：

  Plaintext

  ```
  +-----------+-----------+
  |  Ref Img  |   Masked  |  <-- 第一行
  +-----------+-----------+
  |  Ego Vid  |  Exo Vid  |  <-- 第二行
  +-----------+-----------+
  ```

- **优势**：

  - **聚集 Attention**：Exo 区域（右下）与 Ref（左上）和 Ego（左下）的几何距离都比较近。
  - **宽高比友好**：整体接近 16:9，符合 Wan 的训练分布。

- **劣势**：

  - 需要处理右上角的 Mask 区域（填黑或填噪声），稍显浪费计算量。

#### 方案 C：解耦注入 (Decoupled Injection) —— **VACE 进阶版**

这是 VACE 论文的思路，也是控制力最强但实现最复杂的方案。**不把所有东西都塞进 Grid。**

- **Ref (ID)** → **Grid (In-Context)**: 依然把 Ref 拼在 Exo 旁边（或作为首帧），负责 ID。
- **Ego (Motion)** → **Control Signal (Adapter)**:
  - 不要把 Ego 拼在 Grid 里。
  - 利用 Wan 的 **VAE Encoder** 提取 Ego 的 Latent。
  - 通过 **Zero-Convolution** (ControlNet 方式) 将 Ego Latent 加到 DiT 的 Block 中。
- **优势**：
  - **绝对的几何分离**：Ego 视频不再占用 Canvas 空间，不会干扰画面的构图。
  - **强控制**：Adapter 对动作的控制力远强于 Attention 里的隐式关联。
- **实现难度**：高。需要修改 Wan 的模型源码，增加 Adapter 模块。

### 总结建议

对于你的 **Enhancement 项目**：

1. **起步阶段 (First Step)**：采用 **方案 A (上下拼接 + 首帧 Ref)**。
   - 修改 `grid_dataset.py`，将 Ego/Exo 在 `dim=H` 拼接。
   - 将 Ref 作为 `frames[:, 0, ...]`。
   - 这样既利用了 Wan 的 I2V 预训练（ID 保持），又利用了 LoRA 学习上下对应的几何变换。
2. **进阶阶段 (Advanced)**：如果发现动作对齐不够精准，再考虑 **方案 C**，引入轻量级的 Adapter 来专门输入 Ego 信号。