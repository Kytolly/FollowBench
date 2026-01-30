# 📜 Ego2ExoFollowShot 评测规则与提交指南

## 赛道概览 (Overview)

Ego2ExoFollowShot 旨在解决从第一人称（Ego）到第三人称（Exo）视角的生成问题。为了全面评估生成模型的**长程时序一致性** 和 **高频动作对齐能力**，本挑战赛设立了 **“长程稳定性赛道”**。

本赛道的核心挑战在于克服自回归生成中的**曝光偏差 (Exposure Bias)**，要求模型在长达 **300 帧** 的生成过程中保持画面不崩坏、动作不漂移。

## 技术规格 (Technical Specifications)

所有提交的视频必须**严格遵守**以下参数。任何参数不符的提交（如帧率不足、使用了插帧工具）将在 `validate` 阶段直接被判无效。

| **参数**     | **规格要求**         | **核心说明**                                                 |
| ------------ | -------------------- | ------------------------------------------------------------ |
| **分辨率**   | **426 x 240 (240p)** | 必须严格为此分辨率。我们关注动作与时序，而非纹理细节。请使用 `Center Crop` + `Resize` 预处理。 |
| **帧率**     | **60 fps**           | 必须与 Ground Truth 严格对齐。**严禁使用 RIFE 等插帧工具**进行后处理。如果模型生成 30fps，请直接复制帧或忍受时序错位扣分。 |
| **总帧数**   | **300 Frames**       | 对应 5.0 秒时长。少于 300 帧将导致 FVD 计算失败；多于 300 帧将被截断。 |
| **文件格式** | **.mp4 (H.264)**     | 推荐使用 `yuv420p` 像素格式以确保兼容性。                    |

## 数据使用与推理规则 (Data & Inference Policy)

为了保证学术公平性，请严格遵守以下“白名单”机制。

### ✅ 允许使用的输入 (Allowed Inputs)

1. **Ego Video**: 完整的 60fps 第一人称输入视频。
2. **Reference Image**: 指定的人物参考图。
3. **Text Prompts**: 官方提供的 `annotation.json` 中的文本提示词作为 baselines 推理输入，参赛者应该自行定制提示词以便提高模型的可控生成能力。
4. **Initial Context**: 仅允许使用 Ground Truth 的前 **16 帧** 作为自回归生成的初始 Context（Visual Prompt）。

分类按照10min长视频分类；按照ID和场景；

* 训练easy/训练hard/测试easy/测试hard；
* id： 截图；
* 场景：文字描述；
* 如果拿到一组数据，先要判断如果他作为测试集的话属于easy还是hard，再判断它属于测试集还是训练集；
* 我们一定要让有一定比例的hard在训练集；

训练/测试：4：1（40h:10h;5h）

训练数据： 60fps,36000F -> 1920x1080 

测试数据：

easy: 60fps 5s 

hard(不仅在于时长，还在于出现了更难的id和场景): 60fps 2min 

### 🚫 严禁行为 (Strictly Prohibited)

1. **禁止外部插帧**: 禁止使用任何基于深度学习的插帧模型（如 RIFE, FILM）来提升帧率。我们评测的是生成模型的原生时序密度。
2. **禁止超分后处理**: 禁止使用 Real-ESRGAN 等超分工具。
3. **禁止偷看未来**: 在生成第 $t$ 帧时，不得利用 $t$ 时刻之后的 Ground Truth 信息。

### 🤖 Oracle Baseline (特殊豁免)

- **适用对象**: 仅限 Pose-Guided 模型（如 AnimateDiff + ControlNet）。
- **规则**: 允许使用从 Ground Truth 提取的 **Pose Sequence** 作为输入。
- **标记**: 此类提交必须在 `submission.json` 中标记 `"is_oracle": true`。其成绩将在榜单中以星号（*）标注，代表该任务的理论上限。

## 提交格式 (Submission Format)

请使用 JSON 索引方式提交。

**目录结构：**

Plaintext

```
my_submission/
├── videos/
│   ├── 1001_gen.mp4  (240p, 60fps, 300f)
│   ├── 1002_gen.mp4
│   └── ...
└── submission.json
```

**submission.json 内容：**

JSON

```json
{
    "meta": {
        "team_name": "MyTeam",
        "model_name": "EgoGen-V1",
        "mode": "easy",
        "contact": "email@example.com"
    },
    "results": {
        "test-case-1001": {
            "generated video": "1001_gen.mp4",
        },
        "test-case-1002":{
            "generated video": "1002_gen.mp4",
        },
        // Key 必须与 Test Set ID 一致
    }
}
```

## 评测指标体系 (Evaluation Metrics)

由于长视频生成的特殊性，本赛道采用 **分段式评测 (Block-wise Evaluation)**。

### 质量与稳定性 (Quality & Stability)

- Block-FVD (↓):

  我们将 300 帧切分为 10 个片段（每段 30 帧），分别计算 FVD。

  - **指标意义**: 观察 FVD 随时间 $t$ 的变化曲线。优秀的模型应保持曲线平稳，而非呈指数级上升（Collapse）。

- TCS (Temporal Consistency Score) (↑):

  计算相邻帧的 CLIP Image Embedding 余弦相似度。衡量画面是否发生突变或闪烁。

### B. 动作与控制 (Motion & Control)

- HAA-60 (Human Action Alignment) (↑):

  在 60fps 密度下提取骨架并计算与 Ego 动作的对齐度。

  - *注意*: 如果您的模型是 30fps，该分数可能会因为“动作过快（Run-ahead）”而受到惩罚。

### C. 身份保持 (Identity)

- AC (Appearance Consistency) (↑):

  计算整个视频序列中人物特征与 Reference Image 的平均相似度。

## 自测脚本 (Self-Check)

在提交前，请使用以下脚本验证您的视频是否符合 **300帧/60fps/240p** 的硬性标准：

```Bash
# 检查视频属性
ego2exo validate \
    --submission ./my_submission/ \
    --expect-resolution 426x240 \
    --expect-fps 60 \
    --expect-frames 300

# 运行本地评测 (需要 GT 数据集)
ego2exo evaluate \
    --submission ./my_submission/ \
    --dataroot ./assets/Ego2ExoDataset \
    --device cuda:0
```

**Block-FVD 可视化**: 您的评测代码最好能直接输出一张 `.png` 折线图，画出 `Ours` vs `CogVideo` vs `Wan` 的 FVD 随时间变化曲线。这将是您论文中最有说服力的一张图。

## Baselines 说明

### 定制 prompts

所有的 baselines 由一个 `caption.json` 定制。为了公平地对比不同类型的基线模型（Baselines），Prompt 被设计为层级结构（Dictionary），而不是单一字符串。

1. **`t2v_generic` (文生视频基线)**:

   - **内容**：纯粹的场景和动作描述。
   - **用途**：给 CogVideo, Sora 等仅接受文本的模型。
   - *示例*："A woman is cutting vegetables in a kitchen."

2. **`i2v_generic` (图生视频/I2V基线)**:

   - **内容**：强调“匹配参考图”的指令。
   - **用途**：给 Wan-I2V, SVD-XT 等接受首帧/参考图的模型。
   - *示例*："A woman matching the reference image is cutting vegetables..."

3. **`for fullymodal model` (您的模型/Ego-Exo 专用模型)**:

   - **内容**：包含完整的任务指令，可能包含上述的特殊 Token。
   - **用途**：给 Ego2ExoFollowShot (Ours) 使用，以此以此激活最佳性能。
   - *示例*："Transform the first person view to third person view..." 或复用训练时的模板。

4. `WanVACE `模型使用 `TASK-TYPE` Token 作为前缀。默认将其映射为 `TASK-MOTION`（对应 Motion Transfer），因为 Ego2Exo 本质上是基于 Ego 运动生成 Exo 视频，这最符合 VACE 的定义。

5. 我们的参考 Baselines 采用如下结构化的 Prompt 进行标注，以适配 In Context LoRA 的要求。

   ```
   [EGO2EXO] [REF-ID] <主体描述> [ACTION] <动作描述> [TARGET-VIEW] <运镜/视角描述>
   [EGO2EXO]: 任务触发词，告诉模型“我要做视角转换任务”。
   [REF-ID]: 身份触发词，后接对人物外貌的详细描述（如 A woman with long red hair...），提示模型利用 Reference Image 进行特征注入。
   [ACTION]: 动作触发词，后接具体的行为描述（如 is cutting vegetables...），提示模型从 Ego 视频中提取运动信息。
   [TARGET-VIEW]: 视角触发词，后接期望的第三人称视角风格（如 cinematic medium shot from the side）。
   ```

6. **Negative Prompt**: 标准的负面词库（去抖动、去模糊、去第一人称视角残留）。

我们配置了两套 prompt 标注工具

- **Qwen2.5-VL**: 适合本地批量处理，无速率限制，且对视频理解能力极强（尤其是长视频）。代码使用了 `qwen_vl_utils` 标准处理流程。
- **Gemini 1.5 Pro**: 适合处理超长上下文或需要极高推理能力的场景。代码处理了视频上传到 Google File API 的异步等待逻辑。

### 参考归类建议

| **基线模型 (Model ID)**  | **建议归类 (Prompt Strategy)** | **命名含义**   | **适用场景**                                               |
| ------------------------ | ------------------------------ | -------------- | ---------------------------------------------------------- |
| **CogVideo**             | `t2v_generic`                  | 通用文生视频   | 仅依赖纯文本描述，无图像/视频条件。                        |
| **SVDTX**                | `i2v_generic`                  | 通用图生视频   | 依赖“文本 + 参考图(Ref Image)”。提示词需强调“匹配参考图”。 |
| **WanI2V**               | `i2v_generic`                  | ^              | ^                                                          |
| **WanVACE**              | `vace_instruct`                | VACE 指令微调  | 需要特定的 `TASK-MOTION` 前缀 Token。                      |
| AnimateDiff + ControlNet | exo->pose + image              |                | 接近ground truth  探究理论上限                             |
| **Ours (InContext)**     | `ours_lora`                    | 自研 LoRA 格式 | 需要 `[EGO2EXO] [REF-ID] ...` 等结构化特殊 Token。         |

### 数据集标注结构

```
{
    "case-1001": {
        "the first view": "test/1001/ego.mp4",
        "the third view": "test/1001/exo.mp4",
        "reference": "test/1001/ref.png"
        }
}
```

## Benchmark数据集说明

### 标准目录结构

```
assets/
├── train/
│   ├── train-case-xxx
│   │   ├── ego.mp4
│   │   ├── exo.mp4
│   │   └── ref.png
│   ├── ...
│   ├── train-case-xxx
│   │   ├── ego.mp4
│   │   ├── exo.mp4
│   │   └── ref.png
│   └── annotation.json
├── test/
│   ├── test-case-xxx
│   │   ├── ego.mp4
│   │   ├── exo.mp4
│   │   └── ref.png
│   ├── ...
│   ├── test-case-xxx
│   │   ├── ego.mp4
│   │   ├── exo.mp4
│   │   └── ref.png
│   └── annotation.json
├── ...
```

