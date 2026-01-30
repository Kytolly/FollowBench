目前的生成式模型（大多基于 Web-scale 视频训练）和数据集（如 Ego4D，通常只有 Ego 视频而没有同步的高精度眼动数据）确实构成了评估“视线-躯干解耦”的双重壁垒。

将这一部分作为 **Limitations** 或 **Future Work** 撰写，可以为您的论文增加理论厚度。为您草拟了一段学术表达，供您参考：

您可以从 **数据约束** 和 **模型能力** 两个维度来论述为什么目前无法完美评估这一点：

> **Limitations and Future Directions: Head-Body Decoupling & Gaze Anticipation**
>
> While our benchmark establishes a robust framework for assessing locomotion and action semantics in Ego2Exo translation, we acknowledge a limitation regarding the fine-grained evaluation of **head-body decoupling**.
>
> 1. **Data Constraint (Gaze Ground Truth)**: Current large-scale egocentric datasets (e.g., Ego4D) predominantly rely on chest-mounted or head-mounted cameras without synchronized eye-tracking or independent head-pose sensors. This makes it difficult to extract reliable "ground truth gaze" to distinguish attention focus from locomotion direction.
> 2. **Model Capability Gap**: Current state-of-the-art generative models function primarily as "motion followers," inferring trajectory from the camera's optical flow. They have yet to demonstrate the biological capability of **"gaze anticipation"**—where the head turns towards a target *before* the body trajectory changes.
>
> **Future Work**: We envision that future iterations of this benchmark could incorporate **gaze-specific metrics** (e.g., using eye-tracking subsets) to evaluate whether generative models can simulate the cognitive aspect of "looking before moving," moving beyond simple kinematic mapping to behavioral simulation.

这样处理的好处

1. **防守性 (Defensive)**：预判了审稿人可能提出的“为什么不看头？”的质疑，并给出了“数据不支持”这一无法反驳的理由。
2. **前瞻性 (Visionary)**：指出了 Ego2Exo 任务的下一个里程碑——从“物理运动生成”进化到“认知行为模拟”。