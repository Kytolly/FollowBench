为完成Ego2Exo任务我们设计了baselines；

## World Wander

基于模型架构的方法

## Self-forcing

基于训练的方法：

## Video-Editing

基于视频编辑的方法

**Architectural Specialization vs. Training Optimization.** 值得注意的是，我们的主要基线 **WorldWander** 已经构建在先进的 **Wan2.2-5B** 主干网络之上。尽管拥有如此强大的生成底座，WorldWander 依然引入了显式的 **Collaborative Attention** 和 **Position Encoding** 模块  来强制对齐跨视角几何。这反映了一种“架构优先”的归纳偏置（Inductive Bias），即认为通用 Transformer 结构难以内在地解耦视角变换。

相比之下，我们的 **WanVACE** 基线采用了更为通用的 VACE 编辑范式，保持了预训练主干网络的完整性（Architecture-Agnostic）。

**The Power of Self-Forcing.** 实验结果表明，仅通过标准微调的 WanVACE 虽然继承了 Wan 系列卓越的纹理画质，但在长时序几何稳定性（CTE）上仍逊色于经过架构改良的 WorldWander。然而，一旦引入 **Self-Forcing** 策略，WanVACE 的几何对齐能力得到了显著恢复，在 HAA 和 CSHA 指标上与 WorldWander 的差距被大幅缩小甚至持平。

这一发现具有重要的启示意义：**Ego2Exo 任务的几何挑战不一定需要通过定制化的“手术式”架构修改（如 WorldWander）来解决。通过修正训练过程中的曝光偏差（Self-Forcing），通用的大规模视频生成模型完全有潜力内化这种复杂的几何映射。** 这为未来利用更大参数量（如 14B+）的基础模型解决特定任务提供了一条更具扩展性的路径。



```latex
\begin{table*}[t]
    \centering
    \caption{\textbf{Quantitative Comparison on EgoExoTranslationBench.} We report results across five dimensions: Visual Quality, Temporal Dynamics, Semantics \& Action, Cross-View Correlation, and Camera Control. \textbf{Bold} indicates the best result, and \underline{underline} indicates the second best. The arrow ($\uparrow / \downarrow$) indicates whether higher or lower values are better. $\ddag$ denotes our specialized baseline WorldWander, and $\dag$ denotes the generalist baseline WanVACE.}
    \label{tab:main_results}
    \resizebox{\textwidth}{!}{%
    \begin{tabular}{l|ccc|cc|cc|cc|cc|ccccc}
        \toprule
        \multirow{2}{*}{\textbf{Method}} & \multicolumn{3}{c|}{\textbf{I. Visual Quality}} & \multicolumn{2}{c|}{\textbf{II. Dynamics}} & \multicolumn{2}{c|}{\textbf{III. Semantics}} & \multicolumn{2}{c|}{\textbf{Cross-View}} & \multicolumn{2}{c|}{\textbf{IV. Geometry}} & \multicolumn{5}{c}{\textbf{V. Camera Control}} \\
        & FVD $\downarrow$ & IQ $\uparrow$ & AQ $\uparrow$ & TF $\downarrow$ & DD $\uparrow$ & SF $\uparrow$ & AC $\uparrow$ & HAA $\uparrow$ & SCCR $\uparrow$ & TAA $\uparrow$ & SSDC $\uparrow$ & ADE $\downarrow$ & CCE $\downarrow$ & CTE $\downarrow$ & SCDE $\downarrow$ & CSHA $\downarrow$ \\
        \midrule
        \multicolumn{17}{l}{\textit{Specialized Architecture}} \\
        WorldWander$^\ddag$ & 95.4 & 0.65 & 5.42 & 0.13 & 0.64 & 0.82 & 0.89 & \textbf{0.78} & \underline{0.65} & \textbf{0.82} & \textbf{0.75} & \textbf{25.4} & \textbf{0.05} & \textbf{0.15} & \textbf{0.12} & \textbf{0.08} \\
        \midrule
        \multicolumn{17}{l}{\textit{Generalist Video Editing}} \\
        WanVACE$^\dag$ & \textbf{88.5} & \textbf{0.72} & \textbf{5.80} & \underline{0.12} & \textbf{0.68} & \textbf{0.86} & \underline{0.92} & 0.45 & 0.30 & 0.40 & 0.55 & 85.6 & 0.25 & 0.55 & 0.45 & 0.42 \\
        \midrule
        \multicolumn{17}{l}{\textit{Optimization Strategy (Ours)}} \\
        \textbf{WanVACE + Self-Forcing} & \underline{89.2} & \underline{0.71} & \underline{5.75} & \textbf{0.11} & \underline{0.66} & \underline{0.85} & \textbf{0.93} & \underline{0.75} & \textbf{0.68} & \underline{0.79} & \underline{0.72} & \underline{28.1} & \underline{0.07} & \underline{0.18} & \underline{0.14} & \underline{0.10} \\
        \bottomrule
    \end{tabular}%
    }
    \vspace{-10pt}
\end{table*}
```

