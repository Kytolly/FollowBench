这是一个按照 CVPR/ICCV/SIGGRAPH 等顶会开源项目标准重构的 `README.md` 模板。

**核心设计理念：**

1. **First Impression (前3秒原则)**：开头必须是高质量的 GIF/视频展示 (Teaser)，一目了然地展示“输入是什么，输出是什么”。
2. **Usability (易用性)**：将环境配置、推理、评测的命令极简化（One-line command）。
3. **Methodology (技术深度)**：用一张清晰的架构图展示 Spatial Concatenation 和 IC-LoRA 的逻辑，体现学术价值。
4. **Structure (结构化)**：将复杂的比赛提交、Docker 教程折叠或分流，保持主页整洁。

以下是为您整理的文档结构（通常顶会项目使用英文 README，我在关键位置用中文备注了 **[TODO]** 指导您填充素材）。

---

# README.md 模板内容

```markdown
# Ego2ExoFollowShot: Transforming First-Person Chaos into Third-Person Cinematic

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2503.xxxxx-b31b1b.svg)](https://arxiv.org/abs/2503.xxxxx)
[![Hugging Face Space](https://img.shields.io/badge/🤗%20Hugging%20Face-Space-yellow.svg)](https://huggingface.co/spaces/YourName/Ego2Exo)
[![Project Page](https://img.shields.io/badge/Project-Website-blue)](https://your-project-page.github.io)
[![License](https://img.shields.io/badge/License-Apache%202.0-green)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)

</div>

---

<div align="center">

<img src="assets/teaser.gif" width="800px">

**Stabilizing the Unstable.** *We introduce **Ego2ExoFollowShot**, a benchmark and baseline model that leverages In-Context Diffusion Transformers to generate consistent third-person follow shots conditioned on ego-centric videos.*

[**Demo Video**](https://youtube.com/your-video-link) | [**Leaderboard**](https://huggingface.co/spaces/YourName/Ego2Exo/leaderboard) | [**Dataset**](https://huggingface.co/datasets/YourName/Ego2ExoDataset)

</div>

---

## 📰 News

* **[2025.04.20]** 🚀 Release the inference code and pretrained weights for **EgoGen-V1**.
* **[2025.04.15]** 🏆 **Ego2ExoFollowShot Competition** is now live! Check the [Submission Guide](#-submission--competition).
* **[2025.03.10]** 📄 Paper uploaded to arXiv.

## 💡 Introduction

<div align="center">
  <img src="assets/method_diagram.png" width="800px">
</div>

**Ego2ExoFollowShot** tackles the challenge of generating stable, third-person views from shaky first-person footage. Unlike traditional video generation, our approach:
1.  **Spatial Concatenation**: Concatenates Ego and Exo latents spatially, using the Ego video as a strong visual prompt.
2.  **In-Context LoRA (IC-LoRA)**: Adopts a tuning-free or lightweight adapter approach to align motion and identity.
3.  **Benchmark**: Provides a standardized testbed with metrics for **Control** (HAA, CCE), **Consistency** (AC, BSC), and **Quality** (FVD).

## 🚀 Quick Start

### 1. Installation

We provide two ways to set up the environment: **Conda** (Recommended for dev) and **Docker** (Recommended for evaluation).

#### Option A: Conda
```bash
git clone [https://github.com/YourUsername/Ego2ExoFollowShot.git](https://github.com/YourUsername/Ego2ExoFollowShot.git)
cd Ego2ExoFollowShot
pip install -e .
# Install Flash Attention (Optional but recommended)
pip install flash-attn --no-build-isolation

```

#### Option B: Docker

```bash
docker-compose build
docker-compose up -d
docker-compose exec ego2exo bash

```

### 2. Download Weights & Data

We provide a script to automatically download the **Wan-based Backbone**, **IC-LoRA Adapters**, and the **Benchmark Dataset**.

```bash
ego2exo setup --download-weights --download-data

```

### 3. Inference (Demo)

Generate a follow shot from your own ego-video:

```bash
python scripts/inference.py \
    --ego_video assets/examples/hiking_ego.mp4 \
    --ref_image assets/examples/hiker_ref.png \
    --prompt "A man hiking in the mountains, cinematic lighting" \
    --output result.mp4

```

## 📊 Evaluation & Benchmark

We propose a comprehensive evaluation protocol focusing on **Quality**, **Control**, and **Consistency**.

| Metric | Full Name | Description |
| --- | --- | --- |
| **FVD** | Frechet Video Distance | Measures the realism and temporal coherence. |
| **HAA** | Human Action Alignment | Calculates the overlap of OpenPose skeletons between Ego and Exo. |
| **CCE** | Camera Centering Error | Measures how well the subject is centered (Exo-view stability). |
| **AC** | Appearance Consistency | Identity preservation score (ReID/CLIP features). |

### Run Evaluation

Evaluate your model on the validation set with a single command:

```bash
# 1. Generate videos (save to ./submission folder)
# 2. Run evaluation
ego2exo evaluate \
    --submission ./submission_folder/ \
    --dataroot ./assets/Ego2ExoDataset \
    --device cuda:0

```

## 🏆 Submission & Competition

We host a competition to push the boundaries of Ego2Exo generation.

**Submission Format (JSON Index Recommended):**

```json
{
    "meta": { "team_name": "MyTeam", "model_name": "EgoGen-V1" },
    "results": {
        "1001": "videos/1001_exo.mp4",
        "1002": "videos/1002_exo.mp4"
    }
}

```

*For detailed rules and folder structure, please refer to [COMPETITION.md](https://www.google.com/search?q=docs/COMPETITION.md).*

## 🧩 Model Zoo

| Model | Resolution | Frames | Checkpoint |
| --- | --- | --- | --- |
| **EgoGen-Base** | 480x480 | 16 | [Download](https://www.google.com/search?q=) |
| **EgoGen-XL** | 720x720 | 32 | [Download](https://www.google.com/search?q=) |

## 🗓️ Roadmap

* [ ] **Trajectory Alignment (TA)**: Implement ADE metric for trajectory comparison.
* [ ] **WanVideoDataset**: Support dense captioning pipeline.
* [x] **Evaluation Pipeline**: Support FVD, HAA, CCE metrics.
* [x] **Docker**: Full environment support.

## 🖊️ Citation

If you find this project useful, please cite our work:

```bibtex
@article{YourName2025Ego2Exo,
  title={Ego2ExoFollowShot: Stabilizing First-Person Videos into Third-Person Follow Shots},
  author={Your Name and Collaborators},
  journal={arXiv preprint arXiv:2503.xxxxx},
  year={2025}
}

```

## 🤝 Acknowledgement

This project is built upon [Wan](https://www.google.com/search?q=https://github.com/Wan-Video/Wan-Video) and utilizes ideas from [In-Context LoRA](https://arxiv.org/abs/2410.23775). We thank the community for their contributions.


