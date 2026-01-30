ComfyUI 工作流搭建 (Workflow Construction)

回到 ComfyUI 界面，按照以下顺序连接节点：

1. **Loaders (加载器)**
   - `Load Checkpoint`: 选择 `Wan_2.1_1.3B.safetensors`。
   - `Load LoRA`: 加载您训练的 `Ego2Exo_InContext_LoRA.safetensors` (连接到 Checkpoint)。
   - `Load Image`: 加载参考图 (Ref Image)。
   - `Load Video (VHS)`: 加载 Ego 视频。
2. **Preprocessing (预处理)**
   - `Image Resize`: 将 Ref Image 调整为 **240x240** (假设 Latent 高度对应 240p)。
   - `Image Resize`: 将 Ego Video 调整为 **426x240**。
3. **Core Inference (核心推理)**
   - **添加节点**: 双击搜索 `WanVACE In-Context Sampler` (我们需要刚才写的自定义节点)。
   - **连接**:
     - `model`: 来自 Load LoRA。
     - `vae`: 来自 Load Checkpoint。
     - `ref_image`: 来自 Resize 后的参考图。
     - `ego_video`: 来自 Resize 后的视频。
   - **设置参数**:
     - `steps`: 30
     - `cfg`: 7.0
     - `prompt`: `[EGO2EXO] [REF-ID] A cinematic shot...`
     - `window_size`: 16 (用于 Self-forcing)
4. **Output (输出)**
   - `Video Combine (VHS)`: 连接 Sampler 的输出 `exo_video`。设置帧率 60fps。
5. **导出为 API 格式**
   - 在 ComfyUI 设置中开启 "Enable Dev Mode Options"。
   - 点击 "Save (API Format)"。
   - 将生成的 `.json` 文件保存到 `baselines-vdms/workflows/wan_vace_ic_lora.json`。