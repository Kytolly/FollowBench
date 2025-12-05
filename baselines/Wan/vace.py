import os
import logging
import torch
import torch.distributed as dist
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm

# 假设 wan 包的内部结构，请根据实际情况调整 import
from wan.modules.model import WanModel
from wan.modules.vae import WanVAE
from wan.modules.t5 import T5Encoder
from wan.modules.clip import XLMRobertaCLIP
# 假设存在一个基础的 Scheduler 或使用简单的 Euler 步进
# from wan.utils.scheduler import FlowMatchScheduler 

class WanVace:
    def __init__(
        self,
        config,
        checkpoint_dir,
        device_id,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=False,
    ):
        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.rank = rank
        self.t5_cpu = t5_cpu

        logging.info(f"Initializing WanVace Pipeline on device {self.device}")

        # ------------------------------------------------------------------
        # 1. 加载 VAE (手动加载权重，避免使用不存在的辅助函数)
        # ------------------------------------------------------------------
        # 假设 config 中有 vae 的配置参数
        vae_path = os.path.join(checkpoint_dir, "Wan2.1_VAE.pth")
        logging.info(f"Loading VAE from {vae_path}")
        
        # 初始化 VAE 模型结构
        self.vae = WanVAE(
            model_config=config.vae_config # 假设 config 中包含 vae 配置
        ).to(self.device)
        
        # 加载权重
        vae_state_dict = torch.load(vae_path, map_location=self.device)
        self.vae.load_state_dict(vae_state_dict)
        self.vae.eval()
        # VACE 论文提到 VAE 用于 Context Latent Encoding 

        # ------------------------------------------------------------------
        # 2. 加载 Text Encoder (T5)
        # ------------------------------------------------------------------
        t5_path = os.path.join(checkpoint_dir, "models_t5_umt5-xxl-enc-bf16.pth")
        logging.info(f"Loading T5 from {t5_path}")
        
        self.text_encoder = T5Encoder(
            model_name="google/umt5-xxl", # 或者是 config 中的配置
            checkpoint_path=t5_path,
            device="cpu" if t5_cpu else self.device,
            shard=t5_fsdp
        )
        self.text_encoder.eval()

        # ------------------------------------------------------------------
        # 3. 加载 Image Encoder (CLIP) - 用于 VACE 的 Reference Encoding
        # ------------------------------------------------------------------
        clip_path = os.path.join(checkpoint_dir, "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth")
        logging.info(f"Loading CLIP from {clip_path}")
        
        self.image_encoder = CLIPImageEncoder(
            checkpoint_path=clip_path,
            device=self.device
        )
        self.image_encoder.eval()

        # ------------------------------------------------------------------
        # 4. 加载 DiT (WanModel) - VACE 核心
        # ------------------------------------------------------------------
        # VACE 论文 [cite: 4117] 提到 DiT 需要重构以支持 VCU 输入
        dit_path = os.path.join(checkpoint_dir, "Wan2.1_I2V.pth") # 假设使用 I2V 权重作为基础
        logging.info(f"Loading DiT from {dit_path}")
        
        self.model = WanModel(
            config=config.dit_config,
            use_usp=use_usp
        ).to(self.device)
        
        # 处理 FSDP 加载逻辑 (简化版)
        if dit_fsdp:
            # 这里需要具体的 FSDP 加载逻辑，通常涉及 dist.barrier() 等
            pass 
        else:
            dit_state_dict = torch.load(dit_path, map_location=self.device)
            self.model.load_state_dict(dit_state_dict)
        
        self.model.eval()
        self.model.to(torch.bfloat16) # 推荐使用 bf16

    def prepare_source(
        self,
        src_video_paths,
        src_mask_paths,
        src_ref_image_paths,
        frame_num,
        size,
        device
    ):
        """
        构建 Video Condition Unit (VCU) 的组件 
        """
        width, height = size
        
        # --- 处理视频 (Context Frames F) ---
        video_tensor = None
        if src_video_paths and src_video_paths[0]:
            video_path = src_video_paths[0]
            # 读取视频，Resize，归一化到 [-1, 1]
            frames = self._read_video(video_path, frame_num, width, height)
            # [T, H, W, C] -> [1, C, T, H, W]
            video_tensor = torch.tensor(frames).permute(3, 0, 1, 2).unsqueeze(0).float().to(device)
            video_tensor = (video_tensor / 127.5) - 1.0

        # --- 处理掩码 (Mask M) ---
        # VACE 中 Mask 定义了编辑区域 (Reactive) 和保持区域 (Inactive) [cite: 4124]
        mask_tensor = None
        if src_mask_paths and src_mask_paths[0]:
            mask_path = src_mask_paths[0]
            # 读取 Mask (通常是单张图或视频)，二值化
            mask_frames = self._read_mask(mask_path, frame_num, width, height)
            # [T, H, W, 1] -> [1, 1, T, H, W]
            mask_tensor = torch.tensor(mask_frames).permute(3, 0, 1, 2).unsqueeze(0).float().to(device)
            mask_tensor = (mask_tensor > 0.5).float() # 二值化
        else:
            # 如果没有 mask，默认为全 1 (全编辑) 或根据任务需求调整
            if video_tensor is not None:
                 mask_tensor = torch.ones((1, 1, frame_num, height, width), device=device)

        # --- 处理参考图 (Reference R) ---
        ref_tensor = None
        if src_ref_image_paths and src_ref_image_paths[0]:
            # 处理列表中的所有参考图
            ref_list = []
            for p in src_ref_image_paths[0]:
                if p:
                    img = Image.open(p).convert("RGB")
                    # CLIP 预处理通常包括 Resize 和 Norm
                    # 这里简化为 Tensor，具体需匹配 CLIPImageEncoder 的输入要求
                    img_t = torch.tensor(np.array(img)).permute(2, 0, 1).float().to(device) / 255.0
                    ref_list.append(img_t)
            if ref_list:
                ref_tensor = torch.stack(ref_list).unsqueeze(0) # [1, N_ref, C, H, W]

        return video_tensor, mask_tensor, ref_tensor

    def generate(
        self,
        prompt,
        src_video,
        src_mask,
        src_ref_images,
        size,
        frame_num,
        shift,
        sample_solver,
        sampling_steps,
        guide_scale,
        seed,
        offload_model
    ):
        # 设置随机种子
        if seed >= 0:
            torch.manual_seed(seed)

        # 1. 文本编码
        context = self.text_encoder([prompt], device=self.device)
        context_null = self.text_encoder([""], device=self.device) # 用于 CFG

        # 2. 视频与掩码编码 (Context Latent Encoding) 
        latents_context = None
        latents_mask = None
        
        if src_video is not None:
            # VAE Encode: [B, 3, T, H, W] -> [B, 16, t, h, w]
            # WanVAE 会进行时空压缩 (4x8x8)
            with torch.no_grad():
                latents_context = self.vae.encode(src_video)
        
        if src_mask is not None:
            # Mask 需要下采样以匹配 Latent 尺寸
            target_t = (frame_num - 1) // 4 + 1
            target_h = size[1] // 8
            target_w = size[0] // 8
            
            # 简单下采样
            latents_mask = torch.nn.functional.interpolate(
                src_mask[:, 0], size=(target_h, target_w), mode='nearest'
            ).unsqueeze(1)
            # 时间维度下采样 (取每第4帧或平均)
            latents_mask = latents_mask[:, :, ::4, :, :]

        # 3. 参考图编码 (Context Embedder) 
        latents_ref = None
        if src_ref_images is not None:
            latents_ref = self.image_encoder(src_ref_images)

        # 4. 初始化噪声
        noise = torch.randn(
            1, 16, (frame_num - 1) // 4 + 1, size[1] // 8, size[0] // 8,
            device=self.device, dtype=torch.bfloat16
        )
        
        # 5. 采样循环 (Flow Matching)
        latents = noise
        timesteps = np.linspace(1.0, 0.0, sampling_steps + 1)[:-1]
        
        logging.info("Starting Flow Matching sampling...")
        for t in tqdm(timesteps):
            t_tensor = torch.tensor([t], device=self.device, dtype=torch.bfloat16)
            
            # Concept Decoupling Logic 
            # 如果有源视频和掩码，需要将 Latent 混合
            # F_c (Reactive) = F * M (由模型去噪生成)
            # F_k (Inactive) = F * (1-M) (保持原样/加噪混合)
            if latents_context is not None and latents_mask is not None:
                # 构造当前时刻 t 的 noisy context
                noise_ctx = torch.randn_like(latents_context)
                # Flow Matching: x_t = (1-t)*x_1 + t*x_0 (假设 x_1 是数据)
                # 注意 Wan 的具体 FM 公式可能略有不同，需参照 config
                noisy_context_t = (1 - t) * latents_context + t * noise_ctx
                
                # 混合：Mask区域使用生成的latents，非Mask区域使用Noisy Context
                latents = latents * latents_mask + noisy_context_t * (1 - latents_mask)

            # 模型预测 Velocity
            # VACE 的 DiT forward 接口通常包含 y (图像条件) 和 mask
            # 假设 model.forward(x, t, context, y, mask)
            
            # 条件预测
            v_cond = self.model(
                latents, 
                t_tensor, 
                context=context, 
                y=latents_ref,  # 注入参考图特征
                mask=latents_mask # 注入掩码特征 (Context Embedder)
            )
            
            # 无条件预测 (CFG)
            if guide_scale > 1.0:
                v_uncond = self.model(
                    latents, 
                    t_tensor, 
                    context=context_null, 
                    y=torch.zeros_like(latents_ref) if latents_ref is not None else None,
                    mask=latents_mask
                )
                v = v_uncond + guide_scale * (v_cond - v_uncond)
            else:
                v = v_cond

            # Euler 更新 step
            dt = -1.0 / sampling_steps
            latents = latents + v * dt

        # 6. 解码
        logging.info("Decoding video...")
        if offload_model:
            self.model.to("cpu")
            self.vae.to(self.device)
            
        with torch.no_grad():
            video = self.vae.decode(latents)
        
        # 后处理 [-1, 1] -> [0, 1]
        video = ((video + 1) / 2).clamp(0, 1)
        return video

    # --- 内部工具函数 ---
    def _read_video(self, path, frames_num, width, height):
        cap = cv2.VideoCapture(path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = np.linspace(0, total - 1, frames_num, dtype=int)
        frames = []
        for i in range(total):
            ret, frame = cap.read()
            if not ret: break
            if i in indices:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (width, height))
                frames.append(frame)
        cap.release()
        while len(frames) < frames_num:
            frames.append(frames[-1])
        return np.array(frames)

    def _read_mask(self, path, frames_num, width, height):
        # 简化：假设 Mask 是单张图片，扩展到时间轴
        # 如果是视频 Mask，需参考 _read_video 逻辑
        img = Image.open(path).convert("L")
        img = img.resize((width, height), Image.NEAREST)
        arr = np.array(img)
        return np.tile(arr[np.newaxis, ...], (frames_num, 1, 1, 1))