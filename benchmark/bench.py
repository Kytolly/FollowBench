import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from transformers import CLIPProcessor, CLIPModel
from scipy.stats import pearsonr

# 1. 设置文件路径
videos = {
    "GT": "exo1.mp4",
    "Wan-VACE": "tpv.mp4", # 用户上传的文件名为 tpv.mp4 (对应之前VACE结果) 或 tpv-wan-vace.mp4. 检查文件列表，文件名是 tpv-wan-vace.mp4
    "Wan-I2V": "wan_i2v.mp4",
    "SVD-XT": "svd_xt_baseline.mp4",
    "LTX-Video": "ltx_auto_captioned.mp4",
    "CogVideoX": "convideo_tps_output.mp4"
}
# 修正 Wan-VACE 文件名，用户上传列表中有 'tpv.mp4' 和 'tpv-wan-vace.mp4'。
# 根据最后一次上传，Wan-VACE对应 'tpv-wan-vace.mp4'
videos["Wan-VACE"] = "tpv-wan-vace.mp4"

ref_img_path = "ref.jpg"
ego_video_path = "ego1.mp4"

# 2. 初始化模型
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# CLIP for ID and Quality (Similarity to GT)
try:
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
except Exception as e:
    print(f"Failed to load CLIP: {e}")
    clip_model = None

# FasterRCNN for Subject Detection (Viewpoint)
try:
    detection_model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT).to(device)
    detection_model.eval()
except Exception as e:
    print(f"Failed to load Detection Model: {e}")
    detection_model = None

# 3. 辅助函数
def get_video_frames(video_path, max_frames=60):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret or len(frames) >= max_frames:
            break
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    return frames

def calculate_optical_flow_magnitude(frames):
    mags = []
    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_RGB2GRAY)
    for i in range(1, len(frames)):
        curr_gray = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        mags.append(np.mean(mag))
        prev_gray = curr_gray
    return mags

# 4. 指标计算逻辑
results = {}

# 预加载 Reference Features
ref_image = Image.open(ref_img_path)
if clip_model:
    with torch.no_grad():
        inputs = clip_processor(images=ref_image, return_tensors="pt").to(device)
        ref_embed = clip_model.get_image_features(**inputs)
        ref_embed /= ref_embed.norm(dim=-1, keepdim=True)

# 预计算 Ego Video Flow
ego_frames = get_video_frames(ego_video_path)
ego_flow_mags = calculate_optical_flow_magnitude(ego_frames)

# 预加载 GT Video Features (用于 Quality 评估)
gt_frames = get_video_frames(videos["GT"])
gt_embeds = []
if clip_model:
    with torch.no_grad():
        # Batch processing for GT frames to save time/memory, simplified here
        for frame in gt_frames:
            inputs = clip_processor(images=frame, return_tensors="pt").to(device)
            emb = clip_model.get_image_features(**inputs)
            emb /= emb.norm(dim=-1, keepdim=True)
            gt_embeds.append(emb)
    # Stack for easier comparison: [T, D]
    if gt_embeds:
        gt_embeds_tensor = torch.cat(gt_embeds)
        gt_mean_embed = torch.mean(gt_embeds_tensor, dim=0, keepdim=True)

# 遍历所有模型视频进行评估
for name, path in videos.items():
    if name == "GT": continue # Skip GT self-comparison for now
    
    print(f"Evaluating {name}...")
    frames = get_video_frames(path)
    if not frames:
        print(f"Could not read video {path}")
        continue

    # --- Metric 1: Identity Consistency (CLIP Similarity with Ref) ---
    id_scores = []
    quality_scores = []
    if clip_model:
        with torch.no_grad():
            for i, frame in enumerate(frames):
                inputs = clip_processor(images=frame, return_tensors="pt").to(device)
                frame_embed = clip_model.get_image_features(**inputs)
                frame_embed /= frame_embed.norm(dim=-1, keepdim=True)
                
                # ID Score
                sim = (frame_embed @ ref_embed.T).item()
                id_scores.append(sim)
                
                # Quality Score (Similarity to GT Mean Embedding - simplified FVD proxy)
                # Simulating "how close is the semantic content to the real 3rd person video"
                if gt_embeds:
                    q_sim = (frame_embed @ gt_mean_embed.T).item()
                    quality_scores.append(q_sim)
    
    avg_id_score = np.mean(id_scores) if id_scores else 0
    avg_quality_score = np.mean(quality_scores) if quality_scores else 0

    # --- Metric 2: Motion Alignment (Correlation with Ego Flow) ---
    gen_flow_mags = calculate_optical_flow_magnitude(frames)
    
    # 调整长度以匹配 (截断较长的)
    min_len = min(len(ego_flow_mags), len(gen_flow_mags))
    if min_len > 1:
        corr, _ = pearsonr(ego_flow_mags[:min_len], gen_flow_mags[:min_len])
        motion_score = max(0, corr) # 只取正相关，负相关也视为不对齐
    else:
        motion_score = 0

    # --- Metric 3: Viewpoint Validity (Person Detection Rate) ---
    detected_count = 0
    if detection_model:
        # Sample frames to save time (every 5th frame)
        sample_indices = range(0, len(frames), 5)
        for i in sample_indices:
            frame = frames[i]
            # Transform for model
            img_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
            img_tensor = img_tensor.to(device).unsqueeze(0)
            
            with torch.no_grad():
                prediction = detection_model(img_tensor)[0]
            
            # Class 1 is 'person' in COCO
            # Filter by score threshold
            persons = [s for l, s in zip(prediction['labels'], prediction['scores']) if l == 1 and s > 0.7]
            if len(persons) > 0:
                detected_count += 1
        
        viewpoint_score = detected_count / len(sample_indices) if len(sample_indices) > 0 else 0
    else:
        viewpoint_score = 0

    results[name] = {
        "ID Consistency": avg_id_score,
        "Motion Alignment": motion_score,
        "Viewpoint Validity": viewpoint_score,
        "Visual Quality": avg_quality_score
    }

print("Evaluation Results:", results)

# 5. 绘制雷达图
categories = ["Visual Quality", "ID Consistency", "Viewpoint Validity", "Motion Alignment"]
N = len(categories)

angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1] # Close the loop

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

# Helper to normalize data for plotting (scale distinct ranges to 0-1 relative to best/worst)
# For simplicity in this demo, we plot raw scores but scaled visually if needed.
# CLIP scores are usually 0.2-0.3 range, Detection is 0-1, Correlation is 0-1.
# We will normalize each metric across models to 0-1 range for the chart to be readable.

normalized_results = {}
for metric in categories:
    values = [results[m][metric] for m in results]
    min_v = min(values)
    max_v = max(values)
    if max_v - min_v == 0:
        for m in results:
            if m not in normalized_results: normalized_results[m] = {}
            normalized_results[m][metric] = 0.5 # Default middle if no variance
    else:
        for m in results:
            if m not in normalized_results: normalized_results[m] = {}
            normalized_results[m][metric] = (results[m][metric] - min_v) / (max_v - min_v)

# Plot each model
colors = ['b', 'g', 'r', 'c', 'm']
for i, (name, metrics) in enumerate(normalized_results.items()):
    values = [metrics[cat] for cat in categories]
    values += values[:1]
    ax.plot(angles, values, linewidth=2, linestyle='solid', label=name, color=colors[i % len(colors)])
    ax.fill(angles, values, color=colors[i % len(colors)], alpha=0.1)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories)
plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
plt.title("Ego-to-Exo Video Generation Benchmark (Normalized)")
plt.savefig("radar_chart.png")