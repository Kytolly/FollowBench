import subprocess
import sys
import os
import logging
import re
import torch
# import torch.nn.functional as F
# from torchvision import transforms
from torchvision.ops import roi_align
from utils import *

def FrechetVideoDistance(
    repo_path,
    real_videos_dir, 
    gen_videos_dir,
    mirror,
    gpus,
    resolution,
    metrics,):
    '''计算 FVD 需要视频矩阵'''
    if not repo_path or not os.path.exists(repo_path):
        logging.error(f"StyleGAN-V repo path not found: {repo_path}")
        return None
    script_path = os.path.join(repo_path, 'src', 'scripts', 'calc_metrics_for_dataset.py')
    if not os.path.exists(script_path):
        logging.error(f"Script not found at: {script_path}")
        return None
    
    cmd = [
        sys.executable, script_path,
        '--real_data_path', real_videos_dir,
        '--fake_data_path', gen_videos_dir,
        '--mirror', mirror,
        '--gpus', gpus,
        '--resolution', resolution,
        '--metrics', metrics,
    ]
    logging.info(f"Executing FVD calculation: {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd, 
            cwd=repo_path, 
            capture_output=True, 
            text=True, 
            env=os.environ.copy(),
            check=True
        )
        output_lines = result.stdout.split('\n')
        fvd_score = None
        for line in output_lines:
            if metrics in line:
                parts = line.split()
                matches = re.findall(r"[-+]?\d*\.\d+|\d+", line)
                if matches:
                    fvd_score = float(matches[-1])
        
        logging.info(f"FVD Calculation Success. Output: {fvd_score}")
        return fvd_score

    except subprocess.CalledProcessError as e:
        logging.error(f"FVD Calculation Failed with error:\n{e.stderr}")
        return None

def calculate_temporal_consistency(frames):
    '''Temporal Flickering (闪烁) 和 Motion Smoothness (平滑度) (越低越好)'''
    '''附带计算 Dynamic Degree'''
    if len(frames) < 3:
        return 0.0, 0.0
    
    warp_errors = []       # Flickering
    flow_diffs = []        # Smoothness
    flow_magnitudes = []   # Dynamic
    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_RGB2GRAY)
    h, w = prev_gray.shape
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    grid_x = grid_x.astype(np.float32)
    grid_y = grid_y.astype(np.float32)
    prev_flow = None
    
    for i in range(1, len(frames)):
        curr_gray = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None, 
            0.5, 3, 15, 3, 5, 1.2, 0
        )
        # --- Dynamic Degree ---
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        avg_mag = np.mean(mag)
        flow_magnitudes.append(avg_mag)
        
        map_x = grid_x + flow[..., 0]
        map_y = grid_y + flow[..., 1]
        warped_prev = cv2.remap(frames[i-1], map_x, map_y, interpolation=cv2.INTER_LINEAR)
        diff = cv2.absdiff(frames[i], warped_prev)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
        border = 5
        
        valid_mask = np.ones_like(diff_gray, dtype=bool)
        valid_mask[:border, :] = False
        valid_mask[-border:, :] = False
        valid_mask[:, :border] = False
        valid_mask[:, -border:] = False
        
        frame_error = np.mean(diff_gray[valid_mask])
        warp_errors.append(frame_error)
        
        # --- Motion Smoothness ---
        if prev_flow is not None:
            flow_diff = np.sqrt(np.sum((flow - prev_flow)**2, axis=2))
            avg_accel = np.mean(flow_diff[valid_mask])
            flow_diffs.append(avg_accel)
        prev_gray = curr_gray
        prev_flow = flow

    flickering = np.mean(warp_errors) if warp_errors else 0.0
    smoothness = np.mean(flow_diffs) if flow_diffs else 0.0
    dynamic_degree = np.mean(flow_magnitudes) if flow_magnitudes else 0.0
    
    return flickering, smoothness, dynamic_degree

def AppearanceConsistency(dinov2_model, dino_transform, detection_model, ref_emb, video_tensor):
    '''计算外观一致性 (Re-ID Score)'''
    device = dinov2_model.device
    if ref_emb is None: return 0.0
    scores = []
    step = 5 # 采样间隔
    detection_model.eval()
    
    for i in range(0, len(video_tensor), step):
        frame_tensor = video_tensor[i] # [C, H, W]
        det_input = frame_tensor.unsqueeze(0) # [1, C, H, W]
        with torch.no_grad():
            prediction = detection_model(det_input)[0]
        
        # 筛选置信度最高的人
        best_box = None
        max_score = 0
        # 找到所有 label==1 (person) 且 score > 0.7 的索引
        valid_mask = (prediction['labels'] == 1) & (prediction['scores'] > 0.7)
        if valid_mask.any():
            # 找到最高分的索引
            valid_scores = prediction['scores'][valid_mask]
            valid_boxes = prediction['boxes'][valid_mask]
            
            best_idx = torch.argmax(valid_scores)
            best_box = valid_boxes[best_idx] # [x1, y1, x2, y2]
            
            # 构建 RoI 格式
            batch_index = torch.zeros((1, 1), device=device)
            roi_box = torch.cat([batch_index, best_box.view(1, 4)], dim=1) # [1, 5]
            
            # 在 GPU 上做 Crop + Resize
            person_crop = roi_align(
                input=det_input, # [1, C, H, W]
                boxes=roi_box,
                output_size=(224, 224),
                spatial_scale=1.0, # 因为我们在原图上操作
                aligned=True # 提高精度
            ) # [1, C, 224, 224]
            
            # Normalize + DINOv2 推理
            person_crop = dino_transform(person_crop)
            with torch.no_grad():
                frame_emb = dinov2_model(person_crop)
            
            # 计算相似度
            sim = torch.nn.functional.cosine_similarity(ref_emb, frame_emb).item()
            scores.append(sim)

    if not scores:
        return 0.0
        
    return sum(scores) / len(scores)

def CameraCenteringError(best_box, img_h, img_w):
    """计算 Camera Centering Error"""
    # Box 中心
    cx = (best_box[0] + best_box[2]) / 2.0; cy = (best_box[1] + best_box[3]) / 2.0
    
    # 画面中心
    img_cx = img_w / 2.0; img_cy = img_h / 2.0
    
    # 计算欧氏距离
    dist = torch.sqrt((cx - img_cx)**2 + (cy - img_cy)**2)
    
    # 归一化 (除以中心到角落的距离)
    max_dist = torch.sqrt(torch.tensor(img_cx**2 + img_cy**2, device=best_box.device))
    
    error = dist / max_dist
    return error.item()

def calculate_subject_metrics(dinov2_model, dino_transform, detection_model, ref_emb, video_tensor):
    device = dinov2_model.device
    if ref_emb is None: return 0.0
    app_scores = []      # 外观一致性分数列表
    center_errors = []   # 中心误差列表
    detected_count = 0   # 检测到的帧数
    total_samples = 0    # 总采样帧数
    step = 5 # 采样间隔
    detection_model.eval()
    _, _, H, W = video_tensor.shape
    
    for i in range(0, len(video_tensor), step):
        total_samples += 1
        frame_tensor = video_tensor[i] # [C, H, W]
        det_input = frame_tensor.unsqueeze(0) # [1, C, H, W]
        with torch.no_grad():
            prediction = detection_model(det_input)[0]
        
        # 筛选置信度最高的人
        best_box = None
        max_score = 0
        # 找到所有 label==1 (person) 且 score > 0.7 的索引
        valid_mask = (prediction['labels'] == 1) & (prediction['scores'] > 0.7)
        if valid_mask.any():
            detected_count += 1
            # 找到最高分的索引
            valid_scores = prediction['scores'][valid_mask]
            valid_boxes = prediction['boxes'][valid_mask]
            
            best_idx = torch.argmax(valid_scores)
            best_box = valid_boxes[best_idx] # [x1, y1, x2, y2]
            
            # --- Camera Centering Error ---
            c_err = CameraCenteringError(best_box, H, W)
            center_errors.append(c_err)
            
            # --- Appearance Consistency ---
            # 构建 RoI 格式
            batch_index = torch.zeros((1, 1), device=device)
            roi_box = torch.cat([batch_index, best_box.view(1, 4)], dim=1) # [1, 5]
            
            # 在 GPU 上做 Crop + Resize
            person_crop = roi_align(
                input=det_input, # [1, C, H, W]
                boxes=roi_box,
                output_size=(224, 224),
                spatial_scale=1.0, # 因为我们在原图上操作
                aligned=True # 提高精度
            ) # [1, C, 224, 224]
            
            # Normalize + DINOv2 推理
            person_crop = dino_transform(person_crop)
            with torch.no_grad():
                frame_emb = dinov2_model(person_crop)
            
            # 计算相似度
            sim = torch.nn.functional.cosine_similarity(ref_emb, frame_emb).item()
            app_scores.append(sim)
        else: # 如果没检测到人：
            app_scores.append(0.0)
            center_errors.append(1.0)
            

    avg_app_score = sum(app_scores) / len(app_scores) if app_scores else 0.0
    avg_center_error = sum(center_errors) / len(center_errors) if center_errors else 1.0
    validity_score = detected_count / total_samples if total_samples > 0 else 0.0
    return avg_app_score, avg_center_error, validity_score

def BackgroundSemanticConsistency(
    clip_model, 
    clip_processor, 
    detection_model, 
    ref_image, 
    video_tensor):
    '''
    计算 Background Semantic Consistency
    逻辑：检测人 -> Mask掉人物区域 -> 计算剩余背景与参考图背景的CLIP特征相似度
    '''
    device = clip_model.device
    detection_model.eval()
    clip_model.eval()
    
    scores = []
    step = 5  # 采样间隔，为了提高计算速度
    ref_tensor = transforms.ToTensor()(ref_image).to(device).unsqueeze(0) # [1, C, H, W]
    
    with torch.no_grad():
        ref_pred = detection_model(ref_tensor)[0]
        ref_mask = torch.ones_like(ref_tensor)
        
        # 寻找参考图中的人并将其 Mask 掉 (置为黑)
        valid_mask = (ref_pred['labels'] == 1) & (ref_pred['scores'] > 0.7)
        if valid_mask.any():
            best_idx = torch.argmax(ref_pred['scores'][valid_mask])
            box = ref_pred['boxes'][valid_mask][best_idx].int()
            
            # 边界保护
            H, W = ref_tensor.shape[2], ref_tensor.shape[3]
            x1, y1, x2, y2 = max(0, box[0]), max(0, box[1]), min(W, box[2]), min(H, box[3])
            ref_mask[:, :, y1:y2, x1:x2] = 0.0
        
        masked_ref = ref_tensor * ref_mask
        
        # 转换回 CLIP Processor 接受的格式 (PIL) 并提取特征
        inputs_ref = clip_processor(images=transforms.ToPILImage()(masked_ref.squeeze().cpu()), return_tensors="pt").to(device)
        ref_bg_emb = clip_model.get_image_features(**inputs_ref)
        ref_bg_emb /= ref_bg_emb.norm(dim=-1, keepdim=True)

    # 遍历生成视频的帧
    for i in range(0, len(video_tensor), step):
        frame_tensor = video_tensor[i].unsqueeze(0) # [1, C, H, W]
        
        with torch.no_grad():
            prediction = detection_model(frame_tensor)[0]
            frame_mask = torch.ones_like(frame_tensor)
            valid_mask = (prediction['labels'] == 1) & (prediction['scores'] > 0.7)
            if valid_mask.any():
                best_idx = torch.argmax(prediction['scores'][valid_mask])
                box = prediction['boxes'][valid_mask][best_idx].int()
                H, W = frame_tensor.shape[2], frame_tensor.shape[3]
                x1, y1, x2, y2 = max(0, box[0]), max(0, box[1]), min(W, box[2]), min(H, box[3])
                frame_mask[:, :, y1:y2, x1:x2] = 0.0 # 将检测到的人物区域置为 0 (黑色)
            
            masked_frame = frame_tensor * frame_mask
            
            # CLIP 编码当前帧背景
            inputs_frame = clip_processor(images=transforms.ToPILImage()(masked_frame.squeeze().cpu()), return_tensors="pt").to(device)
            frame_bg_emb = clip_model.get_image_features(**inputs_frame)
            frame_bg_emb /= frame_bg_emb.norm(dim=-1, keepdim=True)
            
            # 计算 Cosine Similarity
            sim = torch.nn.functional.cosine_similarity(ref_bg_emb, frame_bg_emb).item()
            scores.append(sim)
    if not scores:
        return 0.0
    return sum(scores) / len(scores)

def OpticalFlowCorrelation(gen_frames, ref_frames):
    '''
    计算光流相关性 (Optical Flow Correlation)
    衡量生成视频的运动趋势(主要是相机运动)是否与参考视频同步。
    '''
    min_len = min(len(gen_frames), len(ref_frames))
    if min_len < 2:
        return 0.0
    gen_f = gen_frames[:min_len]
    ref_f = ref_frames[:min_len]

    # 分别提取生成视频和参考视频的运动序列
    gen_dx, gen_dy = get_motion_series(gen_f)
    ref_dx, ref_dy = get_motion_series(ref_f)
    
    # 计算皮尔逊相关系数 平滑化防止除 0 报错
    if np.std(gen_dx) < 1e-6 or np.std(ref_dx) < 1e-6:
        corr_x = 0.0
    else:
        corr_x, _ = pearsonr(gen_dx, ref_dx)
        
    if np.std(gen_dy) < 1e-6 or np.std(ref_dy) < 1e-6:
        corr_y = 0.0
    else:
        corr_y, _ = pearsonr(gen_dy, ref_dy)

    return (corr_x + corr_y) / 2.0

def HumanActionAlignment(gen_frames, ref_frames, model, device):
    '''
    计算人物动作对齐 (Human Action Alignment)
    逻辑：提取骨骼关键点 -> 构建肢体向量 -> 计算向量余弦相似度
    '''
    min_len = min(len(gen_frames), len(ref_frames))
    scores = []
    for i in range(0, min_len, 2): # 步长为2，加速计算
        vec_gen = get_pose_vectors(gen_frames[i], model)
        vec_ref = get_pose_vectors(ref_frames[i], model)
        
        if vec_gen is not None and vec_ref is not None:
            # 计算余弦相似度 (Cosine Similarity)
            cos_sim = (vec_gen * vec_ref).sum(dim=1) # [Num_Limbs]
            
            # 只统计非零向量 (即双方都检测到的肢体)
            valid_limbs = (cos_sim != 0)
            if valid_limbs.any():
                frame_score = cos_sim[valid_limbs].mean().item()
                scores.append(frame_score)
    if not scores:
        return 0.0
    return sum(scores) / len(scores)

def TrajectoryAlignment(frames, device):
    pass