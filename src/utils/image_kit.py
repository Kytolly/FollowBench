import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

def load_image_to_gpu(image_path, device='cuda', target_size=None):
    """读取图片并直接转换为 Tensor [C, H, W]"""
    try:
        img = Image.open(image_path).convert('RGB')
        if target_size is not None:
            img = img.resize(target_size, Image.BILINEAR)
        tensor = transforms.ToTensor()(img).to(device)
        return tensor
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

def prepare_ref_embedding(dinov2_model, dino_transform, ref_img: Image, device='cpu'):
    """
    预计算参考图的 Embedding
    """
    try:
        ref_input = dino_transform(ref_img).unsqueeze(0).to(device)
        with torch.no_grad():
            ref_emb = dinov2_model(ref_input)
        return ref_emb
    except Exception as e:
        print(f"Error loading Ref image: {e}")
        return None
    
def get_person_embedding_from_tensor(full_frame_tensor, box, dinov2_model, dino_transform):
    """
    直接在 GPU Tensor 上裁剪并提取特征
    """
    # box: [x1, y1, x2, y2]
    c, h, w = full_frame_tensor.shape
    x1, y1, x2, y2 = map(int, box)
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(w, x2); y2 = min(h, y2)
    if x2 <= x1 + 5 or y2 <= y1 + 5:
        return None
    
    # full_frame_tensor: [C, H, W] on GPU
    person_crop = full_frame_tensor[:, y1:y2, x1:x2]
    try:
        processed_crop = dino_transform(person_crop)
        input_batch = processed_crop.unsqueeze(0) # [1, C, 224, 224]
        with torch.no_grad():
            embedding = dinov2_model(input_batch)
        return embedding
    except Exception as e:
        # Fallback for transforms that only accept PIL
        print(f"Warning: Tensor transform failed, trying PIL fallback: {e}")
        return None

def get_pose_vectors(image, model, device):
    # 定义 COCO 格式的肢体连接
    # 格式: (start_point_index, end_point_index)
    # COCO Keypoints: 0:nose, 1:l_eye, 2:r_eye, 3:l_ear, 4:r_ear, 5:l_shoulder, 6:r_shoulder, 
    # 7:l_elbow, 8:r_elbow, 9:l_wrist, 10:r_wrist, 11:l_hip, 12:r_hip, 13:l_knee, 14:r_knee, 15:l_ankle, 16:r_ankle
    limbs_id = [
        (5, 7), (7, 9),   # Left Arm
        (6, 8), (8, 10),  # Right Arm
        (11, 13), (13, 15), # Left Leg
        (12, 14), (14, 16), # Right Leg
        (5, 6), (11, 12), # Shoulders & Hips
        (5, 11), (6, 12)  # Torso
    ]
    # 预处理：转 Tensor [C, H, W] 并归一化到 [0, 1]
    img_tensor = transforms.ToTensor()(image).to(device)
    
    with torch.no_grad():
        output = model([img_tensor])[0]
    
    # 筛选置信度最高的人
    valid_mask = output['scores'] > 0.7
    if not valid_mask.any():
        return None
    
    # 取最高分的那个人
    idx = torch.argmax(output['scores'])
    keypoints = output['keypoints'][idx] # [17, 3] (x, y, visibility)
    
    # 提取肢体向量
    vectors = []
    for (start, end) in limbs_id:
        if keypoints[start, 2] > 0 and keypoints[end, 2] > 0:
            v = keypoints[end, :2] - keypoints[start, :2] # [x, y]
            norm = torch.norm(v)
            if norm > 1e-6:
                v = v / norm
                vectors.append(v)
            else:
                vectors.append(torch.zeros(2, device=device)) # 重合点
        else:
            vectors.append(torch.zeros(2, device=device)) # 不可见肢体，填0
    
    if not vectors:
        return None
        
    return torch.stack(vectors) # [Num_Limbs, 2]