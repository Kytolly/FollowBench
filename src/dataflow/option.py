from dataclasses import dataclass

# 模拟配置对象 (通常来自 argparser)
@dataclass
class TestOptions:
    dataroot = "./dataset" # 你的数据集根目录
    json_path = "./dataset/test/test_annotation.json" # 指向你的测试 JSON
    hf_repo_id = "Kytolly/examples_Ego2ExoFollowCamera" # 你的 HF Dataset ID
    phase = "test"
    prompt_mode = "fullymodal" # 可以在这里切换 easy/medium/hard 对应的 prompt 模式
    clip_len = 16
    load_size = 256
    batch_size = 1
    serial_batches = True
    num_workers = 4

# 1. 初始化 Loader
opt = TestOptions()
data_loader = Ego2ExoDataLoader(opt)
dataset = data_loader.load_data()

# 2. 遍历数据进行 Benchmark
for i, data in enumerate(dataset):
    print(f"Processing ID: {data['video_id']}")
    
    # 获取数据
    ego_video = data['ego_video'].cuda() # [B, T, C, H, W]
    ref_img = data['ref_image'].cuda()   # [B, C, H, W]
    prompt = data['pos_prompt']          # List of strings
    
    # 3. 运行模型 (假设 model 是你的生成模型)
    # generated_video = model(ego_video, ref_img, prompt)
    
    # 4. 计算指标 (使用之前设计的 Evaluators)
    # score = evaluator.compute(generated_video, ...)