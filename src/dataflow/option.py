from dataclasses import dataclass

@dataclass
class Options:
    hf_repo_id = "Kytolly/examples_Ego2ExoFollowCamera" # HuggingFace 仓库 ID
    assets = "assets/"                                # 数据集本地根目录
    caption = 'configs/caption.json'                    # 用于训练的 caption
    annotation = "configs/annotation.json"              # 用于测试的 annotation
    phase = "test"                                       # train | test
    mode = "fullymodal"                          # text_only | text_image | fullymodal
    clip_len = 300                                      # 视频片段长度    
    load_size = 256                                     # 输入图像 resize 大小
    batch_size = 1                                      # 数据导入批次大小
    serial_batches = True
    num_workers = 4