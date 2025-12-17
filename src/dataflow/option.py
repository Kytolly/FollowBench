from dataclasses import dataclass

@dataclass
class Options:
    repo_id:        str  = "Kytolly/examples_Ego2ExoFollowCamera" # HuggingFace 仓库 ID
    assets:         str  = "assets/"                              # 数据集本地根目录
    caption:        str  = "configs/caption.json"                 # 用于训练的 caption
    annotation:     str  = "configs/annotation.json"              # 用于测试的 annotation
    phase:          str  = "test"                                 # train | test
    modal:          str  = "fullymodal"                           # text_only | text_image | fullymodal
    mode:           str  = "easy"                                 # easy | medium | hard       
    clip_len:       int  = 300                                    # 视频片段长度 
    load_size:      int  = 256                                    # 输入图像 resize 大小
    batch_size:     int  = 1                                      # 数据导入批次大小
    serial_batches: bool = True                             
    num_workers:    int  = 4                                    