from dataclasses import dataclass
from ..configs import CONFIG

@dataclass
class Options:
    repo_id:        str  = "Kytolly/examples_Ego2ExoFollowCamera" # HuggingFace 仓库 ID
    assets:         str  = "assets/"                              # 数据集本地根目录
    caption:        str  = "assets/train/caption.json"            # 用于训练的 caption
    annotation:     str  = "assets/test/annotation.json"          # 用于测试的 annotation
    phase:          str  = "test"                                 # train | test
    modal:          str  = "vace_instruct"                        # t2v_generic | i2v_generic | vace_instruct | other
    mode:           str  = "easy"                                 # easy | medium | hard       
    clip_len:       int  = CONFIG['rules']['clip_len']            # 视频片段长度 
    height:         int  = CONFIG['rules']['resolution_height']   # 输入图像 resize 大小
    width:          int  = CONFIG['rules']['resolution_width']
    batch_size:     int  = 1                                      # 数据导入批次大小
    serial_batches: bool = True                             
    num_workers:    int  = 4                                    