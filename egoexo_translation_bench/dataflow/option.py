"""Runtime options dataclass used across dataflow modules.

Default values are read from the global CONFIG and provide dataset and runtime
settings used by dataset and loader utilities.
"""

from dataclasses import dataclass
from ..configs import CONFIG

@dataclass
class Options:
    """Runtime configuration options.

    Attributes:
        repo_id: HuggingFace repository id for the dataset.
        assets: Local assets directory.
        phase: One of 'train' or 'test'.
        modal: Model/modal setting used by apps.
        mode: Difficulty mode string.
        clip_len: Number of frames per clip.
        height: Frame height in pixels.
        width: Frame width in pixels.
        batch_size: Batch size for data loaders.
        serial_batches: If true, disables shuffling.
        num_workers: Number of worker processes for data loading.
    """
    # 基础信息
    repo_id:        str  = "Kytolly/examples_Ego2ExoFollowCamera" # HuggingFace 仓库 ID
    
    # 路径配置
    assets:         str  = "assets/"                              # 数据集本地根目录
    # annotation:     str  = "assets/test/annotation.json"          # 用于训练/测试的 annotation
    
    # 运行模式
    phase:          str  = "test"                                 # train | test
    modal:          str  = "vace_instruct"                        # t2v_generic | i2v_generic | vace_instruct
    mode:           str  = "easy"                                 # easy | medium | hard       
    
    # 规则配置 (从 CONFIG 读取默认值)
    # OmegaConf 允许使用点号访问属性 (CONFIG.rules.clip_len)
    clip_len:       int  = CONFIG.rules.clip_len            
    height:         int  = CONFIG.rules.resolution_height   
    width:          int  = CONFIG.rules.resolution_width
    
    # Dataloader 配置
    batch_size:     int  = 1                                      
    serial_batches: bool = True                             
    num_workers:    int  = 4