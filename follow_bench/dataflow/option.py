"""Runtime options dataclass used across dataflow modules.

Default values are read from the global CONFIG and provide dataset and runtime
settings used by dataset and loader utilities.
"""

from dataclasses import dataclass

@dataclass
class Options:
    """Runtime configuration options.
    """
    # 基础信息
    repo_id:        str  = "Kytolly/FollowBench" # HuggingFace 仓库 ID
    
    # 路径配置
    assets:         str  = "assets/followbench"                              # 数据集本地根目录
    
    # 运行模式
    phase:          str  = "test_unseen"                          # train | test_unseen | test_seen
    
    # 视频配置
    height:         int = 704
    width:          int  = 1280
    num_frames:     int  = 149
        
    # Dataloader 配置
    batch_size:     int  = 1                                      
    serial_batches: bool = True                             
    num_workers:    int  = 4