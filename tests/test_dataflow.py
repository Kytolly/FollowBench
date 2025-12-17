from tqdm import tqdm
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torch.utils.data import DataLoader

from src.dataflow.set import BenchmarkDataset
from src.dataflow.option import Options

from scripts.env import get_env_config
cfg = get_env_config('test')
def test_loading():
    print(f"Checking assets at: {cfg['path_assets']}, repo_id: {cfg['repo_id']}")
    try:
        opt = Options(repo_id=cfg['repo_id'], assets=cfg['path_assets'])
        dataset = BenchmarkDataset(opt) 
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
        print(f"📂 Dataset initialized. Total samples: {len(dataset)}")
        print("🚀 Starting full dataset traversal...")
        
        for i, batch in enumerate(tqdm(loader, desc="Checking data")):
            if i < 3: 
                print(f"\n--- Sample {i} Info ---")
                if isinstance(batch, dict):
                    for key, value in batch.items():
                        if isinstance(value, torch.Tensor):
                            print(f"  Key: '{key}' | Shape: {value.shape} | Type: {value.dtype}")
                        else:
                            print(f"  Key: '{key}' | Value: {value}")

                elif isinstance(batch, (list, tuple)):
                    for idx, item in enumerate(batch):
                        if isinstance(item, torch.Tensor):
                            print(f"  Item {idx} | Shape: {item.shape}")
                        else:
                            print(f"  Item {idx} | Value: {item}")
                else:
                    print(f"  Data: {batch}")

        print("\n✅ Success! All data samples loaded without errors.")
        
    except Exception as e:
        # 4. 捕获并定位错误
        # 'i' 变量会保留在当前作用域，告诉你是第几个样本出错了
        fail_index = i if 'i' in locals() else 'Initialization'
        print(f"\n❌ Error encountered at index {fail_index}: {e}")
        import traceback
        traceback.print_exc()
        
if __name__ == "__main__":
    test_loading()