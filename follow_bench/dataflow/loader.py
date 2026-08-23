"""Helpers to wrap the :class:`BenchmarkDataset` into a DataLoader.

Provides :class:`BenchmarkDataLoader` for use in training/evaluation loops.
"""

import torch
from typing import Iterator, Dict, Any
import torch.nn.functional as F

from .set import BenchmarkDataset
from .option import Options

class BenchmarkDataLoader:
    """Wrapper for BenchmarkDataset providing PyTorch DataLoader functionality.
    
    This class wraps the BenchmarkDataset in a PyTorch DataLoader with
    configurable batch size, shuffling, and multi-processing support.
    
    Attributes:
        opt: Runtime configuration options
        dataset: The underlying BenchmarkDataset instance
        dataloader: PyTorch DataLoader instance
    """
    
    def __init__(self, opt: Options) -> None:
        """Initialize the data loader wrapper.

        Args:
            opt: Options object with runtime configuration.
        """
        self.opt = opt
        self.dataset = BenchmarkDataset(opt)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batch_size,
            shuffle=not opt.serial_batches, # 训练时打乱，测试时不打乱
            num_workers=int(opt.num_workers)
        )

    def load_data(self):
        """Return self for API compatibility with other frameworks.
        
        Returns:
            Self instance for method chaining
        """
        return self

    def __len__(self) -> int:
        """Return the number of samples in the dataset.
        
        Returns:
            Number of samples in the underlying dataset
        """
        return len(self.dataset)
    
    
    
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate over the data loader yielding batches.
        
        Yields:
            Dictionary containing batch data with keys like 'ego_video', 
            'exo_video', 'video_id', etc.
        """
        for i, data in enumerate(self.dataloader):
            yield data
