"""Helpers to wrap the :class:`BenchmarkDataset` into a DataLoader.

Provides :class:`BenchmarkDataLoader` for use in training/evaluation loops.
"""

import torch
from typing import Iterator, Dict, Any

from .set import BenchmarkDataset
from .option import Options

class BenchmarkDataLoader:
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

    def load_data(self) -> "BenchmarkDataLoader":
        """Return self for API compatibility with other frameworks."""
        return self

    def __len__(self) -> int:
        return len(self.dataset)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        for i, data in enumerate(self.dataloader):
            yield data