import torch

class Ego2ExoDataLoader:
    def __init__(self, opt):
        self.opt = opt
        self.dataset = Ego2ExoBenchmarkDataset(opt)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batch_size,
            shuffle=not opt.serial_batches, # 训练时打乱，测试时不打乱
            num_workers=int(opt.num_workers)
        )

    def load_data(self):
        return self

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            yield data