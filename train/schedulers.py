import numpy as np
import torch
import torch.optim.lr_scheduler as sch


class WarmupScheduler(sch._LRScheduler):
    def __init__(
            self, optimizer: torch.optim.Optimizer, warmup: int = 80_000, max_iters: int = 700_000
    ):
        self.warmup, self.max_iters = warmup, max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch / self.warmup
        return lr_factor
