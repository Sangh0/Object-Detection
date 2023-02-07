from torch.optim.lr_scheduler import _LRScheduler


class CosineWarmupLR(_LRScheduler):

    def __init__(self, optimizer, epochs, lr_min=0, warmup_epochs=0, last_epoch=-1):
        self.lr_min = lr_min
        self.warmup_epochs = warmup_epochs
        self.cosine_epochs = epochs - warmup_epochs
        super(CosineWarmupLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            return [(self.lr_min + (base_lr - self.lr_min) * self.last_epoch / self.warmup_epochs) for base_lr in self.base_lrs]
        else:
            return [(self.lr_min + (base_lr - self.lr_min) * (1 + cos(pi * (self.last_epoch - self.warmup_epochs) / self.cosine_epochs)) / 2) for base_lr in self.base_lrs]