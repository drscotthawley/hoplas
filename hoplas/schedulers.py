import torch


class WarmupThenPlateauWithReduction:
    """Linear LR warmup for the first `boundary` epochs, then ReduceLROnPlateau.

    SequentialLR can't hold ReduceLROnPlateau (it's metric-driven, not an LRScheduler),
    so this thin router does the phase switch instead. Call once per epoch:
        scheduler.step(metric, epoch)
    The warmup scheduler ignores `metric`; the plateau scheduler ignores `epoch`.

    Shared by train_ops.py and train_kge.py.
    """
    def __init__(self, warmup, plateau, boundary):
        self.warmup, self.plateau, self.boundary = warmup, plateau, boundary

    def step(self, metric, epoch):
        if self.warmup is not None and epoch <= self.boundary:
            self.warmup.step()
        else:
            self.plateau.step(metric)


def make_warmup_plateau(opt, base_lr, warmup_epochs, warmup_start_lr, lr_patience, factor=0.5):
    """Build a WarmupThenPlateauWithReduction for `opt`: linear ramp from
    warmup_start_lr -> base_lr over `warmup_epochs`, then halve on plateau
    (patience=lr_patience, floor base_lr/20). Step once per epoch with (metric, epoch)."""
    plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=factor, patience=lr_patience, min_lr=base_lr / 20)
    warmup = (torch.optim.lr_scheduler.LinearLR(
        opt, start_factor=max(warmup_start_lr / base_lr, 1e-8), total_iters=warmup_epochs)
        if warmup_epochs > 0 else None)
    return WarmupThenPlateauWithReduction(warmup, plateau, warmup_epochs)
