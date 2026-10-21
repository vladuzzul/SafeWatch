"""The selected InceptionTime architecture for fall classification."""

import torch
from torch import nn


class InceptionModule(nn.Module):
    def __init__(self, incoming, width, bottleneck, kernels):
        super().__init__()
        bottleneck = min(incoming, bottleneck)
        self.bottleneck = nn.Conv1d(incoming, bottleneck, 1, bias=False)
        # Odd kernels make symmetric SAME padding unambiguous.
        self.branches = nn.ModuleList([
            nn.Conv1d(bottleneck, width, k, padding=k // 2, bias=False) for k in kernels
        ])
        self.pool = nn.Sequential(nn.MaxPool1d(3, stride=1, padding=1),
                                  nn.Conv1d(incoming, width, 1, bias=False))
        self.finish = nn.Sequential(nn.BatchNorm1d(4 * width), nn.ReLU())

    def forward(self, x):
        reduced = self.bottleneck(x)
        return self.finish(torch.cat([branch(reduced) for branch in self.branches]
                                     + [self.pool(x)], dim=1))


class InceptionTime(nn.Module):
    def __init__(self, channels=6, width=16, depth=6, bottleneck=16,
                 kernels=(9, 19, 39)):
        super().__init__()
        if depth < 3 or depth % 3:
            raise ValueError("Inception depth must be a positive multiple of 3")
        if len(kernels) != 3 or any(k < 1 or k % 2 == 0 for k in kernels):
            raise ValueError("Use three positive odd kernels")
        self.blocks = nn.ModuleList()
        self.shortcuts = nn.ModuleList()
        incoming = channels
        for _ in range(depth // 3):
            self.blocks.append(nn.Sequential(
                InceptionModule(incoming, width, bottleneck, kernels),
                InceptionModule(4 * width, width, bottleneck, kernels),
                InceptionModule(4 * width, width, bottleneck, kernels),
            ))
            self.shortcuts.append(nn.Sequential(nn.Conv1d(incoming, 4 * width, 1,
                                                          bias=False),
                                                  nn.BatchNorm1d(4 * width)))
            incoming = 4 * width
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(incoming, 1)

    def forward(self, windows):
        x = windows
        for block, shortcut in zip(self.blocks, self.shortcuts):
            x = torch.relu(block(x) + shortcut(x))
        return self.head(self.pool(x).squeeze(-1)).squeeze(-1)


def build_model(config):
    return InceptionTime(width=config["inception_width"],
                         depth=config["inception_depth"],
                         bottleneck=config["inception_bottleneck"],
                         kernels=tuple(config["inception_kernels"]))


def bag_logits(window_logits, owners, bag_count):
    """MIL: a positive recording needs at least one positive window.

    No invented event timestamps or per-window targets are used.
    """
    # collate_bags emits contiguous groups. Slicing avoids boolean-index
    # backward operations that are nondeterministic on Apple MPS.
    counts = torch.bincount(owners.cpu(), minlength=bag_count).tolist()
    if len(counts) != bag_count or any(count == 0 for count in counts):
        raise ValueError("Each bag needs at least one window")
    if torch.any(owners[1:] < owners[:-1]):
        raise ValueError("Windows must be contiguous and ordered by bag")
    return torch.stack([part.max() for part in torch.split(window_logits, counts)])
