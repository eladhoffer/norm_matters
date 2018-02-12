import math
import torch
import torch.nn as nn
from torch.nn.functional import relu


class TReLU(nn.Module):
    """docstring for TReLU."""

    def __init__(self):
        super(TReLU, self).__init__()
        self.add_const = math.sqrt(1. / (2 * math.pi))
        self.mul_const = 1 / math.sqrt(0.5 * (1 - 1. / math.pi))

    def forward(self, x):
        return self.mul_const * (relu(x) - self.add_const)
