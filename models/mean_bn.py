import torch
from torch.nn.parameter import Parameter
from torch.autograd import Variable, Function
import torch.nn as nn
import math


def _norm(x, dim, p=2):
    """Computes the norm over all dimensions except dim"""
    if p == -1:
        func = lambda x, dim: x.max(dim=dim)[0] - x.min(dim=dim)[0]
    elif p == float('inf'):
        func = lambda x, dim: x.max(dim=dim)[0]
    else:
        func = lambda x, dim: torch.norm(x, dim=dim, p=p)
    if dim is None:
        return x.norm(p=p)
    elif dim == 0:
        output_size = (x.size(0),) + (1,) * (x.dim() - 1)
        return func(x.contiguous().view(x.size(0), -1), 1).view(*output_size)
    elif dim == x.dim() - 1:
        output_size = (1,) * (x.dim() - 1) + (x.size(-1),)
        return func(x.contiguous().view(-1, x.size(-1)), 0).view(*output_size)
    else:
        return _norm(x.transpose(0, dim), 0).transpose(0, dim)


def _mean(p, dim):
    """Computes the mean over all dimensions except dim"""
    if dim is None:
        return p.mean()
    elif dim == 0:
        output_size = (p.size(0),) + (1,) * (p.dim() - 1)
        return p.contiguous().view(p.size(0), -1).mean(dim=1).view(*output_size)
    elif dim == p.dim() - 1:
        output_size = (1,) * (p.dim() - 1) + (p.size(-1),)
        return p.contiguous().view(-1, p.size(-1)).mean(dim=0).view(*output_size)
    else:
        return _mean(p.transpose(0, dim), 0).transpose(0, dim)


def gather_regularization(self, memo=None, param_func=lambda s: s.std_regularize):
    if memo is None:
        memo = set()
    for p in param_func(self):
        if p is not None and p not in memo:
            memo.add(p)
            yield p
    for m in self.children():
        for p in gather_regularization(m, memo, param_func):
            yield p

nn.Module.gather_regularization = gather_regularization
nn.Module.std_regularize = []



class MeanBN(nn.Module):
    """docstring for MeanBN."""

    def __init__(self, num_features, dim=1, momentum=0.1, bias=True):
        super(MeanBN, self).__init__()
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.momentum = momentum
        self.dim = dim
        if bias:
            self.bias = Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):

        if self.training:
            mean = x.view(x.size(0), x.size(self.dim), -1).mean(-1).mean(0)
            self.running_mean.mul_(self.momentum).add_(
                mean.data * (1 - self.momentum))
        else:
            mean = torch.autograd.Variable(self.running_mean)
        out = x - mean.view(1, mean.size(0), 1, 1)

        if self.bias is not None:
            out = out + self.bias.view(1, self.bias.size(0), 1, 1)
        return out


class L1BatchNorm(nn.Module):
    """docstring for MeanBN."""

    def __init__(self, num_features, dim=1, momentum=0.1, affine=True):
        super(L1BatchNorm, self).__init__()
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_scale', torch.zeros(num_features))
        self.momentum = momentum
        self.dim = dim
        if affine:
            self.bias = Parameter(torch.Tensor(num_features))
            self.weight = Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('bias', None)
            self.register_parameter('weight', None)

    def forward(self, x):

        if self.training:
            x_flat = x.view(x.size(0), x.size(self.dim), -1)
            mean = x_flat.mean(-1).mean(0)
            x_centered = (x - mean.view(1, mean.size(0), 1, 1))
            x_flat = x_centered.view(x.size(0), x.size(self.dim), -1)
            scale = x_flat.abs().mean(-1).mean(0) * math.sqrt(math.pi / 2)
            out = x_centered / scale.view(1, scale.size(0), 1, 1)
            self.running_mean.mul_(self.momentum).add_(
                mean.data * (1 - self.momentum))
            self.running_scale.mul_(self.momentum).add_(
                scale.data * (1 - self.momentum))
        else:
            mean = torch.autograd.Variable(self.running_mean)
            scale = torch.autograd.Variable(self.running_scale)
            out = (x - mean.view(1, mean.size(0), 1, 1)) / scale.view(1, scale.size(0), 1, 1)
        if self.weight is not None:
            out = out * self.weight.view(1, self.bias.size(0), 1, 1)
        if self.bias is not None:
            out = out + self.bias.view(1, self.bias.size(0), 1, 1)
        return out
