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


def _std(p, dim):
    """Computes the mean over all dimensions except dim"""
    if dim is None:
        return p.std()
    elif dim == 0:
        output_size = (p.size(0),) + (1,) * (p.dim() - 1)
        return p.contiguous().view(p.size(0), -1).std(dim=1).view(*output_size)
    elif dim == p.dim() - 1:
        output_size = (1,) * (p.dim() - 1) + (p.size(-1),)
        return p.contiguous().view(-1, p.size(-1)).std(dim=0).view(*output_size)
    else:
        return _std(p.transpose(0, dim), 0).transpose(0, dim)

# class MeanBN(nn.Module):
#     """docstring for MeanBN."""
#
#     def __init__(self, num_features, dim=1, momentum=0.1, bias=True, noise=True):
#         super(MeanBN, self).__init__()
#         self.register_buffer('running_mean', torch.zeros(num_features))
#         self.momentum = momentum
#         self.dim = dim
#         self.noise = noise
#         if bias:
#             self.bias = Parameter(torch.Tensor(num_features))
#         else:
#             self.register_parameter('bias', None)
#
#     def forward(self, x):
#
#         if self.training:
#             mean = x.view(x.size(0), x.size(self.dim), -1).mean(-1).mean(0)
#             self.running_mean.mul_(self.momentum).add_(
#                 mean.data * (1 - self.momentum))
#         else:
#             mean = torch.autograd.Variable(self.running_mean)
#         out = x - mean.view(1, mean.size(0), 1, 1)
#         if self.noise and self.training:
#             std_all = _std(x, self.dim).data
#             std_some = _std(x.narrow(0, 0, 128), self.dim).data
#             std_diff =  (std_some**2 - std_all**2).clamp(min=1e-5).sqrt()
#             # print(std_diff.min(), std_diff.mean(), std_diff.max())
#             zeros = torch.zeros_like(x.data)
#             ones = torch.ones_like(x.data)
#
#             std_noise = Variable(torch.normal(zeros, ones) * std_diff)
#             out = out + std_noise
#         if self.bias is not None:
#             out = out + self.bias.view(1, self.bias.size(0), 1, 1)
#         return out


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

#
# class MeanBN(nn.Module):
#     """docstring for MeanBN."""
#
#     def __init__(self, num_features, dim=1, momentum=0.1, bias=True, regularize=False):
#         super(MeanBN, self).__init__()
#         self.register_buffer('running_mean', torch.zeros(num_features))
#         self.momentum = momentum
#         self.dim = dim
#         self.regularize = regularize
#         if bias:
#             self.bias = Parameter(torch.Tensor(num_features))
#         else:
#             self.register_parameter('bias', None)
#
#     def forward(self, x):
#
#         mean = x.view(x.size(0), x.size(1), -
#                       1).mean(-1).view(x.size(0), x.size(1), 1,  1)
#         norm = mean.norm(2, 1, keepdim=True)
#         out = (x - mean) / norm
#         return out


class ReduceMean(Function):

    @staticmethod
    def forward(ctx, inputs, dim):

        output = inputs - inputs.mean(dim, keepdim=True)
        ctx.dim = dim
        return output

    @staticmethod
    def backward(ctx, grad_output):
        N = grad_output.numel() / grad_output.size(ctx.dim)
        grad_input = grad_output * (1 - 1 / N)
        return grad_input, None


def reduce_mean(x, dim):
    return ReduceMean().apply(x, dim)


class MeanBN(nn.Module):
    """docstring for MeanBN."""

    def __init__(self, num_features, dim=1, momentum=0.1, bias=True, regularize=False):
        super(MeanBN, self).__init__()
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.momentum = momentum
        self.dim = dim
        self.regularize = regularize
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

        if self.regularize and self.training:
            std = _std(out, self.dim)
            self.std_regularize = [std.mean()]
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


class MeanRN(nn.Module):
    """docstring for MeanBN."""

    def __init__(self, num_features, dim=1, momentum=0.1, d_max=1, bias=True):
        super(MeanRN, self).__init__()
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.momentum = momentum
        self.dim = dim
        self.d_max = d_max
        if bias:
            self.bias = Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):

        if self.training:
            mean = x.view(x.size(0), x.size(self.dim), -1).mean(-1).mean(0)
            diff = mean.data - self.running_mean
            d = Variable(diff.clamp(-self.d_max, self.d_max))
            self.running_mean.mul_(self.momentum).add_(
                diff * (1 - self.momentum))
            delta = mean - d
        else:
            delta = torch.autograd.Variable(self.running_mean)
        out = x - delta.view(1, delta.size(0), 1, 1)
        if self.bias is not None:
            out = out + self.bias.view(1, self.bias.size(0), 1, 1)
        return out
