import torch
import numpy as np
import random
from torch.nn.parallel.data_parallel import DataParallel


def get_model(model):
    if isinstance(model, DataParallel):
        return model.module
    return model


def normalize_channels(vectors, norm=1, in_place=False):
    flat_vector = vectors.view(vectors.shape[0], -1)
    per_channel_norms = flat_vector.norm(p=2, dim=1).mul(1/norm)
    per_channel_norms = per_channel_norms.view(per_channel_norms.shape[0], 1).expand_as(flat_vector)
    if in_place:
        return vectors.view(flat_vector.shape[0], -1).div_(per_channel_norms).view(vectors.size())
    else:
        return vectors.view(flat_vector.shape[0], -1).div(per_channel_norms).view(vectors.size())


def get_channels_norm(vectors):
    flat_vector = vectors.view(vectors.shape[0], -1)
    return flat_vector.norm(p=2, dim=1)


def tensors_norm(tensors):
    if type(tensors) == list:
        norm = [torch.sum(tn ** 2) for tn in tensors]
        norm = float(np.sqrt(np.sum(norm)))
    else:
        norm = tensors.norm()
    return norm


def set_seed(seed, fully_deterministic=True):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        if fully_deterministic:
            torch.backends.cudnn.deterministic = True


def merge_list_of_dicts(dict_list):
    return {k: v for dict_ in dict_list for k, v in dict_.items()}


class AverageTracker:
    def __init__(self):
        self.n = 0
        self.avg = float(0)

    def reset(self):
        self.n = 0
        self.avg = float(0)
        return self

    def add(self, val, n=1):
        self.avg = ((self.avg * self.n) / (self.n + n)) + ((float(val) * n) / (self.n + n))
        self.n += n
        return self

    def copy(self):
        cp = AverageTracker()
        cp.n = self.n
        cp.avg = self.avg
        return cp

    def __iadd__(self, other):
        assert(type(other) == AverageTracker)
        self.add(other.avg, other.n)
        return self
