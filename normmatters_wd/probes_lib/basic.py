from .top import *
import collections
from utils.utils import merge_list_of_dicts, tensors_norm
from utils.utils import AverageTracker
import sys


class WeightsNormProbe(StatsProbe):
    def __init__(self, **kwargs):
        super(WeightsNormProbe, self).__init__(**kwargs)
        self.last_epoch_stats = {}

    def get_last_epoch_stats(self):
        return self.last_epoch_stats

    def epoch_prologue(self):
        self.last_epoch_stats = {}

    def add_data(self, **kwargs):
        current_weights = kwargs["weights"]
        self.last_epoch_stats["w_norm"] = tensors_norm(current_weights)
        for l_idx, layer in enumerate(current_weights):
            if (layer != layer).sum() > 0:
                self.last_epoch_stats["NaN_WARNING"] = 1
            self.last_epoch_stats["w_norm_layer" + str(l_idx)] = layer.norm()


class WeightsPerChannelNormProbe(StatsProbe):
    def __init__(self, **kwargs):
        super(WeightsPerChannelNormProbe, self).__init__(**kwargs)
        self.layers_indices = kwargs.get("layers_indices")
        self.last_epoch_stats = {}

    def get_last_epoch_stats(self):
        return self.last_epoch_stats

    def epoch_prologue(self):
        self.last_epoch_stats = {}

    def add_data(self, **kwargs):
        current_weights = kwargs["weights"]
        for l_idx, layer in enumerate(current_weights):
            if l_idx in self.layers_indices:
                self.last_epoch_stats["w_norm_per_channel_layer" + str(l_idx)] = (
                    layer.view(layer.shape[0], -1).norm(p=2, dim=1).cpu())


class AccLossProbe(StatsProbe):
    def __init__(self, **kwargs):
        super(AccLossProbe, self).__init__()
        self.type = kwargs["type"]
        assert(self.type == "train" or self.type == "test")
        self.last_epoch_stats = {}

    def get_last_epoch_stats(self):
        return self.last_epoch_stats

    def epoch_prologue(self):
        self.last_epoch_stats = {}

    def add_data(self, **kwargs):
        loss_key = self.type + "_loss"
        acc_key = self.type + "_acc"
        self.last_epoch_stats[loss_key] = kwargs[loss_key]
        self.last_epoch_stats[acc_key] = kwargs[acc_key]


class EpochNumProbe(StatsProbe):
    def __init__(self):
        super(EpochNumProbe, self).__init__()
        self.last_epoch_stats = {}

    def get_last_epoch_stats(self):
        return self.last_epoch_stats

    def epoch_prologue(self):
        self.last_epoch_stats = {}

    def add_data(self, **kwargs):
        self.last_epoch_stats["epoch"] = kwargs["epochs_trained"]
