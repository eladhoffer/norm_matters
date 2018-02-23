# VGG11/13/16/19 in Pytorch.
# From https://github.com/kuangliu/pytorch-cifar/blob/master/models/vgg.py
import torch.nn as nn
from probes_lib.top import ProbesManager
from probes_lib.basic import WeightsPerChannelNormProbe


class VGG(nn.Module):
    def __init__(self, vgg_name, conv_indices_set=None, last_layers_indices_set=None):
        super(VGG, self).__init__()
        cfg = {
            'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
            'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512,
                      'M'],
        }
        self.conv_indices_set = conv_indices_set
        self.last_layers_indices_set = last_layers_indices_set
        self.features = self._make_layers(cfg[vgg_name])
        self.lastlayer_classifier = nn.Linear(512, 10)

    def get_conv_indices_set(self):
        return self.conv_indices_set

    def get_last_layers_indices_set(self):
        return self.last_layers_indices_set

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.lastlayer_classifier(out)
        return out

    @staticmethod
    def _make_layers(cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


def vgg11(**kwargs):
    probes_manager = kwargs.get("probes_manager", None)  # type: ProbesManager
    probes_manager.add_probe(probe=WeightsPerChannelNormProbe(layers_indices={l_idx
                                                                              for l_idx in range(0, 32)
                                                                              if l_idx % 4 == 0}),
                             probe_name="channels_norm_conv", probe_locs=["post_test_forward"])
    probes_manager.add_probe(probe=WeightsPerChannelNormProbe(layers_indices={32}),
                             probe_name="last_channels_norm", probe_locs=["post_test_forward"])
    return VGG('VGG11', conv_indices_set={l_idx for l_idx in range(0, 32) if l_idx % 4 == 0},
               last_layers_indices_set={32})
