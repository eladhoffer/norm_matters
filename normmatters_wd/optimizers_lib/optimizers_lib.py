import torch.optim as optim
from optimizers_lib.custom_sgd import SGDWDMimic, SGDWDMimicNormSchedInsteadLR
from utils.utils import get_model


# [WD on] Regular WD on
def sgd_wd0_0005_lr0_1_momentum0_9(model, **kwargs):
    all_params = [{'params': params, 'name': name} for l, (name, params) in enumerate(model.named_parameters())]
    return optim.SGD(all_params, momentum=0.9, lr=0.1, weight_decay=5e-4)


# [WD off + correction] - Step size correction
def sgd_lastlayerwd0_0005_otherlayerswd0_lr0_1_with_correction(model, **kwargs):
    conv_layers_indices = get_model(model).get_conv_indices_set()
    lastlayer_params = [{'params': params, 'name': name, 'weight_decay': 5e-4}
                        for l, (name, params) in enumerate(model.named_parameters()) if "lastlayer" in name]
    notconv_notlastlayer_params = [{'params': params, 'name': name, 'weight_decay': 0}
                                   for l, (name, params) in enumerate(model.named_parameters())
                                   if (not ("lastlayer" in name)) and l not in conv_layers_indices]
    convlayer_params = [{'params': params, 'name': name, 'weight_decay': 0, 'l_idx': l, 'wd_norms': None}
                        for l, (name, params) in enumerate(model.named_parameters())
                        if (not ("lastlayer" in name)) and l in conv_layers_indices]

    all_params = lastlayer_params + notconv_notlastlayer_params + convlayer_params
    return SGDWDMimic(all_params, momentum=0.9, lr=0.1, weight_decay=0.0)


# [WD off] - Last layer wd on, rest off
def sgd_lastlayerwd0_0005_otherlayerswd0_lr0_1_momentum0_9(model, **kwargs):
    lastlayer_params = [{'params': params, 'name': name, 'weight_decay': 5e-4}
                        for l, (name, params) in enumerate(model.named_parameters()) if "lastlayer" in name]
    notlastlayer_params = [{'params': params, 'name': name, 'weight_decay': 0}
                           for l, (name, params) in enumerate(model.named_parameters()) if not ("lastlayer" in name)]
    all_params = lastlayer_params + notlastlayer_params
    return optim.SGD(all_params, momentum=0.9, lr=0.1, weight_decay=0.0)


# WD mimic+norm scheduling instead of lr scheduling
def sgd_lastlayerwd0_0005_otherlayerswd0_lr0_1_norm_sched_instead(model, **kwargs):
    conv_layers_indices = get_model(model).get_conv_indices_set()
    lastlayer_params = [{'params': params, 'name': name, 'weight_decay': 5e-4}
                        for l, (name, params) in enumerate(model.named_parameters()) if "lastlayer" in name]
    notconv_notlastlayer_params = [{'params': params, 'name': name, 'weight_decay': 0}
                                   for l, (name, params) in enumerate(model.named_parameters())
                                   if (not ("lastlayer" in name)) and l not in conv_layers_indices]
    convlayer_params = [{'params': params, 'name': name, 'weight_decay': 0, 'l_idx': l, 'wd_norms': None}
                        for l, (name, params) in enumerate(model.named_parameters())
                        if (not ("lastlayer" in name)) and l in conv_layers_indices]

    all_params = lastlayer_params + notconv_notlastlayer_params + convlayer_params
    return SGDWDMimicNormSchedInsteadLR(all_params, momentum=0.9, lr=0.1, weight_decay=0.0)

