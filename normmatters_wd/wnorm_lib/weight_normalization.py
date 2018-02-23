from utils.utils import normalize_channels, get_model


class WeightsNormalization:
    def __init__(self, param_groups):
        self.param_groups = param_groups  # Dict with keys: ["params"] , ["name"] , ["norm"]
        self.epoch = 0

    def update_epoch(self, epoch):
        self.epoch = epoch

    def step(self):
        for group in self.param_groups:
            if not isinstance(group['norm'], float) and not isinstance(group['norm'], int):
                norm = group['norm'][self.epoch]
            else:
                norm = group['norm']
                if norm == 0:
                    continue
            normalize_channels(group['params'], norm=norm, in_place=True)


def per_channel_normalization_norm(model, norm, normalization_lst_method_name="get_conv_indices_set"):
    model = get_model(model)
    if norm == 0 or not hasattr(model, normalization_lst_method_name):
        return None
    layers_set = getattr(model, normalization_lst_method_name)()
    channels_params = [{'params': params.data, 'name': name, 'norm': norm}
                       for l, (name, params) in enumerate(model.named_parameters()) if l in layers_set]
    return WeightsNormalization(channels_params)


def per_channel_normalization_norm_as_wd(model, norm, normalization_lst_method_name="get_conv_indices_set", norms_dict=None):
    model = get_model(model)

    if not hasattr(model, normalization_lst_method_name):
        return None
    layers_set = getattr(model, normalization_lst_method_name)()
    import numpy as np
    norms_dict_for_l = {}
    for l in layers_set:
        norms_dict_for_l[l] = norms_dict["channels_norm_conv_w_norm_per_channel_layer" + str(l)]
        for ep in range(0,norms_dict["channels_norm_conv_w_norm_per_channel_layer" + str(l)].__len__()):
            factor = float(np.sqrt(10))**(ep//20)
            norms_dict_for_l[l][ep] *= factor
    channels_params = [{'params': params.data, 'name': name, 'norm': norms_dict_for_l[l]}
                       for l, (name, params) in enumerate(model.named_parameters()) if l in layers_set]
    return WeightsNormalization(channels_params)

