import torch
from torch.optim.optimizer import Optimizer, required
from utils.utils import get_channels_norm


class SGDWDMimic(Optimizer):
    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        self.epoch = None
        self.norms_dict = None
        super(SGDWDMimic, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGDWDMimic, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def update_epoch_and_norms_dict(self, epoch, norms_dict):
        self.epoch = epoch
        self.norms_dict = norms_dict

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf
                if "l_idx" in group.keys() and group["l_idx"] is not None:
                    # Get norms of each channel
                    channel_numel = p.data[0].numel()
                    num_channels = p.data.shape[0]
                    current_norms_pow2 = get_channels_norm(p.data).pow(2)
                    wd_norms_pow2 = (self.norms_dict["channels_norm_conv_w_norm_per_channel_layer" +
                                                     str(group["l_idx"])][self.epoch].pow(2))
                    assert(current_norms_pow2.shape == wd_norms_pow2.shape)
                    # Adjust ratio to get the same stepsize
                    current_norms_pow2.div_(wd_norms_pow2)
                    # Multiply by LR
                    current_norms_pow2.mul_(-group['lr'])

                    # Transform shape to match lr shape
                    step_size = current_norms_pow2.view(num_channels, 1).expand(num_channels, channel_numel)

                    eff_d_p = d_p.view(num_channels, channel_numel).mul(step_size).view(d_p.size())
                    eff_d_p[eff_d_p != eff_d_p] = 0  # reset NaN values
                    p.data.add_(eff_d_p)
                else:
                    d_p[d_p != d_p] = 0  # reset NaN values
                    p.data.add_(-group['lr'], d_p)

        return loss


class SGDWDMimicNormSchedInsteadLR(Optimizer):
    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SGDWDMimicNormSchedInsteadLR, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGDWDMimicNormSchedInsteadLR, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf
                if "l_idx" in group.keys() and group["l_idx"] is not None:
                    p.data.add_(-0.1, d_p)
                else:
                    p.data.add_(-group['lr'], d_p)

        return loss

