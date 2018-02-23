from utils.utils import *
from utils.logging_utils import Logger
from utils.datasets import DatasetsLoaders
from nn_utils.NNTrainer import NNTrainer
from probes_lib.top import *
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch
import argparse
import models
import optimizers_lib
from time import time
from wnorm_lib.weight_normalization import per_channel_normalization_norm, per_channel_normalization_norm_as_wd
import pickle


###########################################################################
# Script's arguments
###########################################################################
archs_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

optimizers_names = sorted(name for name in optimizers_lib.__dict__
                          if name.islower() and not name.startswith("__")
                          and callable(optimizers_lib.__dict__[name]))

parser = argparse.ArgumentParser(description='Train and record statistics of a Neural Network')
parser.add_argument('--dataset', default="CIFAR10", type=str, choices=["CIFAR10", "CIFAR100", "MNIST"],
                    help='The name of the dataset to train. [Default: CIFAR10]')
parser.add_argument('--nn_arch', type=str, default="vgg11", choices=archs_names,
                    help='Neural network architecture')
parser.add_argument('--optimizer', type=str, default="sgd_wd0_0005_lr0_1_momentum0_9", choices=optimizers_names,
                    help='Optimizer')
parser.add_argument('--logname', type=str, required=True,
                    help='Prefix of logfile name')
parser.add_argument('--seed', type=int,
                    help='Seed for randomization. If provided num_workers must be 0')
parser.add_argument('--num_workers', default=8, type=int,
                    help='Num of workers for data loader')
parser.add_argument('--num_epochs', default=400, type=int,
                    help='Maximum number of training epochs. If resuming from a trained network, '
                         + 'training will stop at num_epochs [Default: 400]')
parser.add_argument('--batch_size', default=128, type=int,
                    help='Batch size [Default: 128]')
# Optimizer
parser.add_argument('--sgd_momentum', default=0.9, type=float,
                    help='SGD momentum [Default: 0.9]')
# LR scheduler args
parser.add_argument('--lr_sched_step', type=int,
                    help='LR scheduler step size [Default: num_epochs//4]')
parser.add_argument('--lr_sched_gamma', default=0.1, type=float,
                    help='LR scheduler gamma (factor) [Default: 0.1]')

# Save convolution layers' channels norms
parser.add_argument('--save_conv_channels_norms', default=False, action='store_true',
                    help='Save convolution layers channels norms')

# Pickle file with values of norms to normalize convolution layers
parser.add_argument('--wd_conv_norms_dict', type=str, default="",
                    help='wd_conv_norms_dict filename')

# Use norm instead of LR
parser.add_argument('--norm_lr_sched', default=False, action='store_true',
                    help='Use norm scheduling instead of LR')

args = parser.parse_args()

###########################################################################
# Verify arguments
###########################################################################
if args.seed:
    assert(args.num_workers == 0)

###########################################################################
# CUDA
###########################################################################
use_benchmark = False
if not args.seed:
    use_benchmark = True
    args.seed = int(time()*10000) % (2**31)

set_seed(args.seed, fully_deterministic=not use_benchmark)

if torch.cuda.is_available() and use_benchmark:
    cudnn.benchmark = True


###########################################################################
# Logging
###########################################################################
# Create logger
logger = Logger(True, args.logname, True, True)

logger.info("Script args: " + str(args))
logger.create_desc_file(str(args))

###########################################################################
# Model and training
###########################################################################
# Params
acts_save_handler = None

# Dataset
dataset = DatasetsLoaders(args.dataset, batch_size=args.batch_size, num_workers=args.num_workers)

# Probes manager
probes_manager = ProbesManager()

# Model
model = models.__dict__[args.nn_arch]

model_args = {"per_layer_saved_acts_size": 10000,
              "test_set_size": dataset.test_set.test_data.shape[0],
              "train_set_size": dataset.train_set.train_data.shape[0],
              "batch_size": args.batch_size,
              "save_conv_channels_norms": args.save_conv_channels_norms}

model = model(probes_manager=probes_manager, **model_args)

# Transform net to cuda if available
if torch.cuda.is_available():
    model = torch.nn.DataParallel(model).cuda()
    logger.info("Transformed model to CUDA")

criterion = nn.CrossEntropyLoss()

# optimizer model
optimizer_model = optimizers_lib.__dict__[args.optimizer]
optimizer = optimizer_model(model)

weight_normalization = [None]

if args.norm_lr_sched:
    wd_conv_norms_dict = None
    if args.wd_conv_norms_dict != "":
        with open(args.wd_conv_norms_dict, 'rb') as f:
            wd_conv_norms_dict = pickle.load(f)
        for l in wd_conv_norms_dict.values():
            for ep_idx in range(0, l.__len__()):
                if torch.cuda.is_available():
                    l[ep_idx] = l[ep_idx].cuda()
    weight_normalization_conv = per_channel_normalization_norm_as_wd(model, 1,
                                                                     normalization_lst_method_name="get_conv_indices_set",
                                                                     norms_dict=wd_conv_norms_dict)
    weight_normalization = [weight_normalization_conv]

# Schedulers
lr_sched = torch.optim.lr_scheduler.StepLR(optimizer, step_size=(args.lr_sched_step or args.num_epochs//4 or 1),
                                           gamma=args.lr_sched_gamma, last_epoch=-1)


trainer = NNTrainer(train_loader=dataset.train_loader, test_loader=dataset.test_loader, criterion=criterion,
                    optimizer=optimizer, net=model, logger=logger, probes_manager=probes_manager,
                    lr_scheduler=lr_sched, weight_normalization=weight_normalization,
                    wd_conv_norms_dict=args.wd_conv_norms_dict)

trainer.train_epochs(verbose_freq=100, max_epoch=args.num_epochs)

if args.save_conv_channels_norms:
    trainer.save_conv_channels_weight_norms()

print("Done")
