from utils.logging_utils import Logger
import torch
from time import gmtime, strftime, time
from probes_lib.basic import *
from utils.utils import AverageTracker, normalize_channels
import pickle


class NNTrainer:
    def __init__(self, train_loader, test_loader, criterion, optimizer, net, logger, **kwargs):
        # NN Configurations variables
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.criterion = criterion       # type: torch.nn.modules.loss
        self.logger = logger             # type: Logger
        self.net = net                   # type: torch.nn.Module
        self.using_cuda = False

        self.wd_conv_norms_dict = None
        if kwargs.get("wd_conv_norms_dict", "") != "":
            with open(kwargs.get("wd_conv_norms_dict", ""), 'rb') as f:
                self.wd_conv_norms_dict = pickle.load(f)
            for l in self.wd_conv_norms_dict.values():
                for ep_idx in range(0, l.__len__()):
                    if torch.cuda.is_available():
                        l[ep_idx] = l[ep_idx].cuda()

        self.weight_normalization = kwargs.get("weight_normalization", [])

        self.desc = kwargs.get("desc", None)

        self.lr_sched = kwargs.get("lr_scheduler", [])

        # Statistics variables
        self.epochs_trained = 0
        self.probes_manager = kwargs.get("probes_manager", None)

        # Create probes
        self.probes_manager.add_probe(probe=AccLossProbe(type="train"), probe_name="train_acc_loss",
                                      probe_locs=["post_train_forward"])
        self.probes_manager.add_probe(probe=AccLossProbe(type="test"), probe_name="test_acc_loss",
                                      probe_locs=["post_test_forward"])
        self.probes_manager.add_probe(probe=EpochNumProbe(),
                                      probe_name="epoch", probe_locs=["post_test_forward"])

        if torch.cuda.is_available():
            self.using_cuda = True

        # Save initial weights
        self.probes_manager.add_probe(probe=WeightsNormProbe(),
                                      probe_name="weight_norm", probe_locs=["post_test_forward"])
        self.num_params_in_layer = [w.data.numel() for w in list(self.net.parameters())]

        # Print parameters
        self.print_num_of_net_params()
        self.print_optimizer_params()
        self.print_criterion_params()
        self.layers_names = {l: name for l, (name, _) in enumerate(self.net.named_parameters())}

    # Training and testing
    def train_epochs(self, verbose_freq=2000, max_epoch=1, save_model_on_epochs=None):
        if self.epochs_trained >= max_epoch:
            return False

        self.logger.info("Running training from epoch " + str(self.epochs_trained + 1) + " to epoch " + str(max_epoch))

        for epoch_number in range(self.epochs_trained, max_epoch):
            start_time = time()
            #####
            # Change learning rate or weight decay if necessary
            #####
            self.lr_sched.step() if self.lr_sched else None

            for weight_normalize in self.weight_normalization:
                weight_normalize.update_epoch(epoch=self.epochs_trained) if weight_normalize is not None else None

            ######
            ### LR correction
            if hasattr(self.optimizer, "update_epoch_and_norms_dict"):
                self.optimizer.update_epoch_and_norms_dict(self.epochs_trained, self.wd_conv_norms_dict)

            ######

            self.probes_manager.epoch_prologue()
            #####
            # Train
            #####
            self.logger.info("Training epoch number " + str(epoch_number + 1))
            self.train_mode()
            return_predictions_data = self.probes_manager.return_predictions_data["train"]
            train_loss, train_acc = self.forward(data_loader=self.train_loader, training=True,
                                                                       verbose_freq=verbose_freq,
                                                                       return_predictions_data=return_predictions_data)
            probe_data = {"train_loss": train_loss,
                          "train_acc": train_acc,
                          "net": self.net,
                          "weights": self.weights_lst(),
                          "epochs_trained": self.epochs_trained}
            self.probes_manager.add_data(probe_loc="post_train_forward", **probe_data)

            #####
            # Test
            #####
            self.logger.info("Running test set for epoch number " + str(epoch_number + 1))
            self.eval_mode()
            return_predictions_data = self.probes_manager.return_predictions_data["test"]
            test_loss, test_acc = self.forward(data_loader=self.test_loader, training=False,
                                                                     verbose_freq=0,
                                                                     return_predictions_data=return_predictions_data)
            probe_data = {"test_loss": test_loss,
                          "test_acc": test_acc,
                          "net": self.net,
                          "weights": self.weights_lst(),
                          "optimizer": self.optimizer,
                          "epochs_trained": self.epochs_trained}
            self.probes_manager.add_data(probe_loc="post_test_forward", **probe_data)

            #####
            # Save statistics and model
            #####
            self.probes_manager.epoch_epilogue()
            self.probes_manager.calc_epoch_stats()
            self.save_current_stats()

            #self.save_model(filename="model.pth.tar")

            # Save model to specific file (to allow reproduction of specific epoch)
            if save_model_on_epochs and (epoch_number + 1 in save_model_on_epochs):
                model_of_epoch_fname = (self.logger.get_log_basename() + "_model_epoch" + str(epoch_number + 1) +
                                        ".pth.tar")
                self.logger.info("Saving model of epoch " + str(epoch_number + 1) + " to " + str(model_of_epoch_fname))
                self.save_model(filename=model_of_epoch_fname)

            self.logger.info("Finished epoch number " + str(epoch_number + 1) +
                             ", Took " + str(int(time() - start_time)) + " seconds")
        return True

    def transform_data_to_cuda_if_necessary(self, data):
        if self.using_cuda:
            return data.cuda()
        return data

    def save_current_stats(self):
        vars_dict = {'per_epoch_stats': self.probes_manager.per_epoch_stats,
                     'epochs_trained': self.epochs_trained,
                     'layers_names': self.layers_names,
                     'desc': self.desc}
        self.logger.save_variables(var=vars_dict, var_name="stats")

    def save_conv_channels_weight_norms(self):
        conv_columns = [col_name for col_name in self.probes_manager.per_epoch_stats.columns if
                        "norm_conv" in col_name]
        cols_stats = {col_name: self.probes_manager.per_epoch_stats[col_name] for col_name in conv_columns}
        self.logger.save_variables(var=cols_stats, var_name="conv_norms")

    def weights_lst(self):
        return [w.data.clone() for w in list(self.net.parameters())]

    def weights_grad_lst(self):
        return [w.grad.clone() for w in list(self.net.parameters())]

    def train_mode(self):
        self.net.train()
        self.probes_manager.train()

    def eval_mode(self):
        self.net.eval()
        self.probes_manager.eval()

    def forward(self, data_loader=None, verbose_freq=2000, return_predictions_data=False, training=True):
        if training:
            self.train_mode()
            fwd_name = "train"
        else:
            self.eval_mode()
            fwd_name = "test"

        loss_avg = AverageTracker()
        acc_avg = AverageTracker()

        set_size = 0
        all_predictions = torch.LongTensor()
        all_labels = torch.LongTensor()

        for i, data in enumerate(data_loader, 0):
            # Get data
            inputs, labels = data

            inputs, labels = (self.transform_data_to_cuda_if_necessary(inputs),
                              self.transform_data_to_cuda_if_necessary(labels))
            inputs, labels = (torch.autograd.Variable(inputs, volatile=not training),
                              torch.autograd.Variable(labels, volatile=not training))

            if not training:
                self.optimizer.zero_grad()

            outputs = self.net(inputs)

            # Calc loss
            loss = self.criterion(outputs, labels)
            loss_avg.add(loss.data[0], inputs.size(0))

            # Backprop
            if training:
                # Zero the gradient
                self.optimizer.zero_grad()
                loss.backward()
                if hasattr(self.optimizer, "weights_norm"):
                    self.optimizer.set_weights_norm(tensors_norm(self.weights_lst()))
                if hasattr(self.optimizer, "w_norm"):
                    self.optimizer.set_w_norm(tensors_norm(self.weights_lst()))
                self.probes_manager.add_data(probe_loc="post_backward_pre_optim_step",
                                             weights_grad=self.weights_grad_lst(),
                                             weights=self.weights_lst())
                self.optimizer.step()
                for weight_normalize in self.weight_normalization:
                    weight_normalize.step() if weight_normalize is not None else None

            # Calc acc
            _, predicted = torch.max(outputs.data, dim=1)
            correct_tags = (predicted == labels.data).sum()
            acc_avg.add((float(correct_tags)/labels.size(0))*100, labels.size(0))

            if return_predictions_data:
                all_predictions = torch.cat([all_predictions, predicted.cpu()])
                all_labels = torch.cat([all_labels, labels.data.cpu()])
            set_size += labels.size(0)
            if verbose_freq and verbose_freq > 0 and (i % verbose_freq) == (verbose_freq - 1):
                self.logger.info("Epoch " + str(self.epochs_trained + 1) + ", " + fwd_name + " set, " +
                                 "Iter " + str(i + 1) +
                                 " current average loss " + str(loss_avg.avg) +
                                 " current average acc " + str(acc_avg.avg) + "%")

        if training:
            self.epochs_trained += 1

        self.logger.info("Stats for " + fwd_name + " set of size " + str(set_size) + ", " +
                         "loss is " + str(loss_avg.avg) + ", " +
                         "acc is " + str(acc_avg.avg) + "%")

        return loss_avg.avg, acc_avg.avg

    # Print functions
    def print_num_of_params(self):
        num_of_params = 0
        for parameter in self.net.parameters():
            num_of_params += parameter.numel()
        self.logger.info("Number of parameters in the model is " + str("{:,}".format(num_of_params)))

    def print_layers(self):
        for l, (name, parameter) in enumerate(self.net.named_parameters()):
            self.logger.info("layer " + str(l) + " is " + name + " with number of parameters " +
                             str("{:,}".format(parameter.numel())) + " and norm " + str(parameter.data.cpu().norm()))

    def print_criterion_params(self):
        self.logger.info("Criterion parameters: type=" + str(type(self.criterion)))

    def print_optimizer_params(self):
        message = ""
        message += "Optimizer type=" + str(type(self.optimizer))
        for param_idx, param_group in enumerate(self.optimizer.param_groups):
            message += "Optimizer parameters group " + str(param_idx) + ": "
            if 'lr' in self.optimizer.param_groups[param_idx].keys():
                message += ", lr=" + str(self.optimizer.param_groups[param_idx]['lr'])
            if 'momentum' in self.optimizer.param_groups[param_idx].keys():
                message += ", momentum=" + str(self.optimizer.param_groups[param_idx]['momentum'])
            if 'weight_decay' in self.optimizer.param_groups[param_idx].keys():
                message += ", weight_decay=" + str(self.optimizer.param_groups[param_idx]['weight_decay'])
            if 'name' in self.optimizer.param_groups[param_idx].keys():
                message += ", name=" + str(self.optimizer.param_groups[param_idx]['name'])

        self.logger.info(message)

    def print_num_of_net_params(self):
        num_of_params = 0
        for parameter in self.net.parameters():
            num_of_params += parameter.numel()
        self.logger.info("Number of parameters in the model is " + str("{:,}".format(num_of_params)))
