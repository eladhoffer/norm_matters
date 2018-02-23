import torchvision.transforms as transforms
import torchvision
import torch


class DatasetsLoaders:
    def __init__(self, dataset, batch_size=4, num_workers=4, pin_memory=True, **kwargs):
        self.dataset_name = dataset
        self.valid_loader = None
        self.num_workers = num_workers
        if self.num_workers is None:
            self.num_workers = 4

        self.random_erasing = kwargs.get("random_erasing", False)

        pin_memory = pin_memory if torch.cuda.is_available() else False
        self.batch_size = batch_size
        mnist_mean = [33.318421449829934]
        mnist_std = [78.56749083061408]

        if dataset == "CIFAR10":
            # CIFAR10:
            #   type               : uint8
            #   shape              : train_set.train_data.shape (50000, 32, 32, 3)
            #   test data shape    : (10000, 32, 32, 3)
            #   number of channels : 3
            #   Mean per channel   : train_set.train_data[:,:,:,0].mean() 125.306918046875
            #                        train_set.train_data[:,:,:,1].mean() 122.95039414062499
            #                        train_set.train_data[:,:,:,2].mean() 113.86538318359375
            #   Std per channel   :  train_set.train_data[:, :, :, 0].std() 62.993219278136884
            #                        train_set.train_data[:, :, :, 1].std() 62.088707640014213
            #                        train_set.train_data[:, :, :, 2].std() 66.704899640630913

            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

            self.train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                          download=True, transform=transform_train)
            self.train_loader = torch.utils.data.DataLoader(self.train_set, batch_size=self.batch_size,
                                                            shuffle=True, num_workers=self.num_workers,
                                                            pin_memory=pin_memory)

            self.test_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                         download=True, transform=transform_test)
            self.test_loader = torch.utils.data.DataLoader(self.test_set, batch_size=self.batch_size,
                                                           shuffle=False, num_workers=self.num_workers,
                                                           pin_memory=pin_memory)


        if dataset == "MNIST":
            # MNIST:
            #   type               : torch.ByteTensor
            #   shape              : train_set.train_data.shape torch.Size([60000, 28, 28])
            #   test data shape    : [10000, 28, 28]
            #   number of channels : 1
            #   Mean per channel   : 33.318421449829934
            #   Std per channel    : 78.56749083061408
            self.mean = mnist_mean
            self.std = mnist_std
            transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize(self.mean, self.std)])

            self.train_set = torchvision.datasets.MNIST(root='./data', train=True,
                                                        download=True, transform=transform)
            self.train_loader = torch.utils.data.DataLoader(self.train_set, batch_size=self.batch_size,
                                                            shuffle=True, num_workers=self.num_workers,
                                                            pin_memory=pin_memory)

            self.test_set = torchvision.datasets.MNIST(root='./data', train=False,
                                                       download=True, transform=transform)
            self.test_loader = torch.utils.data.DataLoader(self.test_set, batch_size=self.batch_size,
                                                           shuffle=False, num_workers=self.num_workers,
                                                           pin_memory=pin_memory)
