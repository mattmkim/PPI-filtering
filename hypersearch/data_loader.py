import numpy as np
import torch
from ml_filtering_pytorch import get_data
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.dataset import Dataset


class thymidyalte_data(Dataset):
    def __init__(self, transforms=None):
        x_train, x_test, y_train, y_test, df_train = get_data()
        print(y_train)
        self.data, self.labels = np.concatenate((x_train, x_test), axis=0), np.concatenate((y_train, y_test), axis=0)
        self.transforms = transforms

    def __getitem__(self, index):
        single_label = self.labels[index]
        single_data = self.data[index]
        return (single_data, single_label)

    def __len__(self):
        return len(self.data)

    def get_train_end(self):
        x_train, x_test, y_train, y_test, df_train = get_data()
        return len(x_train - 1)


def get_train_valid_loader(batch_size,
                           shuffle=True,
                           num_workers=4,
                           pin_memory=False):
    """
    Utility function for loading and returning train and valid
    multi-process iterators over the desired dataset.

    Params
    ------
    - data_dir: path directory to the dataset.
    - name: string specifying which dataset to load. Can be `mnist`,
      `cifar10`, `cifar100`.
    - batch_size: how many samples per batch to load.
    - valid_size: percentage split of the training set used for
      the validation set. Should be a float in the range [0, 1].
      In the paper, this number is set to 0.1.
    - shuffle: whether to shuffle the train/validation indices.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.

    Returns
    -------
    - train_loader: training set iterator.
    - valid_loader: validation set iterator.
    """
    # error_msg1 = "[!] valid_size should be in the range [0, 1]."
    # error_msg2 = "[!] Invalid dataset name."
    # assert ((valid_size >= 0) and (valid_size <= 1)), error_msg1
    # assert name in ['mnist', 'cifar10', 'cifar100'], error_msg2

    # # define transforms
    # if name == 'mnist':
    #     normalize = transforms.Normalize(
    #         mean=(0.1307,),
    #         std=(0.3081,)
    #     )
    # else:
    #     normalize = transforms.Normalize(
    #         mean=[0.4914, 0.4822, 0.4465],
    #         std=[0.2023, 0.1994, 0.2010],
    #     )

    # train_trans = transforms.Compose([
    #     transforms.ToTensor(),
    #     normalize,
    # ])
    # valid_trans = transforms.Compose([
    #         transforms.ToTensor(),
    #         normalize,
    # ])

    # load the dataset
    # if name == 'mnist':
    #     train_dataset = datasets.MNIST(
    #         root=data_dir, train=True,
    #         download=True, transform=train_trans,
    #     )
    #     valid_dataset = datasets.MNIST(
    #         root=data_dir, train=True,
    #         download=True, transform=valid_trans,
    #     )
    # elif name == 'cifar10':
    #     train_dataset = datasets.CIFAR10(
    #         root=data_dir, train=True,
    #         download=True, transform=train_trans,
    #     )
    #     valid_dataset = datasets.CIFAR10(
    #         root=data_dir, train=True,
    #         download=True, transform=valid_trans,
    #     )
    # else:
    #     train_dataset = datasets.CIFAR100(
    #         root=data_dir, train=True,
    #         download=True, transform=train_trans,
    #     )
    #     valid_dataset = datasets.CIFAR100(
    #         root=data_dir, train=True,
    #         download=True, transform=valid_trans,
    #     )

    train_dataset = thymidyalte_data(transforms=None)
    valid_dataset = thymidyalte_data(transforms=None)

    # create dataloaders
    num_train = len(train_dataset)
    indices = list(range(num_train))

    train_end = train_dataset.get_train_end()
    train_idx, valid_idx = indices[0:train_end], indices[train_end:]

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, sampler=valid_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    print(train_loader)
    return (train_loader, valid_loader)
