import torch
import torchvision
from torchvision.transforms import (
    Compose, Resize, Lambda, ToTensor, Normalize, Grayscale)

from PIL import Image
import numpy as np


class BinaryICLDataset(object):
    def __init__(self, train_set_0, train_set_1, test_set, transform, is_mnist=True):
        # is_mnist should be true for all mnist families
        data_list = []
        targets_list = []
        self.transform = transform

        for index in range(len(test_set)):
            raw_data, _ = test_set[index]
            test_label = test_set.targets[index]
            transformed_test_data = self.transform(raw_data)
            if is_mnist:
                transformed_train_data_0 = self.transform(
                    Image.fromarray(train_set_0[index].numpy(), mode="L"))
                transformed_train_data_1 = self.transform(
                    Image.fromarray(train_set_1[index].numpy(), mode="L"))
            else:
                transformed_train_data_0 = self.transform(
                    Image.fromarray(train_set_0[index]))
                transformed_train_data_1 = self.transform(
                    Image.fromarray(train_set_1[index]))
            transformed_data = torch.stack(
                [transformed_train_data_0, transformed_train_data_1,
                 transformed_test_data], dim=0)  # always in this order, OK for testing. This has to be consistent with _icl_input_labels in main
            data_list.append(transformed_data)
            if isinstance(test_label, int):
                test_label = torch.tensor(test_label)
            targets_list.append(test_label)

        self.data = torch.stack(data_list, dim=0)
        self.targets = torch.stack(targets_list, dim=0)

    def __getitem__(self, index):
        return (self.data[index], self.targets[index])
    
    def __len__(self):
        return len(self.targets)


def get_icl_dataset(dataset_name, data_dir, eval_batch_size, num_workers,
                    binary_class_choice="01", grey_scale=False):
    if dataset_name.lower() in ["mnist", "fmnist"]:
        return get_icl_mnist(
            dataset_name, data_dir, eval_batch_size, num_workers,
            binary_class_choice, grey_scale)
    elif dataset_name.lower() == "cifar10":
        return get_icl_cifar10(
            data_dir, eval_batch_size, num_workers, binary_class_choice,
            grey_scale)
    else:
        assert False, f"Unknown dataset name: {dataset_name}"


def get_icl_mnist(dataset_name, data_dir, eval_batch_size, num_workers,
                  binary_class_choice="01", grey_scale=False):
    if dataset_name == "mnist":
        norm_params = {'mean': [0.1307], 'std': [0.3081]}
        _dataset_cls = torchvision.datasets.MNIST
    else:
        norm_params = {'mean': [0.286], 'std': [0.353]}
        _dataset_cls = torchvision.datasets.FashionMNIST

    if grey_scale:
        transform = Compose([Resize(32), ToTensor(), Normalize(**norm_params)])
    else:
        transform = Compose(
            [Resize(32), ToTensor(), Normalize(**norm_params),
            Lambda(lambda x: x.repeat(3, 1, 1))])

    train_dataset = _dataset_cls(
        download=True, root=data_dir, train=True)

    test_dataset = _dataset_cls(
        download=True, root=data_dir, train=False)

    class_list = list(binary_class_choice)
    class_0 = int(class_list[0])
    class_1 = int(class_list[1])
    assert 0 <= class_0 <= 9
    assert 0 <= class_1 <= 9

    # mnist train
    idx_0 = train_dataset.targets == class_0
    idx_1 = train_dataset.targets == class_1

    train_data_0 = train_dataset.data[idx_0]
    train_data_1 = train_dataset.data[idx_1]

    # mnist test
    idx_0 = test_dataset.targets == class_0
    idx_1 = test_dataset.targets == class_1

    test_data_0 = test_dataset.data[idx_0]
    test_data_1 = test_dataset.data[idx_1]

    test_targets_0 = (
        test_dataset.targets[idx_0].int() - test_dataset.targets[idx_0].int() - 1)
    test_targets_1 = (
        test_dataset.targets[idx_1].int() - test_dataset.targets[idx_1].int() + 1)

    test_dataset.targets = torch.cat(
        [test_targets_0, test_targets_1], dim=0)

    test_dataset.data = torch.cat(
        [test_data_0, test_data_1], dim=0)

    # construct ICL dataset
    icl_mnist_binary_dataset = BinaryICLDataset(
        train_data_0, train_data_1, test_dataset, transform=transform)
    
    dataloader = torch.utils.data.DataLoader(
        icl_mnist_binary_dataset, batch_size=eval_batch_size,
        num_workers=num_workers, pin_memory=True)

    return dataloader


def get_icl_cifar10(data_dir, eval_batch_size, num_workers,
                    binary_class_choice="01", grey_scale=False):

    if grey_scale:
        norm_params = {
            'mean': [0.4874], 'std': [0.2506]}
        transform = Compose(
            [Resize(32), Grayscale(num_output_channels=1),
             ToTensor(), Normalize(**norm_params)])
    else:
        norm_params = {
            'mean': [0.4914, 0.4822, 0.4465], 'std': [0.247, 0.243, 0.261]}
        transform = Compose(
            [Resize(32), ToTensor(), Normalize(**norm_params)])

    train_dataset = torchvision.datasets.CIFAR10(
        download=True, root=data_dir, train=True)

    test_dataset = torchvision.datasets.CIFAR10(
        download=True, root=data_dir, train=False)

    class_list = list(binary_class_choice)
    class_0 = int(class_list[0])
    class_1 = int(class_list[1])
    assert 0 <= class_0 <= 9
    assert 0 <= class_1 <= 9

    tmp_targets = torch.ByteTensor(train_dataset.targets)
    idx_0 = tmp_targets == class_0
    idx_1 = tmp_targets == class_1

    train_data_0 = train_dataset.data[idx_0]
    train_data_1 = train_dataset.data[idx_1]

    # test
    tmp_targets = torch.ByteTensor(test_dataset.targets)
    idx_0 = tmp_targets == class_0
    idx_1 = tmp_targets == class_1

    targets_0 = (tmp_targets[idx_0].int() - tmp_targets[idx_0].int() - 1).tolist()
    targets_1 = (tmp_targets[idx_1].int() - tmp_targets[idx_1].int() + 1).tolist()

    test_dataset.targets = targets_0 + targets_1  # list concatenation
    test_dataset.data = np.concatenate(
        (test_dataset.data[idx_0], test_dataset.data[idx_1]))

    # construct ICL dataset
    icl_binary_dataset = BinaryICLDataset(
        train_data_0, train_data_1, test_dataset, transform=transform,
        is_mnist=False)
    
    dataloader = torch.utils.data.DataLoader(
        icl_binary_dataset, batch_size=eval_batch_size,
        num_workers=num_workers, pin_memory=True)

    return dataloader