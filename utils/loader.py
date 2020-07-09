from torchvision import transforms
import torchvision
import torch
import logging
from torch.utils.data import DataLoader, SubsetRandomSampler, ConcatDataset
import numpy as np

from ..datasets import mnist
from ..datasets import convex
from ..datasets import cifar
from ..datasets.data import Dataset
from .cutout import cutout, normalize, to_tensor, \
    DEFAULT_CUTOUT_CONFIG_CIFAR10, DEFAULT_CUTOUT_CONFIG_CIFAR100, DEFAULT_CUTOUT_CONFIG_SVHN


class ImagesetLoader(object):
    """
    image dataset loader using multiton to avoid duplicate-loading
    """
    # store the instances for multiton purpose
    _instances = {}
    # dictionary of available dataset class
    _dataset_classes = {
        'mb': mnist.MB,
        'mbi': mnist.MBI,
        'mdrbi': mnist.MDRBI,
        'mrb': mnist.MRB,
        'mrd': mnist.MRD,
        'convex': convex.CONVEX,
        'cifar10': cifar.CIFAR10,
    }

    @staticmethod
    def load(dataset_name, **kwargs):
        """
        load dataset
        :param dataset_name: dataset name
        :type dataset_name: str
        :param kwargs: kwargs passed to dataset object
        :type kwargs: dict
        :return: dataset object
        :rtype: Dataset
        """
        partial_dataset_ratio = kwargs[
            'partial_dataset_ratio'] if 'partial_dataset_ratio' in kwargs else 0
        train_validation_split_point = kwargs[
            'train_validation_split_point'] if 'train_validation_split_point' in kwargs else 0
        instance_key = dataset_name + '_' + str(partial_dataset_ratio) + '_' + str(train_validation_split_point)
        if instance_key not in ImagesetLoader._instances:
            dataset_name = dataset_name.lower()
            if dataset_name not in ImagesetLoader._dataset_classes:
                raise Exception('{} dataset do not exists'.format(dataset_name))
            ImagesetLoader._instances[instance_key] = ImagesetLoader._dataset_classes[dataset_name](**kwargs).load()
        return ImagesetLoader._instances[instance_key]

    @staticmethod
    def dataset_classes():
        return ImagesetLoader._dataset_classes


def torch_vision_load_mnist():
    torch_mnist_root = 'datasets'
    train_dataset = torchvision.datasets.MNIST(root=torch_mnist_root, train=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_dataset = torchvision.datasets.MNIST(root=torch_mnist_root, train=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False)
    return (train_loader, test_loader)


def torch_vision_load_fashion_mnist():
    torch_mnist_root = 'datasets'
    train_dataset = torchvision.datasets.FashionMNIST(root=torch_mnist_root, train=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_dataset = torchvision.datasets.FashionMNIST(root=torch_mnist_root, train=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False)
    return (train_loader, test_loader)


def torch_vision_load_cifar10(is_aug, distributed=False, world_size=None, rank=None,
                              use_cutout=False, cutout_config=DEFAULT_CUTOUT_CONFIG_CIFAR10):
    torch_cifar10_root = 'datasets'
    mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    if is_aug == 1:
        logging.debug('---use data augmentation---')
        if use_cutout:
            train_transform_cifar10 = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                normalize(mean, std),
                cutout(cutout_config['cutout_size'],
                       cutout_config['cutout_prob'],
                       cutout_config['cutout_inside']),
                to_tensor(),
            ])
        else:
            train_transform_cifar10 = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
    else:
        logging.debug('---do not use data augmentation---')
        train_transform_cifar10 = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    test_transform_cifar10 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    train_dataset = torchvision.datasets.CIFAR10(root=torch_cifar10_root, train=True, transform=train_transform_cifar10)
    test_dataset = torchvision.datasets.CIFAR10(root=torch_cifar10_root, train=False, transform=test_transform_cifar10)
    train_sampler = None
    train_shuffle = True
    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank
        )
        train_shuffle = False
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=train_shuffle,
                                               sampler=train_sampler)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False)
    return train_loader, test_loader


def torch_vision_load_cifar100(is_aug, use_cutout=False, cutout_config=DEFAULT_CUTOUT_CONFIG_CIFAR100):
    torch_cifar100_root = 'datasets'
    mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
    if is_aug == 1:
        logging.debug('---use data augmentation---')
        if use_cutout:
            train_transform_cifar100 = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                normalize(mean, std),
                cutout(cutout_config['cutout_size'],
                       cutout_config['cutout_prob'],
                       cutout_config['cutout_inside']),
                to_tensor(),
            ])
        else:
            train_transform_cifar100 = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
    else:
        logging.debug('---do not use data augmentation---')
        train_transform_cifar100 = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    test_transform_cifar100 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    train_dataset = torchvision.datasets.CIFAR100(root=torch_cifar100_root, train=True,
                                                  transform=train_transform_cifar100)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_dataset = torchvision.datasets.CIFAR100(root=torch_cifar100_root, train=False,
                                                 transform=test_transform_cifar100)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False)
    return (train_loader, test_loader)


def torch_vision_load_svhn(is_aug, with_extra=True,
                           use_cutout=False, cutout_config=DEFAULT_CUTOUT_CONFIG_SVHN):
    torch_svhn_root = 'datasets/svhn'
    mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    if is_aug == 1:
        logging.debug('---use data augmentation---')
        if use_cutout:
            train_transform_svhn = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                normalize(mean, std),
                cutout(cutout_config['cutout_size'],
                       cutout_config['cutout_prob'],
                       cutout_config['cutout_inside']),
                to_tensor(),
            ])
        else:
            train_transform_svhn = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
    else:
        logging.debug('---do not use data augmentation---')
        train_transform_svhn = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    test_transform_svhn = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    train_dataset = torchvision.datasets.SVHN(root=torch_svhn_root, split='train',
                                              transform=train_transform_svhn)
    if with_extra:
        extra_dataset = torchvision.datasets.SVHN(root=torch_svhn_root, split='extra',
                                                  transform=train_transform_svhn)
        train_dataset = ConcatDataset([train_dataset, extra_dataset])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_dataset = torchvision.datasets.SVHN(root=torch_svhn_root, split='test',
                                             transform=test_transform_svhn)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False)
    return (train_loader, test_loader)


def torch_vision_load_stl10(is_aug):
    torch_stl10_root = 'datasets'
    if is_aug == 1:
        logging.debug('---use data augmentation---')
        train_transform_stl10 = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.3, 0.3, 0.3)),
        ])
    else:
        logging.debug('---do not use data augmentation---')
        train_transform_stl10 = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.3, 0.3, 0.3)),
        ])
    test_transform_stl10 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.3, 0.3, 0.3)),
    ])
    train_dataset = torchvision.datasets.STL10(root=torch_stl10_root, train=True,
                                               transform=train_transform_stl10)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_dataset = torchvision.datasets.STL10(root=torch_stl10_root, train=False,
                                              transform=test_transform_stl10)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False)
    return (train_loader, test_loader)


def torch_vision_load_imagenet(is_aug, download=False, distributed=False, world_size=None, rank=None):
    torch_imagenet_root = 'datasets/imagenet/ILSVRC2012'
    if is_aug == 1:
        logging.debug('---use data augmentation---')
        train_transform_imagenet = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        logging.debug('---do not use data augmentation---')
        train_transform_imagenet = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    test_transform_imagenet = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train_dataset = torchvision.datasets.ImageNet(root=torch_imagenet_root, split='train',
                                                  transform=train_transform_imagenet, download=download)
    test_dataset = torchvision.datasets.ImageNet(root=torch_imagenet_root, split='val',
                                                 transform=test_transform_imagenet, download=download)
    train_sampler = None
    train_shuffle = True
    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank
        )
        train_shuffle = False
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=train_shuffle,
                                               sampler=train_sampler)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False)
    return train_loader, test_loader


def torch_vision_split_cifar10(data_loader, split_point, random_seed=1, batch_sizes=None):
    """
    split one data loader to two
    :param data_loader: pytorch data loader
    :type data_loader: DataLoader
    :param split_point: split point where the data is split into two
    :type split_point: int
    :return: split data loaders
    """
    num_data = len(data_loader.dataset)
    indices = list(range(num_data))
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    first_batch_size, second_batch_size = batch_sizes if type(batch_sizes) == tuple and len(batch_sizes) == 2 else (
        data_loader.batch_size, data_loader.batch_size)
    first_set_idx, second_set_idx = indices[:split_point], indices[split_point:]
    first_dataset = torch.utils.data.Subset(data_loader.dataset, first_set_idx)
    second_dataset = torch.utils.data.Subset(data_loader.dataset, second_set_idx)
    first_set_loader = torch.utils.data.DataLoader(first_dataset, batch_size=first_batch_size, shuffle=True)
    second_set_loader = torch.utils.data.DataLoader(second_dataset, batch_size=second_batch_size, shuffle=True)
    return first_set_loader, second_set_loader
