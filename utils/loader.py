from torchvision import transforms
import torchvision
import torch
import logging

from ..datasets import mnist
from ..datasets import convex
from ..datasets import cifar
from ..datasets.data import Dataset


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

def torch_vision_load_cifar10(is_aug):
    torch_cifar10_root = 'datasets'
    if is_aug == 1:
        logging.debug('---use data augmentation---')
        train_transform_cifar10 = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    else:
        logging.debug('---do not use data augmentation---')
        train_transform_cifar10 = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    test_transform_cifar10 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    train_dataset = torchvision.datasets.CIFAR10(root=torch_cifar10_root, train=True, transform=train_transform_cifar10)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_dataset = torchvision.datasets.CIFAR10(root=torch_cifar10_root, train=False, transform=test_transform_cifar10)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False)
    return (train_loader, test_loader)