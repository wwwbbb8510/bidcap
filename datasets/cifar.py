from .data import BatchDataset
import os


class CIFAR10(BatchDataset):
    """
    CIFAR10 dataset
    """
    def __init__(self, dataset_folder='datasets', image_shape=(3, 32, 32), **kwargs):
        super(CIFAR10, self).__init__(**kwargs)

        self.image_shape = image_shape
        self.train_path = os.path.join(dataset_folder, 'cifar-10',
                                       'data_batch') \
            if self.train_path is None else self.train_path
        self.test_path = os.path.join(dataset_folder, 'cifar-10',
                                      'test_batch') \
            if self.test_path is None else self.test_path
