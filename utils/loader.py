from ..datasets import mnist
from ..datasets import convex
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
            'train_validation_split_point'] if 'train_validation_split_point' in kwargs else 0
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
