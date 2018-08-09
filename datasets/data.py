import numpy as np
import logging
import os


class Dataset(object):
    def __init__(self, train_path=None, test_path=None, image_shape=None, train_validation_split_point=None, mode=None,
                 partial_dataset_ratio=None):
        """
        init Dataset object
        :param train_path: the path where the training dataset reside
        :type train_path: str
        :param test_path: the path where the test dataset reside
        :type test_path: str
        :param image_shape: the image shape
        :type image_shape: tuple
        :param train_validation_split_point: the point where the training set is split into training and validation sets
        :type train_validation_split_point: int
        :param mode: running mode - 1:production, 0/None: debug
        :type mode: int
        :param partial_dataset_ratio: The percentage of the partial dataset loaded
        :type partial_dataset_ratio: float
        """
        super(Dataset, self).__init__()
        # properties set by arguments
        self._train_path = train_path
        self._test_path = test_path
        self._image_shape = image_shape
        self._train_validation_split_point = train_validation_split_point
        self._mode = mode
        self._partial_dataset_ratio = partial_dataset_ratio

        # properties default value define
        self._train = None
        self._validation = None
        self._test = None

    def load(self):
        if self.train_path is not None:
            # only load training data from file once
            if self.train is None:
                self._train = self._load_txt_image_data_with_label_at_end(self.train_path)

        # only split the training data once
        if self.validation is None and self.train is not None \
                and self.train_validation_split_point is not None and self.train_validation_split_point > 0:
            split_train = {
                'images': self.train['images'][0:self.train_validation_split_point, :, :, :],
                'labels': self.train['labels'][0:self.train_validation_split_point]
            }
            split_validation = {
                'images': self.train['images'][self.train_validation_split_point:, :, :, :],
                'labels': self.train['labels'][self.train_validation_split_point:]
            }
            self._train = split_train
            self._validation = split_validation

        if self.test_path is not None:
            # only load test data from file once
            if self._test is None:
                self._test = self._load_txt_image_data_with_label_at_end(self.test_path)

        logging.debug('Loaded Dataset:{}'.format(self))
        return self

    def _load_txt_image_data_with_label_at_end(self, path):
        """
        load text image data with label at the end of each row
        :param path: the path where the dataset resides
        :type path: str
        :return:
        """
        data = np.loadtxt(path)
        if self.mode is not None and self.mode == 0:
            data = data[0:1000, :]
        elif self.partial_dataset_ratio is not None and 0.0 < self.partial_dataset_ratio < 1.0:
            # randomly pick partial dataset
            cut_point = int(data.shape[0] * self.partial_dataset_ratio)
            indices = np.random.permutation(data.shape[0])
            training_idx = indices[:cut_point]
            data = data[training_idx, :]
        images = data[:, 0:-1]
        labels = data[:, -1]
        images = np.reshape(images, (-1,) + self.image_shape)

        return {
            'images': images,
            'labels': labels
        }

    def __repr__(self):
        str_repr = super(Dataset, self).__repr__() + os.linesep
        str_repr += 'Training data shape:{}'.format(
            self.train['images'].shape) if self.train is not None else 'Training data is empty'
        str_repr += os.linesep
        str_repr += 'Validation data shape:{}'.format(
            self.validation['images'].shape) if self.validation is not None else 'Validation data is empty'
        str_repr += os.linesep
        str_repr += 'Test data shape:{}'.format(
            self.test['images'].shape) if self.test is not None else 'Test data is empty'
        str_repr += os.linesep
        return str_repr

    @property
    def train(self):
        return self._train

    @property
    def validation(self):
        return self._validation

    @property
    def test(self):
        return self._test

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, mode):
        self._mode = mode

    @property
    def partial_dataset_ratio(self):
        return self._partial_dataset_ratio

    @partial_dataset_ratio.setter
    def partial_dataset_ratio(self, partial_dataset_ratio):
        self._partial_dataset_ratio = partial_dataset_ratio

    @property
    def train_path(self):
        return self._train_path

    @train_path.setter
    def train_path(self, train_path):
        self._train_path = train_path

    @property
    def test_path(self):
        return self._test_path

    @test_path.setter
    def test_path(self, test_path):
        self._test_path = test_path

    @property
    def image_shape(self):
        return self._image_shape

    @image_shape.setter
    def image_shape(self, image_shape):
        self._image_shape = image_shape

    @property
    def train_validation_split_point(self):
        return self._train_validation_split_point

    @train_validation_split_point.setter
    def train_validation_split_point(self, train_validation_split_point):
        self._train_validation_split_point = train_validation_split_point
