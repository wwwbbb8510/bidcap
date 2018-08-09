from .data import Dataset
import os


class MB(Dataset):
    """
    MNIST basic
    """

    def __init__(self, dataset_folder='datasets', **kwargs):
        super(MB, self).__init__(**kwargs)
        self.image_shape = (28, 28)
        self.train_path = os.path.join(dataset_folder, 'mnist',
                                       'mnist_train.amat') \
            if self.train_path is None else self.train_path
        self.test_path = os.path.join(dataset_folder, 'mnist',
                                      'mnist_train.amat') \
            if self.test_path is None else self.test_path


class MBI(Dataset):
    """
    MNIST with background images
    """

    def __init__(self, dataset_folder='datasets', **kwargs):
        super(MBI, self).__init__(**kwargs)
        self.image_shape = (28, 28)
        self.train_path = os.path.join(dataset_folder, 'mnist_background_images',
                                       'mnist_background_images_train.amat') \
            if self.train_path is None else self.train_path
        self.test_path = os.path.join(dataset_folder, 'mnist_background_images',
                                      'mnist_background_images_test.amat') \
            if self.test_path is None else self.test_path


class MDRBI(Dataset):
    """
    MNIST digits with rotation and background images
    """

    def __init__(self, dataset_folder='datasets', **kwargs):
        super(MDRBI, self).__init__(**kwargs)
        self.image_shape = (28, 28)
        self.train_path = os.path.join(dataset_folder, 'mnist_rotation_back_image_new',
                                       'mnist_all_background_images_rotation_normalized_train_valid.amat') \
            if self.train_path is None else self.train_path
        self.test_path = os.path.join(dataset_folder, 'mnist_rotation_back_image_new',
                                      'mnist_all_background_images_rotation_normalized_test.amat') \
            if self.test_path is None else self.test_path


class MRB(Dataset):
    """
    MNIST with random background
    """

    def __init__(self, dataset_folder='datasets', **kwargs):
        super(MRB, self).__init__(**kwargs)

        self.image_shape = (28, 28)
        self.train_path = os.path.join(dataset_folder, 'mnist_background_random',
                                       'mnist_background_random_train.amat') \
            if self.train_path is None else self.train_path
        self.test_path = os.path.join(dataset_folder, 'mnist_background_random',
                                      'mnist_background_random_test.amat') if self.test_path is None else self.test_path


class MRD(Dataset):
    """
    MNIST with rotated digits
    """

    def __init__(self, dataset_folder='datasets', **kwargs):
        super(MRD, self).__init__(**kwargs)

        self.image_shape = (28, 28)
        self.train_path = os.path.join(dataset_folder, 'mnist_rotation_new',
                                       'mnist_all_rotation_normalized_float_train_valid.amat') \
            if self.train_path is None else self.train_path
        self.test_path = os.path.join(dataset_folder, 'mnist_rotation_new',
                                      'mnist_all_rotation_normalized_float_test.amat') \
            if self.test_path is None else self.test_path
