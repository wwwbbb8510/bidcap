from .data import Dataset
import os


class CONVEX(Dataset):
    """
    Convex dataset
    """
    def __init__(self, dataset_folder='datasets', image_shape=(1, 28, 28), **kwargs):
        super(CONVEX, self).__init__(**kwargs)

        self.image_shape = image_shape
        self.train_path = os.path.join(dataset_folder, 'convex',
                                       'convex_train.amat') \
            if self.train_path is None else self.train_path
        self.test_path = os.path.join(dataset_folder, 'convex', '50k',
                                      'convex_test.amat') \
            if self.test_path is None else self.test_path
