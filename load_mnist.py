import _pickle as pickle
import gzip
import numpy as np


def load_dataset(path='data/mnist.pkl.gz'):
    """
    return train_set, valid_set, test_set
    """
    f = gzip.open(path, 'rb')
    return pickle.load(f, encoding='latin1')

def augment_1s_col(dataset):
    return np.hstack((np.ones((dataset.shape[0], 1)), dataset))

