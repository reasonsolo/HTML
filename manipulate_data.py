import _pickle as pickle
import gzip
import numpy as np


def load_mnist(path='data/mnist.pkl.gz'):
    """
    return train_set, valid_set, test_set
    """
    f = gzip.open(path, 'rb')
    return pickle.load(f, encoding='latin1')


def augment_1s_col(dataset):
    return np.hstack((np.ones((dataset.shape[0], 1)), dataset))


def generate_batch_data(data, labels, batch_size):
    batch_indices = np.random.choice(np.arange(data.shape[0]),  # or simply X.shape[0]
                                     batch_size,
                                     replace=False)
    return data[batch_indices, :], labels[batch_indices]
