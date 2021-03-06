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

def batch_iterator(data, labels=None, batch_size=64):
    total = data.shape[0]
    for i in np.arange(0, total, batch_size):
        begin, end = i, min(i + batch_size, total)
        if labels is not None:
            yield data[begin:end], labels[begin:end]
        else:
            yield data[begin:end]

def indices_to_one_hot(data, class_count):
    targets = np.array(data).reshape(-1)
    return np.eye(class_count)[targets]

