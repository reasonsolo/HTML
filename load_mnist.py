import _pickle as pickle
import gzip


def load_dataset(path='data/mnist.pkl.gz'):
    """
    return train_set, valid_set, test_set
    """
    f = gzip.open(path, 'rb')
    return pickle.load(f, encoding='latin1')
