import numpy as np


class Sigmoid():
    def __call__(self, x):
        return 1 / (1 + np.exp(x))

    def gradient(self, x):
        y = self.__call__(x)
        return y * (1 - y)


class Softmax():
    def __call__(self, x):
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)

    def gradient(self, x):
        v = self.__call__(x)
        return v * (1 - v)


class TanH():
    def __call__(self, x):
        return 2 / (1 + np.exp(-2 * x)) - 1

    def gradient(self, x):
        return 1 - np.power(self.__call__(x), 2)


class ReLU():
    def __call__(self, x):
        zeros = np.zeros(x.shape)
        return np.where(x >= 0, x, zeros)

    def gradient(self, x):
        ones = np.ones(x.shape)
        zeros = np.zeros(x.shape)
        return np.where(x >= 0, ones, zeros)


class LeakyReLU():
    def __init__(self, alpha=0.2):
        self.alpha = alpha

    def __call__(self, x):
        return np.where(x >= 0, x, self.alpha * x)

    def gradient(self, x):
        return np.where(x >= 0, 1, self.alpha)
