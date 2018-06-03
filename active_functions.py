import numpy as np


class Sigmoid():
    def __call__(self, x):
        return 1 / (1 + np.exp(x))

    def gradient(self, x):
        y = self.__call__(x)
        return y * (1 - y)


class Softmax():
    def __call__(self, x):
        e_x = np.exp(x - np.max(x, axis=-1))  # keepdims=True
        return e_x / np.sum(e_x, axis=-1)

    def gradient(self, x):
        v = self.__call__(x)
        return v * (1 - v)


class TanH():
    def __call__(self, x):
        return 2 / (1 + np.exp(-2 * x)) - 1

    def gradient(self, x):
        return 1 - np.power(self.__call__(x), 2)
