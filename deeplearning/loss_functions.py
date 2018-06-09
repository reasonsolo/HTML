import numpy as np


class Loss():
    def __call__(self, truth, predict):
        raise NotImplementedError()

    def gradient(self, truth, predict):
        raise NotImplementedError()

class SquareLoss(Loss):
    def __call__(self, truth, predict):
        return 0.5 * np.power(truth - predict, 2)

    def gradient(self, truth, predict):
        return truth - predict

class CrossEntropyLoss(Loss):
    def __call__(self, y, p):
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return - y * np.log(p) - (1 - y) * np.log(1 - p)

    def gradient(self, y, p):
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return - (y / p) + (1 - y) / (1 - p)
