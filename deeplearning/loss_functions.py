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
