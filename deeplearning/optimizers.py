import numpy as np


# optimizing methods, posted by S.Ruder, suggested by mlfromscratch
# http://ruder.io/optimizing-gradient-descent/index.html
class GradientDescent(object):
    def __init__(self, learning_rate=0.1, momentum=0):
        self.w_update = None
        self.momentum = momentum
        self.learning_rate = learning_rate

    def update(self, W, dW):
        if self.w_update is None:
            self.w_update = np.zeros(np.shape(W))
        self.w_update = self.momentum * self.w_update + (1 - self.momentum) * dW
        return W - self.learning_rate * self.w_update


class Adam(object):
    """
    Adaptive Moment Estimation (Adam) is another method that
    computes adaptive learning rates for each parameter.
    """
    def __init__(self, learning_rate=0.001, b1=0.9, b2=0.999):
        self.learning_rate = learning_rate
        self.eps = 1e-8
        self.m = None
        self.v = None
        # decay rates
        self.b1 = b1
        self.b2 = b2

    def update(self, W, dW):
        if self.m is None:
            self.m = np.zeros(np.shape(dW))
            self.n = np.zeros(np.shape(dW))
        self.m = self.b1 * self.m + (1 - self.b1) * dW
        self.n = self.b2 * self.v + (1 - self.b2) * np.power(dW, 2)

        m_hat = self.m / (1 - self.b1)
        v_hat = self.v / (1 - self.b2)

        w_update = self.learning_rate * m_hat / (np.sqrt(v_hat) + self.eps)
        return W - w_update


