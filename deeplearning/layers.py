from __future__ import print_function, division
from manipulate_data import augment_1s_col
from active_functions import *
import numpy as np
import copy
import math


class Layer(object):
    def init_param(self, optimiser):
        raise NotImplementedError()

    def name():
        return self.__class__.__name__

    def set_trainable(self, trainable=True):
        self.trainable = trainable

    def set_input_shape(self, shape):
        self.input_shape = shape

    def output_shape(self):
        raise NotImplementedError()

    def layer_type(self):
        return self.__class__.__name__

    def forward_pass(self, data, training=True):
        raise NotImplementedError()

    def backward_pass(self):
        raise NotImplementedError()

    def parameter_size(self):
        raise NotImplementedError()


class DenceLayer(Layer):
    def __init__(self, n_units, input_shape=None):
        self.input = None
        self.input_shape = input_shape
        self.n_units = n_units
        self.trainable = True
        self.W = None

    def init_params(self, optimizer):
        param_size = self.input_shape[0] + 1
        limit = 1 / math.sqrt(param_size)
        self.W = np.random.uniform(-limit, limit, (param_size, self.n_units))
        self.optimizer = copy.copy(optimizer)

    def parameter_size(self):
        return np.prod(self.params.shape)

    def forward_pass(self, data, training=True):
        self.input = augment_1s_col(data)
        return np.dot(self.input, self.W)

    def backward_pass(self, accum_grad):
        W = self.W
        if self.trainable:
            dW = np.dot(self.input.T, accum_grad)
            self.W = self.optimizer.update(self.W, dW)
        accum_grad = np.dot(accum_grad, W.T)
        return accum_grad

    def output_shape(self):
        return (self.n_units,)


ACTIVE_FUNCTIONS = {
    'relu': ReLU,
    'tanh': TanH,
    'softmax': Softmax,
    'sigmoid': Sigmoid
}


class ActiveLayer(Layer):
    def __init__(self, active_type):
        self.active_type = active_type
        self.active_func = ACTIVE_FUNCTIONS[active_type]

    def layer_type(self):
        return 'ActiveLayer(%s)' % self.active_type

    def forward_pass(self, data, trainning=True):
        self.input = data
        return self.active_func(data)

    def backward_pass(self, accum_grad):
        return accum_grad * self.active_func.gradient(self.input)

    def output_shape(self):
        return self.input_shape
