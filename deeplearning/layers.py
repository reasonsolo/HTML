from __future__ import print_function, division
import numpy as np
from manipulate_data import augment_1s_col


class Layer(object):
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
        self.layer_input = None
        self.input_shape = input_shape
        self.n_units = n_units
        self.trainable = True
        self.params = None

    def _init_params(self, optimizor):
        pass

    def parameter_size(self):
        return np.prod(self.params.shape)

    def forward_pass(self, data, training=True):
        self.layer_input = augment_1s_col(data)
        return np.dot(self.layer_input, self.params)

    def backward_pass(self, accum_grad):
        params = self.params
        if self.trainable:
            pass

