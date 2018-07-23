from __future__ import print_function, division
from manipulate_data import augment_1s_col
from active_functions import ReLU, Softmax, TanH, Sigmoid, LeakyReLU
from img_proc import image_to_column, column_to_image, determine_padding
import numpy as np
import scipy
import copy
import math


class Layer(object):
    def init_param(self, optimizer):
        self.optimizer = optimizer

    def name(self):
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
        return 0


class DenseLayer(Layer):
    def __init__(self, n_units, input_shape=None):
        self.input = None
        self.input_shape = input_shape
        self.n_units = n_units
        self.trainable = True
        self.W = None
        self.b = None

    def init_param(self, optimizer):
        param_size = self.input_shape[-1]
        limit = 1 / math.sqrt(param_size)
        self.W = np.random.uniform(-limit, limit, (param_size, self.n_units))
        self.b = np.random.uniform(-limit, limit, (self.n_units))
        self.w_optimizer = copy.copy(optimizer)
        self.b_optimiser = copy.copy(optimizer)

    def parameter_size(self):
        return np.prod(self.W.shape)

    def forward_pass(self, data, training=True):
        self.input = data
        return np.dot(self.input, self.W) + self.b

    def backward_pass(self, accum_grad):
        W = self.W
        if self.trainable:
            dW = np.dot(self.input.T, accum_grad)
            db = np.sum(accum_grad, axis=0, keepdims=True)
            self.W = self.w_optimizer.update(self.W, dW)
            self.b = self.b_optimiser.update(self.b, db)
        accum_grad = np.dot(accum_grad, W.T)
        return accum_grad

    def output_shape(self):
        return (self.n_units,)


ACTIVE_FUNCTIONS = {
    'relu': ReLU,
    'tanh': TanH,
    'softmax': Softmax,
    'sigmoid': Sigmoid,
    'leaky_relu': LeakyReLU,
}


class ActiveLayer(Layer):
    def __init__(self, active_type):
        self.active_type = active_type
        self.active_func = ACTIVE_FUNCTIONS[active_type]()

    def layer_type(self):
        return 'ActiveLayer(%s)' % self.active_type

    def forward_pass(self, data, trainning=True):
        self.input = data
        return self.active_func(data)

    def backward_pass(self, accum_grad):
        return accum_grad * self.active_func.gradient(self.input)

    def output_shape(self):
        return self.input_shape


class DropoutLayer(Layer):
    def __init__(self, p=0.2):
        self.p = p
        self.mask = None
        self.input_shape = None

    def layer_type(self):
        return 'Dropout(%f)' % self.p

    def forward_pass(self, data, training=True):
        c = 1 - self.p
        if training:
            self.mask = np.random.uniform(size=data.shape) > self.p
            c = self.mask
        return data * c

    def backward_pass(self, accum_grad):
        return accum_grad * self.mask

    def output_shape(self):
        return self.input_shape


def padding_size(filter_shape, padding='same'):
    if padding == 'valid':
        return (0, 0), (0, 0)
    if padding == 'same':
        filter_w, filter_h = filter_shape
        pad_w1 = int(math.floor((filter_w - 1) / 2))
        pad_w2 = int(math.ceil((filter_w - 1) / 2))
        pad_h1 = int(math.floor((filter_h - 1) / 2))
        pad_h2 = int(math.ceil((filter_h - 1) / 2))
        return (pad_w1, pad_w2), (pad_h1, pad_h2)


def conv2d(image, kernel, padding):
    return scipy.convolve2d(image, kernel, padding)


class Conv2DLayer(Layer):
    def __init__(self, n_filters, filter_shape, input_shape=None, padding='same', stride=1):
        # input shape should be [batch, channel, width, height]
        print("input shape", input_shape)
        assert(padding in ['same', 'valid'])
        filter_w, filter_h = filter_shape
        assert(filter_w == filter_h)
        self.n_filters = n_filters
        self.filter_shape = filter_shape
        self.input_shape = input_shape
        self.padding = padding
        self.stride = stride
        self.trainable = True

    def init_param(self, optimizer):
        self.w_optimizer = copy.copy(optimizer)
        self.b_optimizer = copy.copy(optimizer)
        filter_w, filter_h = self.filter_shape
        channels = self.input_shape[0]
        limit = 1 / math.sqrt(np.prod(self.filter_shape))
        # why init params like this?
        # http://cs231n.github.io/neural-networks-2/#init
        self.W = np.random.uniform(-limit, limit, size=(self.n_filters, channels, filter_h, filter_w))
        self.b = np.zeros((self.n_filters, 1))

    def parameter_size(self):
        return np.prod(self.W.shape) + np.prod(self.b.shape)

    def output_shape(self):
        channels, h, w = self.input_shape
        pad_h, pad_w = determine_padding(self.filter_shape, self.padding)
        output_h = (h + np.sum(pad_h) - self.filter_shape[0]) / self.stride + 1
        output_w = (w + np.sum(pad_w) - self.filter_shape[0]) / self.stride + 1

        return self.n_filters, int(output_h), int(output_w)

    def forward_pass(self, X, training=True):
        # convolution 2d by reference
        # https://wiseodd.github.io/techblog/2016/07/16/convnet-conv-layer/
        batch_size, channels, height, width = X.shape
        self.layer_input = X
        self.X_col = image_to_column(X, self.filter_shape, self.stride, self.padding)
        self.W_col = self.W.reshape((self.n_filters, np.prod(self.filter_shape)))
        output = self.W_col.dot(self.X_col) + self.b
        # Reshape into (n_filters, out_height, out_width, batch_size)
        output = output.reshape(self.output_shape() + (batch_size, ))
        #output = conv2d(X,
        return output.transpose(3, 0, 1, 2)

    def backward_pass(self, accum_grad):
        accum_grad  = accum_grad.transpose(1, 2, 3, 0).reshape(self.n_filters, -1)
        if self.trainable:
            grad_w = accum_grad.dot(self.X_col.T).reshape(self.W.shape)
            grad_b = np.sum(accum_grad, axis=1, keepdims=True)
            self.W = self.w_optimizer.update(self.W, grad_w)
            self.b = self.b_optimizer.update(self.b, grad_b)

        accum_grad = self.W_col.T.dot(accum_grad)
        accum_grad = column_to_image(accum_grad, self.layer_input.shape,
                                     self.filter_shape, self.stride, self.padding)

        return accum_grad


class BatchNormLayer(Layer):
    """
    batch normalization
    http://dengyujun.com/2017/09/30/understanding-batch-norm/
    """
    pass


class FlattenLayer(Layer):
    """
    convert multi-dimension matrix to 2-dimension matrix
    """
    def __init__(self, input_shape=None):
        self.prev_shape = None
        self.trainable = True
        self.input_shape = input_shape

    def forward_pass(self, X, training=True):
        self.prev_shape = X.shape
        return X.reshape((X.shape[0], -1))

    def backward_pass(self, accum_grad):
        return accum_grad.reshape(self.prev_shape)

    def output_shape(self):
        return (np.prod(self.input_shape),)


class ArgmaxLayer(Layer):
    pass
