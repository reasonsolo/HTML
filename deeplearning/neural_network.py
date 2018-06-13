import numpy as np
from manipulate_data import batch_iterator
from manipulate_data import augment_1s_col

class NeuralNetwork():
    def __init__(self, optimizer, loss, val_set=None):
        self.optimizer = optimizer
        self.layers = []
        self.errors = {'training': [], 'validation': []}
        self.loss_func = loss
        self.val_set = val_set

    def set_trainable(self, trainable):
        for layer in self.layers:
            layer.set_trainable = trainable

    def add(self, layer):
        if len(self.layers) > 0:
            layer.set_input_shape(self.layers[-1].output_shape())
        layer.init_param(optimizer=self.optimizer)
        self.layers.append(layer)

    def train_with_batch(self, data, labels):
        predicts = self._forward_pass(data)
        loss = np.mean(self.loss_func(labels, predicts))
        loss_grad = self.loss_func.gradient(labels, predicts)
        self._backward_pass(loss_grad=loss_grad)
        return loss, loss_grad

    def test_with_batch(self, data, labels):
        predicts = self._forward_pass(data, training=False)
        loss = np.mean(self.loss_func(labels, predicts))
        return loss

    def train(self, train_set, train_labels, epochs=1000, batch_size=64):
        for i in range(epochs):
            batch_error = []
            for batch, labels in batch_iterator(train_set, train_labels, batch_size):
                loss = self.train_with_batch(batch, labels)
                batch_error.append(loss)
            self.errors['training'].append(batch_error)

            val_loss = None
            if self.val_set is not None:
                val_loss = self.test_with_batch(self.val_set[0], self.val_set[1])
                self.errors['validation'].append(val_loss)
            print("epoch %d, validation loss %s" % (i, str(np.mean(val_loss))))

    def predict(self, data):
        return self._forward_pass(data, training=False)

    def _forward_pass(self, data, training=True):
        layer_input = data
        for layer in self.layers:
            layer_input = layer.forward_pass(layer_input, training)
        return layer_input

    def _backward_pass(self, loss_grad):
        for layer in reversed(self.layers):
            loss_grad = layer.backward_pass(loss_grad)
        return loss_grad

    def summary(self):
        total_params = 0
        for layer in self.layers:
            layer_summary = [layer.layer_type(), layer.parameter_size(), layer.output_shape()]
            total_params += layer.parameter_size()
            print(layer_summary)
        print('total params %d' % total_params)
