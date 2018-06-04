from __future__ import print_function

import _pickle as pickle
import argparse
import math
import numpy as np
from active_functions import Softmax
from load_mnist import load_dataset, augment_1s_col
from optimisors import SDG


class SoftmaxClassifier(object):
    def __init__(self, learning_rate=0.05):
        self.param = None

    def _init_param(self, X):
        # X shape (w*h,) 10 is the labels posibilitys
        # self.param_shape = (X.shape[0] + 1, 10)
        self.param_shape = (X.shape[0], 10)
        limit = 1 / math.sqrt(self.param_shape[0])
        self.params = np.random.uniform(-limit, limit, self.param_shape)
        print('init params shape %s, limit %f' % (str(self.params.shape), limit))


    def softmax(self, X, params):
        softmax_func = Softmax()
        return softmax_func(X.dot(params))
        #return softmax_func(np.append(X, 1).dot(params))

    def batch_softmax_loss(self, l2_reg):
        def loss_func(data, labels, params):
            """
            data of shape (a, w)
            labels shape (a,) value in range[0, t-1]
            """
            # this should be of shape (a, t)
            # patched_data = np.apply_along_axis(lambda X: np.append(X, 1), -1, data)
            softmax_array = np.apply_along_axis(lambda X: self.softmax(X, params), -1, data)

            #print(softmax_array)
            # print (patched_data.shape, params.shape, softmax_array.shape)

            n_data = data.shape[0]
            neg_log_probs = -np.log(softmax_array[range(n_data), labels])
             #print(neg_log_probs, labels)
            loss = np.sum(neg_log_probs) / n_data \
                + l2_reg * np.sum(params * params) / 2  # L2 regulation

            d_param = softmax_array
            # minus 1 for each correct probability
            d_param[range(n_data), labels] -= 1
            d_param = np.dot(data.T, d_param) / n_data
            # print(d_param.shape)
            d_param += params * l2_reg

            return loss, d_param
        return loss_func

    def train(self, train_set, val_set, mini_batch=32,
              max_epoch=10000, learning_rate=0.05):
        data, labels = augment_1s_col(train_set[0]), train_set[1]
        self._init_param(data[0])
        self.loss = None
        epoch = 0
        done = False
        iter = 0

        train_num = train_set[0].shape[0]
        val_num = val_set[0].shape[0]

        patience = 5000
        patience_increase = 2
        improve_threshold = 0.995
        validation_freq = min(val_num, patience // 2)

        train_data, train_labels = augment_1s_col(train_set[0]), train_set[1]
        val_data, val_labels = augment_1s_col(val_set[0]), val_set[1]

        params = self.params.copy()
        best_val_loss = np.inf
        loss_func = self.batch_softmax_loss(0.0000)
        print('start training..')
        while epoch < max_epoch and not done:
            epoch += 1
            # compute gradient
            loss, params = SDG(train_data, train_labels, params,
                               loss_func, mini_batch, learning_rate)
            iter += 1
            if iter % 100 == 0:
                print('epoch %d iter %d loss %f' % (epoch, iter, loss))
            if (iter + 1) % validation_freq == 0:
                val_loss = self.validate(val_data, val_labels, params)
                print('epoch %d iter %d validate model, get validation loss %f,'
                      'best so far is %f' % (epoch, iter, val_loss, best_val_loss))
                if val_loss < best_val_loss:
                    if val_loss < best_val_loss * improve_threshold:
                        patience = max(patience, iter * patience_increase)
                        print("update patience to %d" % patience)
                    best_val_loss = val_loss

                self.params = params

            if patience < iter:
                print('iter %d patience %d, break' % (iter, patience))
                done = True
        print('finish training...')

    def validate(self, val_data, val_labels, params):
        predict_labels = self.evaluate(val_data, params)
        return np.mean(np.not_equal(predict_labels, val_labels))

    def evaluate(self, data, params):
        # patched_data = np.apply_along_axis(lambda X: np.append(X, 1), -1, data)
        # return np.dot(patched_data, params).argmax(axis=-1)
        return np.dot(data, params).argmax(axis=-1)

    def predict(self, test_data):
        return self.evaluate(test_data, self.params)

    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    def load(self, filepath):
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
            self.params = model.params


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--max_epoch', default=10000, type=int)
    argparser.add_argument('--batch_size', default=32, type=int)
    argparser.add_argument('--type', default='train', type=str)
    argparser.add_argument('--param_path', default='classifier.pkl', type=str)

    args = argparser.parse_args()
    np.random.seed(101)

    train_set, val_set, test_set = load_dataset()
    print('data loaded...')
    classifier = SoftmaxClassifier()
    if args.type == 'train':
        classifier.train(train_set, val_set, args.batch_size, args.max_epoch)
        classifier.save(args.param_path)
    else:
        classifier.load(args.param_path)
        classifier.predict(test_set)
