# multi class adaboost aka SAMME
# ref: https://web.stanford.edu/~hastie/Papers/samme.pdf
from softmax_classifier import SoftmaxClassifier

class WeakClassifier():
    def __init__(self, n_label=2):
        self.n_label = n_label
        self.softmax = SoftmaxClassifier()

    def train(self, train_set):
        return self.softmax.simple_train(train_set[0], train_set[1])

    def predict(self, data):
        return self.softmax.predict(data)


class AdaBoost():
    def __init__(self, clf=None, n_clf=10):
        self.clf = clf
        self.n_clf = n_clf

    def train(self, train_set):
        for _ in range(self.n_clf):
            pass
