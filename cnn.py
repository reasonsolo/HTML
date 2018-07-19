from deeplearning.neural_network import NeuralNetwork
from deeplearning.layers import DenseLayer, ActiveLayer, Conv2DLayer, FlattenLayer
from deeplearning.optimizers import Adam
from deeplearning.loss_functions import CrossEntropyLoss
from manipulate_data import load_mnist, augment_1s_col, indices_to_one_hot


def main():
    optimiser = Adam()
    loss = CrossEntropyLoss()
    net = NeuralNetwork(optimiser, loss)
    net.set_trainable(True)

    img_size = 28
    net.add(Conv2DLayer(n_filters=32, filter_shape=(3,3),
                        stride=1, input_shape=(1, img_size, img_size),
                        padding='same'))
    net.add(ActiveLayer('relu'))
    net.add(FlattenLayer(10))
    net.add(DenseLayer(32))
    net.add(DenseLayer(10))
    net.add(ActiveLayer('softmax'))

    net.summary()
    net.set_trainable(True)

    train_set, val_set, test_set = load_mnist()

    X_train, Y_train = train_set
    X_test, Y_test = test_set
    print (X_train.shape)
    X_train = X_train.reshape((X_train.shape[0], 1, img_size, img_size))
    X_test = X_test.reshape((X_test.shape[0], 1, img_size, img_size))
    Y_train = indices_to_one_hot(Y_train, 10)

    print (X_train.shape)
    print (Y_train.shape)
    net.train(X_train, Y_train, epochs=50)
    result = net.predict(X_test)
    Y_predict = np.argmax(result, axis=1)
    correct = np.sum(Y_predict == Y_test)

    print("test set %d, correct %d, accuracy %f" % (Y_test.shape[0], correct,
                                                    correct / Y_test.shape[0]))


if __name__ == '__main__':
    main()
