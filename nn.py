from layers import Layer
from activations import ReLU, Sigmoid
from losses import CrossEntropy
from utils.data_generator import generator

import numpy as np
import pickle


class NeuralNetwork:
    def __init__(self, batch=-1):
        self.layers = []
        self.mean = None
        self.std = None
        self.batch = batch
        self.epoch = 0
        self.training_losses = []
        self.test_losses = []
        self.trained = False

    def standardise_input(self, X):
        if self.mean is None:
            assert self.std is None
            self.mean = np.mean(X, axis=0, keepdims=True)
            self.std = np.std(X, axis=0, keepdims=True) + 1e-8
        return (X - self.mean) / self.std

    def add_layer(self, layer, activation=None):
        assert (self.layers == [] or self.layers[-1].output_size == layer.input_size)
        self.layers.append(layer)

        if activation is not None:
            self.layers.append(activation(layer.output_size))

    def forward(self, X, update=True):
        assert (self.layers and X.shape[1] == self.layers[0].input_size)
        value = X
        for layer in self.layers:
            value = layer.forward(value, update)
        return value

    def backward(self, y, eta):
        value = y
        for layer in reversed(self.layers):
            value = layer.backward(value, eta)
        # return value

    def compute(self, X, classification=True):
        value = self.forward(self.standardise_input(X), False)
        print("value:", value)
        if classification:
            value = np.where(value > 0.5, 1, 0)
        return value

    # X, y: ndarray, loss_function: Loss, epoch: gradient descent iterations, eta: learning rate, training_set: proportion of training set, visualisation: whether to record intermediate data
    def train(self, X, y, loss_function, epoch=50, eta=0.01, training_set=1.0, visualisation=False):
        assert (epoch > 0 and eta > 0 and 1 >= training_set > 0 and X.shape[0] == y.shape[0])
        print("Start training")
        X = self.standardise_input(X)
        self.epoch = epoch
        X_train = None
        X_test = None
        y_train = None
        y_test = None
        if training_set < 1:
            n = int(training_set * X.shape[0])
            X_train = X[:n]
            y_train = y[:n]
            X_test = X[n:]
            y_test = y[n:]

        else:
            X_train = X
            y_train = y

        if visualisation:

            if self.batch < 1:
                self.training_losses = []
                self.test_losses = []
                for i in range(epoch):
                    y_hat = self.forward(X_train)
                    loss = loss_function.compute_loss(y_hat, y_train)
                    self.training_losses.append(loss)
                    if (training_set < 1):
                        loss = loss_function.compute_loss(self.forward(X_test, False), y_test)
                        self.test_losses.append(loss)

                    grad = loss_function.gradient(y_hat, y_train)
                    self.backward(grad, eta)
            else:
                self.training_losses = [0 for i in range(epoch)]
                self.test_losses = [0 for i in range(epoch)]
                for i in range(epoch):
                    batches = (X_train.shape[0] + 1) // self.batch
                    for b in range(batches):
                        sli = slice(b * self.batch, b * self.batch + self.batch, 1)
                        y_hat = self.forward(X_train[sli])
                        loss = loss_function.compute_loss(y_hat, y_train[sli])
                        self.training_losses[i] += loss
                        if (training_set < 1):
                            loss = loss_function.compute_loss(self.forward(X_test, False), y_test)
                            self.test_losses[i] += loss

                        grad = loss_function.gradient(y_hat, y_train[sli])
                        self.backward(grad, eta)
        else:
            if self.batch < 1:
                for i in range(epoch):
                    y_hat = self.forward(X_train)
                    grad = loss_function.gradient(y_hat, y_train)
                    self.backward(grad, eta)
            else:
                for i in range(epoch):
                    batches = (X_train.shape[0] + 1) // self.batch
                    for b in range(batches):
                        sli = slice(b * self.batch, b * self.batch + self.batch, 1)
                        y_hat = self.forward(X_train[sli])
                        grad = loss_function.gradient(y_hat, y_train[sli])
                        self.backward(grad, eta)
        print("weights: ", self.layers[0].weights)
        print("bias: ", self.layers[0].biases)
        self.trained = True

    def save(self, file_path):
        with open(file_path, "wb") as file:
            pickle.dump(self, file)


if __name__ == '__main__':
    nn = NeuralNetwork(50)
    size = 500
    layer = Layer(2, 16)
    nn.add_layer(layer)
    nn.add_layer(ReLU(16))
    nn.add_layer(Layer(16, 4))
    nn.add_layer(ReLU(4))
    nn.add_layer(Layer(4, 1))
    nn.add_layer(Sigmoid(1))
    X, y = generator(size, 2, 0, 5, lambda x, y: (x - 2) * (x - 2) + (y - 2) * (y - 2) < 4)
    print("X: ", X)
    print("y: ", y)
    nn.train(X, y, CrossEntropy(), 500, 0.0001, 0.9)
