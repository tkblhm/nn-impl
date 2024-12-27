from layers import *
from activations import *
from losses import *
from utils.data_generator import *
from utils.data_reader import *

import numpy as np


class NeuralNetwork:
    def __init__(self, batch=-1):
        self.layers = []
        self.mean = None
        self.std = None
        self.batch = batch
        self.epoch = 0
        self.training_losses = []
        self.test_losses = []


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
    def train(self, X, y, loss_function, epoch=50, eta=0.01, training_set=1, visualisation=False):
        assert (epoch > 0 and eta > 0 and 1 >= training_set > 0)
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
                    if i == 5:
                        pass
                    y_hat = self.forward(X_train)
                    loss = loss_function.compute_loss(y_hat, y_train)
                    self.training_losses.append(loss)
                    if (training_set < 1):
                        loss = loss_function.compute_loss(self.forward(X_test, False), y_test)
                        self.test_losses.append(loss)

                    grad = loss_function.gradient(y_hat, y_train)
                    self.backward(grad, eta)
        else:
            if self.batch < 1:
                for i in range(epoch):
                    y_hat = self.forward(X_train)
                    grad = loss_function.gradient(y_hat, y_train)
                    self.backward(grad, eta)
        print("weights: ", self.layers[0].weights)
        print("bias: ", self.layers[0].biases)




if __name__ == '__main__':
    nn = NeuralNetwork()
    size = 100
    layer = Layer(2, 1)
    nn.add_layer(layer)
    nn.add_layer(Sigmoid(1))
    # X, y = generator(size, 2, 0, 5, lambda x, y: x*x+y*y<16)
    X, y = csv_reader(r"C:\Users\hxtx1\Downloads\Logistic_Regression_Data.csv", 2)
    print("X: ", X)
    print("y: ", y)
    nn.train(X, y, CrossEntropy(), 50, 0.01)