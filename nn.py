from layers import *

import numpy as np


class NeuralNetwork:
    def __init__(self):
        self.layers = []

    def add_layer(self, layer, activation=None):
        assert (self.layers == [] or self.layers[-1].output_size == layer.input_size)
        self.layers.append(layer)

        if activation is not None:
            self.layers.append(activation(layer.output_size))

    def forward(self, X):
        assert (self.layers and X.shape[1] == self.layers[0].input_size)
        value = X
        for layer in self.layers:
            value = layer.forward(value)
        return value

    def backward(self, y, eta):
        value = y
        for layer in reversed(self.layers):
            value = layer.backward(value, eta)
        # return value

    def train(self, X, y, loss_function, epoch=1, eta=0.01):
        y_hat = self.forward(X)
        print("loss:", loss_function.compute_loss(y_hat, y))
        grad = loss_function.gradient(y_hat, y)
        self.backward(grad, eta)
        print(self.layers[0].weights)
        print(self.layers[0].biases)


class CrossEntropy:
    def compute_loss(self, y_hat, y):
        print("y_hat:", y_hat)
        print("y:", y)
        assert (y_hat.shape == y.shape and y_hat.shape[1] == 1)

        loss = -1 / y_hat.shape[0] * np.sum(y * np.log(y_hat + 1e-8) + (1 - y) * np.log(1 - y_hat + 1e-8))
        return loss

    def gradient(self, y_hat, y):
        assert (y_hat.shape == y.shape and y_hat.shape[1] == 1)
        return -(y / (y_hat + 1e-8)) + (1 - y) / (1 - y_hat + 1e-8)


if __name__ == '__main__':
    nn = NeuralNetwork()
    nn.add_layer(Layer(2, 1, 10))
    nn.add_layer(Sigmoid(1, 10))
    nn.train(np.array([
    [2.5, 2.4],
    [0.5, 0.7],
    [2.2, 2.9],
    [1.9, 2.2],
    [3.1, 3.0],
    [2.3, 2.7],
    [2, 1.6],
    [1, 1.1],
    [1.5, 1.6],
    [1.1, 0.9]
]), np.array([
    1,
    0,
    1,
    1,
    1,
    1,
    0,
    0,
    0,
    0
]).reshape(-1,1), CrossEntropy(), 10, eta=0.1)
