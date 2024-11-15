from layers import *
from loss import *
from utils.data_generator import *
from utils.data_reader import *

import numpy as np


class NeuralNetwork:
    def __init__(self, batch=-1):
        self.layers = []
        self.mean = None
        self.std = None
        self.batch = batch

    def standardise_input(self, X):
        self.mean = np.mean(X, axis=0, keepdims=True)
        self.std = np.std(X, axis=0, keepdims=True) + 1e-8
        return (X - self.mean) / self.std

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

    def train(self, X, y, loss_function, epoch, eta=0.01):
        X = self.standardise_input(X)
        if self.batch < 1:
            for i in range(epoch):
                y_hat = self.forward(X)
                print("loss:", loss_function.compute_loss(y_hat, y))
                grad = loss_function.gradient(y_hat, y)
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