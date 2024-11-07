import numpy as np


class Level:
    # input: (m0, n), output: (m1, n)
    def __init__(self, input_size, output_size, epoch=1):
        self.input_size = input_size
        self.output_size = output_size
        self.size = epoch

        self.input = None
        self.output = None

    # X: (n, m0)
    def forward(self, X):
        return self.output

    def backward(self, y, eta):
        return


class Layer(Level):
    # weights: (m_0, m_1), biases: (1, m_1)
    def __init__(self, input_size, output_size, epoch=1):
        super(Layer, self).__init__(input_size, output_size, epoch)
        # self.weights = np.random.randn(input_size, output_size) * 0.01
        self.weights = np.random.randn(input_size, output_size)
        self.biases = np.zeros((1, output_size))

    # X: (n, m0)
    def forward(self, X):
        assert (X.shape[1] == self.input_size and X.shape[0] == self.size)
        print("layer.forward:", X)
        self.input = X
        # X: (n, m0), w: (m0, m1)
        self.output = X @ self.weights + self.biases
        print("layer.output:", self.output)
        return self.output

    # y: dL/dz : (n, m1)
    # return dL/da: (n, m0)
    def backward(self, y, eta):
        assert (y.shape[0] == self.size and y.shape[1] == self.output_size)
        grad_weights = self.input.T @ y
        grad_biases = np.sum(y, axis=0, keepdims=True)
        grad_input = y @ self.weights.T

        self.weights -= eta * grad_weights
        self.biases -= eta * grad_biases
        return grad_input



class Activation(Level):
    def __init__(self, size, epoch=1):
        super(Activation, self).__init__(size, size, epoch)


class Sigmoid(Activation):
    def __init__(self, size, epoch):
        super(Sigmoid, self).__init__(size, epoch)

    # X: (n, m)
    def forward(self, X):
        assert (X.shape[1] == self.input_size)
        print("sigmoid.forward:", X)
        self.input = X
        # X: (n, m)
        self.output = 1 / (1 + np.exp(-X))
        print("sigmoid.output:", self.output)
        return self.output

    # y: dL/da : (n, m)
    # return dL/dz : (n, m)
    def backward(self, y, eta):
        sigmoid = 1 / (1 + np.exp(-self.input))
        self.input = y * sigmoid * (1 - sigmoid)
        return self.input


if __name__ == '__main__':
    layer = Layer(3, 2)
    layer.forward()