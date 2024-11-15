import numpy as np


class L:
    # input: (m0, n), output: (m1, n)
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size

        self.input = None
        self.output = None

    # X: (n, m0)
    def forward(self, X):
        return self.output

    def backward(self, y, eta):
        return


class Layer(L):
    # weights: (m_0, m_1), biases: (1, m_1)
    def __init__(self, input_size, output_size):
        super(Layer, self).__init__(input_size, output_size)
        # self.weights = np.random.randn(input_size, output_size) * 0.01
        self.weights = np.random.randn(input_size, output_size) * 0.1
        self.biases = np.zeros((1, output_size))
        # print("weights: ", self.weights)
        # print("bias: ", self.biases)

    # X: (n, m0)
    def forward(self, X):
        assert (X.shape[1] == self.input_size)
        # print("layer.forward:", X)
        self.input = X
        # X: (n, m0), w: (m0, m1)
        self.output = X @ self.weights + self.biases
        # print("layer.output:", self.output)
        return self.output

    # y: dL/dz : (n, m1)
    # return dL/da: (n, m0)
    def backward(self, y, eta):
        assert (y.shape[1] == self.output_size)
        grad_weights = self.input.T @ y
        grad_biases = np.sum(y, axis=0, keepdims=True)
        grad_input = y @ self.weights.T

        self.weights -= eta * grad_weights
        self.biases -= eta * grad_biases
        return grad_input



class Activation(L):
    def __init__(self, size):
        super(Activation, self).__init__(size, size)


class Sigmoid(Activation):
    def __init__(self, size):
        super(Sigmoid, self).__init__(size)

    # X: (n, m)
    def forward(self, X):
        assert (X.shape[1] == self.input_size)
        # print("sigmoid.forward:", X)
        self.input = X
        # X: (n, m)
        self.output = 1 / (1 + np.exp(-X))
        # print("sigmoid.output:", self.output)
        return self.output

    # y: dL/da : (n, m)
    # return dL/dz : (n, m)
    def backward(self, y, eta):
        sigmoid = 1 / (1 + np.exp(-self.input))
        self.input = y * sigmoid * (1 - sigmoid)
        return self.input


if __name__ == '__main__':
    X = np.array([[0.39, 0.82, 0.36],
 [0.55, 0.24, 0.41 ],
 [0.23, 0.97, 0.82],
 [0.01, 0.49, 0.27]])
    y = np.array([[0.06],
 [0.26],
 [0.21],
 [0.33]])

    layer = Layer(3, 1)
    layer.forward(X)
