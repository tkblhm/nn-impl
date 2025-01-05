from layers import L

import numpy as np

class Activation(L):
    def __init__(self, size):
        super(Activation, self).__init__(size, size)


class Sigmoid(Activation):
    def __init__(self, size):
        super(Sigmoid, self).__init__(size)

    # X: (n, m)
    def forward(self, X, update=True):
        assert (X.shape[1] == self.input_size)
        print()
        if update:
            self.input = X
            # X: (n, m)
            self.output = 1 / (1 + np.exp(-X))
            return self.output
        else:
            return 1 / (1 + np.exp(-X))

    # y: dL/da : (n, m)
    # return dL/dz : (n, m)
    def backward(self, y, eta):
        sigmoid = 1 / (1 + np.exp(-self.input))
        return y * sigmoid * (1 - sigmoid)


class ReLU(Activation):
    def __init__(self, size):
        super(ReLU, self).__init__(size)

    # X: (n, m)
    def forward(self, X, update=True):
        assert X.shape[1] == self.input_size
        if update:
            self.input = X
            # X: (n, m)
            self.output = np.maximum(0, X)
            return self.output
        else:
            return np.maximum(0, X)

    # y: dL/da : (n, m)
    # return dL/dz : (n, m)
    def backward(self, y, eta):
        # Gradient of ReLU: 1 if input > 0, else 0
        relu_grad = (self.input > 0).astype(float)
        return y * relu_grad
