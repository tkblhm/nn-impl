import numpy as np


class L:
    # input: (m0, n), output: (m1, n)
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size

        self.input = None
        self.output = None

    # X: (n, m0)
    def forward(self, X, update=True):
        return self.output

    def backward(self, y, eta):
        return


class Layer(L):
    # weights: (m_0, m_1), biases: (1, m_1)
    def __init__(self, input_size, output_size):
        super(Layer, self).__init__(input_size, output_size)
        self.weights = np.random.randn(input_size, output_size) * 0.01
        # self.weights = np.random.randn(input_size, output_size) * 0.1
        # self.weights = np.random.randn(input_size, output_size)
        self.biases = np.zeros((1, output_size))
        # print("weights: ", self.weights)
        # print("bias: ", self.biases)

    # X: (n, m0)
    def forward(self, X, update=True):
        assert (X.shape[1] == self.input_size)
        if update:
            self.input = X
            # X: (n, m0), w: (m0, m1)
            self.output = X @ self.weights + self.biases
            # print("layer.output:", self.output)
            return self.output
        else:
            return X @ self.weights + self.biases

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


class Conv2D(L):
    def __init__(self, input_shape, num_filters, kernel_size, stride=1, padding=0):
        """
                :param input_shape: (height, width, channels)
                :param num_filters: Number of convolutional filters
                :param kernel_size: Size of the convolutional kernel (filter height and width)
                :param stride: Stride of the convolution
                :param padding: Amount of zero-padding around the input
        """
        input_height, input_width, input_channels = input_shape
        self.input_shape = input_shape
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Initialize weights and biases
        self.weights = np.random.randn(num_filters, input_channels, kernel_size, kernel_size) * 0.1
        self.biases = np.zeros((num_filters, 1))

        # Compute output dimensions
        self.output_height = (input_height - kernel_size + 2 * padding) // stride + 1
        self.output_width = (input_width - kernel_size + 2 * padding) // stride + 1
        self.output_size = (self.output_height, self.output_width, num_filters)

        self.input = None
        self.output = None


if __name__ == '__main__':
    pass
