import numpy as np

class Loss:
    def compute_loss(self, y_hat, y):
        pass

    def gradient(self, y_hat, y):
        pass


class MeanSquaredError(Loss):
    def compute_loss(self, y_hat, y):
        assert (y_hat.shape == y.shape and y_hat.shape[1] == 1)
        loss = 1 / y_hat.shape[0] * np.sum(np.square(y_hat - y))
        # print("y_hat and y:", np.hstack((y_hat, y)))
        print("accuracy:", sum([(y_hat[i][0]>0.5 and y[i][0]) or (y_hat[i][0]<0.5 and not y[i][0]) for i in range(y_hat.shape[0])]) / y_hat.shape[0])
        return loss

    def gradient(self, y_hat, y):
        assert (y_hat.shape == y.shape and y_hat.shape[1] == 1)
        return 2 * (y_hat - y)

class CrossEntropy(Loss):
    def compute_loss(self, y_hat, y):
        # print("y_hat and y:", np.hstack((y_hat, y)))
        assert (y_hat.shape == y.shape and y_hat.shape[1] == 1)
        print("CrossEntropy accuracy:", sum([(y_hat[i][0]>0.5 and y[i][0]) or (y_hat[i][0]<0.5 and not y[i][0]) for i in range(y_hat.shape[0])]) / y_hat.shape[0])

        loss = -1 / y_hat.shape[0] * np.sum(y * np.log(y_hat + 1e-8) + (1 - y) * np.log(1 - y_hat + 1e-8))
        return loss

    def gradient(self, y_hat, y):
        assert (y_hat.shape == y.shape and y_hat.shape[1] == 1)
        return -(y / (y_hat + 1e-8)) + (1 - y) / (1 - y_hat + 1e-8)
