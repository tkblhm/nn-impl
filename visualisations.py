import matplotlib
from matplotlib.lines import lineStyles

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from nn import *


class Visualisations:
    def __init__(self, nn: NeuralNetwork):
        self.nn = nn

    def plot_learning_curves(self):
        assert (self.nn.epoch != 0 and self.nn.training_losses)
        x = range(self.nn.epoch)
        plt.plot(x, self.nn.training_losses, label='Training Set', linestyle='-', linewidth=2)
        if self.nn.test_losses:
            plt.plot(x, self.nn.test_losses, label='Test Set', linestyle='--', linewidth=2)
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Learning Curves')
        plt.legend()
        plt.savefig('curve.png')

    def plot_datapoints(self, x_range, y_range, density):

        # Step 2: Generate a grid of (x, y) points
        x = np.linspace(*x_range, density)  # Adjust range and resolution as needed
        y = np.linspace(*y_range, density)
        xx, yy = np.meshgrid(x, y)
        x_flat = xx.reshape((-1, 1))
        y_flat = yy.reshape((-1, 1))

        output = self.nn.compute(np.hstack((x_flat, y_flat)))
        print("output:", output)
        zz = output.reshape((density, density))

        plt.figure(figsize=(8, 6))
        plt.contourf(xx, yy, zz, levels=1, cmap="coolwarm", alpha=0.8)  # Fill regions
        plt.colorbar(label="Classification (True=1, False=0)")
        plt.title("Classification Visualization")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.savefig("distribution.png")


if __name__ == '__main__':
    nn = NeuralNetwork(100)
    size = 2000
    nn.add_layer(Layer(2, 10))
    # nn.add_layer(Sigmoid(5))
    nn.add_layer(ReLU(10))
    nn.add_layer(Layer(10, 1))
    nn.add_layer(Sigmoid(1))


    X, y = generator(size, 2, 0, 5, lambda x1, x2: (x1-2)*(x1-2) + (x2-2)*(x2-2) < 4, 1)
    # print("X: ", X)
    # print("y: ", y)
    nn.train(X, y, CrossEntropy(), 300, 0.003, 0.8, True)
    vis = Visualisations(nn)
    vis.plot_learning_curves()
    vis.plot_datapoints((0, 5), (0, 5), 50)
