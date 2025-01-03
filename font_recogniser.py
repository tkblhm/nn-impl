from layers import Layer
from activations import ReLU, Sigmoid
from losses import CrossEntropy
from nn import NeuralNetwork
from utils.data_generator import FontGenerator

import numpy as np
from PIL import Image

from visualisations import Visualisations


class Recogniser:
    def __init__(self, nn: NeuralNetwork):
        assert nn.trained
        self.nn = nn

    def recognise(self, image_path):
        image =Image.open(image_path).convert("L")
        arr = np.array(image).flatten().reshape((1, -1))
        print(arr)
        arr = self.nn.standardise_input(arr)
        result = self.nn.forward(arr, update=False)
        print(result)


if __name__ == '__main__':
    font_generator = FontGenerator(range(30, 60, 2))
    # font_generator.generate_and_append_dataset(["utils/fonts/cour.ttf", "utils/fonts/courbd.ttf", "utils/fonts/courbi.ttf", "utils/fonts/couri.ttf"], ["utils/fonts/comic.ttf", "utils/fonts/comicbd.ttf", "utils/fonts/comici.ttf", "utils/fonts/comicz.ttf"], 10, False, "utils/pics")
    # font_generator.save_arrays("resources/arrays.npz")
    font_generator.load_arrays("resources/arrays.npz")
    # font_generator.shuffle()
    # font_generator.save_arrays("resources/arrays.npz")
    print(font_generator.X, font_generator.y)
    net = NeuralNetwork(2000)
    net.add_layer(Layer(4096, 512), ReLU)
    net.add_layer(Layer(512, 128), ReLU)
    net.add_layer(Layer(128, 1), Sigmoid)

    net.train(font_generator.X, font_generator.y, CrossEntropy(), 15, 0.0001, 0.9, True)
    vis = Visualisations(net)
    vis.plot_learning_curves()
