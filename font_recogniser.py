from layers import Layer
from activations import ReLU, Sigmoid
from losses import CrossEntropy
from nn import NeuralNetwork
from utils.data_generator import FontGenerator

import numpy as np

from visualisations import Visualisations

if __name__ == '__main__':
    font_generator = FontGenerator(range(30, 60, 2))
    # font_generator.generate_and_append_dataset(["utils/fonts/cour.ttf", "utils/fonts/courbd.ttf", "utils/fonts/courbi.ttf", "utils/fonts/couri.ttf"], ["utils/fonts/comic.ttf", "utils/fonts/comicbd.ttf", "utils/fonts/comici.ttf", "utils/fonts/comicz.ttf"], 10, False, "utils/pics")
    # font_generator.save_arrays("resources/arrays.npz")
    font_generator.load_arrays("resources/arrays.npz")
    font_generator.shuffle()
    print(font_generator.X, font_generator.y)
    # net = NeuralNetwork(2000)
    # net.add_layer(Layer(4096, 512), ReLU)
    # net.add_layer(Layer(512, 128), ReLU)
    # net.add_layer(Layer(128, 1), Sigmoid)
    #
    # net.train(font_generator.X, font_generator.y, CrossEntropy(), 8, 0.0001, 0.9, True)
    # vis = Visualisations(net)
    # vis.plot_learning_curves()
