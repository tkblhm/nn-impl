from layers import Layer
from activations import ReLU, Sigmoid
from losses import CrossEntropy
from nn import NeuralNetwork
from utils.data_generator import FontGenerator

import numpy as np
from PIL import Image
import pickle
from pathlib import Path

from visualisations import Visualisations


class FontNet:
    def __init__(self):
        self.net = None
        self.dataset_path = "resources/arrays.npz"
        self.net_path = "resources/nn.pkl"

    def generate_and_save_dataset(self):
        font_generator = FontGenerator(range(30, 60, 1))
        font_generator.generate_and_append_dataset(
            ["utils/fonts/cour.ttf", "utils/fonts/courbd.ttf", "utils/fonts/courbi.ttf", "utils/fonts/couri.ttf"],
            ["utils/fonts/comic.ttf", "utils/fonts/comicbd.ttf", "utils/fonts/comici.ttf", "utils/fonts/comicz.ttf"],
            20, False, "utils/pics")
        font_generator.shuffle()
        font_generator.save_arrays(self.dataset_path)
        print(font_generator.X, font_generator.y)

    def load_dataset(self):
        font_generator = FontGenerator(range(30, 60, 2))
        font_generator.load_arrays(self.dataset_path)
        print(font_generator.X, font_generator.y)
        return (font_generator.X, font_generator.y)

    def train_and_save_net(self):
        self.net = NeuralNetwork(2000)
        self.net.add_layer(Layer(4096, 512), ReLU)
        self.net.add_layer(Layer(512, 128), ReLU)
        self.net.add_layer(Layer(128, 1), Sigmoid)

        X, y = self.load_dataset()
        self.net.train(X, y, CrossEntropy(), 40, 0.0001, 0.9, True)
        vis = Visualisations(self.net)
        vis.plot_learning_curves()
        self.net.save(self.net_path)

    def load_net(self):
        with open(self.net_path, "rb") as file:
            self.net = pickle.load(file)

    def recognise(self, image_path):
        image = Image.open(image_path).convert("L")
        image = image.crop((0, 0, 64, 64))
        arr = np.array(image).flatten().reshape((1, -1)) / 255.0
        arr = self.net.standardise_input(arr)
        result = self.net.forward(arr, update=False)
        print(image_path, result)

    def batch_recognise(self, image_dir):
        path = Path(image_dir)
        for file in path.rglob("*.png"):
            self.recognise(file)


if __name__ == '__main__':
    font_net = FontNet()
    # font_net.generate_and_save_dataset()
    # font_net.train_and_save_net()
    font_net.load_net()
    # y = font_net.load_dataset()[1]
    # print(np.array().)
    # font_net.recognise("/Users/guo/Desktop/Screenshot 2025-01-05 at 19.15.27.png")
    font_net.batch_recognise(r"C:\Users\hxtx1\Pictures\Screenshots")
