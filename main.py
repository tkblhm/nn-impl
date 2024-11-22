from nn import *


nn = NeuralNetwork()
size = 100
layer = Layer(2, 1)
nn.add_layer(layer)
nn.add_layer(Sigmoid(1))
X, y = generator(size, 2, 0, 5, lambda x, y: x*x+y*y<16)
print("X: ", X)
print("y: ", y)
nn.train(X, y, CrossEntropy(), 50, 0.01)