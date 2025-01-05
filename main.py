from nn import *


# nn = NeuralNetwork()
# size = 100
# layer = Layer(2, 1)
# nn.add_layer(layer)
# nn.add_layer(Sigmoid(1))
# X, y = generator(size, 2, 0, 5, lambda x, y: x*x+y*y<16)
# print("X: ", X)
# print("y: ", y)
# nn.train(X, y, CrossEntropy(), 50, 0.01)

# nn = NeuralNetwork()
# nn.add_layer(Layer(1, 1))
# nn.add_layer(Sigmoid(1))
#
# nn.save("resources/nn.pkl")

with open("resources/nn.pkl", "rb") as file:
    nn = pickle.load(file)

print(nn.layers)