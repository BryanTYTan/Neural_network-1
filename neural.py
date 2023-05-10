import numpy as np

# sigmoid activation function
def activation(Power):
    return 1 / (1+np.exp(-Power))

def prediction(weights,bias,input):
    for w,b in zip(weights,bias):
        input = activation(np.matmul(w,input) + b)
    return input

# layers (inputs, hidden layer, hidden layer 2, outputs)
layers = (5,10,3,2)

# Size of weight matrixes
weight_size = [(a,b) for a,b in zip(layers[1:], layers[:-1])]

# make the matrixes and fill with 0
weights = [np.random.standard_normal(i)/i[1]**0.5 for i in weight_size]

# biases for everything but input
bias = [np.zeros((i,1)) for i in layers[1:]]