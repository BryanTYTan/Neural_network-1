import numpy as np
from numpy import True_

class layer:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        pass

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0,inputs)

class activation_sigmoid:
    def forward(self, Power):
        self.output =  1 / (1+np.exp(-Power))

# used as a different activation funct on the outputs
# exponentiates and normalizes input
class activation_Softmax:
    def forward(self, inputs):
        exp_val = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        prob = exp_val / np.sum(exp_val, axis=1, keepdims=True)
        self.output = prob


# these classes are finding out how wrong the network is
class Loss:
    def calc(self, output, intendedval):
        sample_loss = self.forward(output,intendedval)
        batch_loss = np.mean(sample_loss)
        return batch_loss

class loss_categoricalCrossentropy(Loss):
    def forward(self, predictedval, trueval):
        samples = len(predictedval)
        predicted_clip = np.clip(predictedval, 1e-7, 1-1e-7)

        Confidence = predicted_clip[range(samples), trueval]
        
        neglog = -np.log(Confidence)
        return neglog

# Declarations
lossfunct = loss_categoricalCrossentropy()
lowest_loss = 999999
LC = 0.01
NF = 0.005
DATA = []
X = []
y = []
unchangedC = 0

# Gather Training data
with open('TrainData.txt') as f:
    UNlines = f.readlines()

f.close()

Lines = [s.replace("\n","") for s in UNlines]
Lines = [s.replace(" ",",") for s in Lines]

for i in range(len(Lines)):
    li = list(Lines[i].split(","))
    DATA.append(li)

for i in range(len(Lines)):
    tempt = []
    for j in range(len(DATA[i])-1):
        tempt.append(int(DATA[i][j]))
    X.append(tempt)
    
    if DATA[i][-1] == 'True':
        y.append(1)
    else:
        y.append(0)

layer1 = layer(5,6)
activation1 = activation_ReLU()

layer2 = layer(6,2)
activation2 = activation_Softmax()

layer1.forward(X)
activation1.forward(layer1.output)

layer2.forward(activation1.output)
activation2.forward(layer2.output)

best_L1_weight = layer1.weights.copy()
best_L1_biases = layer1.biases.copy()
best_L2_weight = layer2.weights.copy()
best_L2_biases = layer2.biases.copy()

# training the bot
for i in range(2000):
    layer1.weights += LC * np.random.randn(5,6)
    layer1.biases += LC * np.random.randn(1,6)
    layer2.weights += LC * np.random.randn(6,2)
    layer2.biases += LC * np.random.randn(1,2)

    layer1.forward(X)
    activation1.forward(layer1.output)

    layer2.forward(activation1.output)
    activation2.forward(layer2.output)

    loss = lossfunct.calc(activation2.output, y)

    if loss < lowest_loss:
        best_L1_weight = layer1.weights.copy()
        best_L1_biases = layer1.biases.copy()
        best_L2_weight = layer2.weights.copy()
        best_L2_biases = layer2.biases.copy()

        lowest_loss = loss
        print("Found a new loss", lowest_loss)
        unchangedC = 0
    
        
    layer1.weights -= LC * np.random.randn(5,6)
    layer1.biases -= LC * np.random.randn(1,6)
    layer2.weights -= LC * np.random.randn(6,2)
    layer2.biases -= LC * np.random.randn(1,2)

    layer1.forward(X)
    activation1.forward(layer1.output)

    layer2.forward(activation1.output)
    activation2.forward(layer2.output)

    loss = lossfunct.calc(activation2.output, y)

    if loss < lowest_loss:
        best_L1_weight = layer1.weights.copy()
        best_L1_biases = layer1.biases.copy()
        best_L2_weight = layer2.weights.copy()
        best_L2_biases = layer2.biases.copy()

        lowest_loss = loss
        print("Found a new loss", lowest_loss)
        unchangedC = 0
    
    unchangedC += 1
    if unchangedC > 50:
        layer1.weights += NF * np.random.randn(5,6)
        layer1.biases += NF * np.random.randn(1,6)
        layer2.weights += NF * np.random.randn(6,2)
        layer2.biases += NF * np.random.randn(1,2)


prediction = np.argmax(activation2.output, axis=1)
accuracy = np.mean(prediction == y)

print("Neural Network Training Done")
print("accuracy:", accuracy * 100, "loss:", loss)

with open('TestData.txt') as f:
    UNlines = f.readlines()

f.close()

DATA = []
TestInput = []
ExpectOutput = []


Lines = [s.replace("\n","") for s in UNlines]
Lines = [s.replace(" ",",") for s in Lines]

for i in range(len(Lines)):
    li = list(Lines[i].split(","))
    DATA.append(li)

for i in range(len(Lines)):
    tempt = []
    for j in range(len(DATA[i])-1):
        tempt.append(int(DATA[i][j]))
    TestInput.append(tempt)
    
    if DATA[i][-1] == 'True':
        ExpectOutput.append(1)
    else:
        ExpectOutput.append(0)

layer1 = layer(5,6)
activation1 = activation_ReLU()

layer2 = layer(6,2)
activation2 = activation_Softmax()

layer1.forward(TestInput)
activation1.forward(layer1.output)

layer2.forward(activation1.output)
activation2.forward(layer2.output)

predicted_output = []
Counter = 0

for i in range(len(activation2.output)):
    predict = max(layer2.output[i])

    if (predict == layer2.output[i][0]):
        predicted_output.append(0)
    else:
        predicted_output.append(1)

for i in range(len(predicted_output)):
    if predicted_output[i] == ExpectOutput[i]:
        Counter += 1

accuracy = Counter / len(predicted_output)
print("Accuracy of Network with test data:", accuracy * 100)