# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 20:16:50 2020

@author: Husam
"""

import numpy as np
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784')
X, y = mnist["data"], mnist["target"]

X = X/255
X = X.astype('float32')
#print(X.shape)
#print(X.dtype)
#print(y.shape)

from keras.utils.np_utils import to_categorical
y_new = to_categorical(y)
#print(y_new.shape)

split = 60000
split_test = X.shape[0] - split

X_train = X[:split]
y_train = y_new[:split]

X_test = X[split:]
y_test = y_new[split:]

#print(X_train.shape)
#print(y_train.shape)

#print(X_test.shape)
#print(y_test.shape)

print(y_train.shape)

class Task2():
    
    def __init__(self, layers, epochs, learn_r):
        self.layers = layers
        self.epochs = epochs
        self.learn_r = learn_r
        
        #self.h_1 = hidden1
        #self.h_2 = hidden2
        
        self.Errors = []
        self.Epochs = []
        self.Cost = []
        
        self.network = self.instantiate()

# Sigmoid function to be applied to input/hidden layers 
# Output 1 as x -> +∞ | 0 as x -> -∞
    def sigmoid(self, z, isDeriv = False):
            
        if isDeriv: # When backpropagating
            return (np.exp(-z)) / ((np.exp(-z) + 1 )**2)
        
        # Forward feed
        return 1 / (1 + np.exp(-z))
    
# Relu function can also be applied to input/hidden layers
    def relu(self, z, isDeriv = False):
        if isDeriv: 
            if z < 0:
                return 0
            return 1  
            
        r = np.maximum(0, z)
        return r
    
# Softmax function to be applied to the output layer 
    def softmax(self, z, isDeriv = False):
        expo = np.exp(z - z.max())
        
        if isDeriv: # When backpropagating
            return expo / np.sum(expo, axis=0) * (1 - expo / np.sum(expo, axis=0))
        
        # Forward feed
        return expo / np.sum(expo, axis = 0) 
    
    def loss(ytrain, output):

        L_sum = np.sum(np.multiply(ytrain, np.log(output)))
        m = ytrain.shape[1]
        L = -(1/m) * L_sum
        return L

    def instantiate(self):
        
        # Input Layer, Hidden Layer, Output Layer
        inputLayer = self.layers[0]
        hiddenLayer = self.layers[1]
        hiddenLayer2 = self.layers[2]
        outputLayer = self.layers[3]
        
        network = {
                    'W1' : np.random.randn(hiddenLayer, inputLayer) * np.sqrt(1.0/hiddenLayer),     # Weight for Input -> Hidden 1
                    'W2' : np.random.randn(hiddenLayer2, hiddenLayer) * np.sqrt(1.0/hiddenLayer2),  # Weight for hidden 1 -> Hidden 2
                    'W3' : np.random.randn(outputLayer, hiddenLayer2) * np.sqrt(1.0/outputLayer)    # Weight for Hidden 2 -> Output
        }

        return network

    def forward(self, X_train):
        network = self.network

        # Input Layer
        network['A0'] = X_train
        
        # Input -> Hidden
        network['Z1'] = np.dot(network["W1"], network['A0'])
        network['A1'] = self.sigmoid(network['Z1'])

        # Hidden Layer 1 -> Hidden Layer 2
        network['Z2'] = np.dot(network["W2"], network['A1'])
        network['A2'] = self.sigmoid(network['Z2'])

        # Hidden Layer 2 -> Output
        network['Z3'] = np.dot(network["W3"], network['A2'])
        network['A3'] = self.softmax(network['Z3'])
        
        #print(y_train.shape)
        #print("Output: ", network['A2'])
        #output = network['A2']
        #print("Output Network A2: ", output)
        #cost = self.loss(y_train, output)
        #print(cost)
        #self.Cost.append(cost)
        # Return output 
        return network['A3']


    def backward(self, y_train, output):

        network = self.network
        change = {}
        Errors = self.Errors
        
        # Error: subtract target array from the output of a forward propagation 
        
        # Update from Softmax Output 
        #print("Output: ", output)
        #print("Output - ytrain: :", output - y_train)
        #print(y_train)
        #print(output.shape)
        
        # Update Output
        error = 2 * (output - y_train) / output.shape[0] * self.softmax(network['Z3'], isDeriv = True)
        Errors.append(error)
        change['W3'] = np.outer(error, network['A2'])
        
        # Update Hidden Layer 2
        error = np.dot(network['W3'].T, error) * self.sigmoid(network['Z2'], isDeriv = True)
        Errors.append(error)
        #error = (((y_train - output) ** 2) / 2) * (self.softmax(network['Z2'], isDeriv = True)
        #print("Error: ", error)
        change['W2'] = np.outer(error, network['A1'])

        # Update from Hidden Layer 1
        error = np.dot(network['W2'].T, error) * self.sigmoid(network['Z1'], isDeriv = True)
        Errors.append(error)
        change['W1'] = np.outer(error, network['A0'])
        
        #print(Errors.shape)

        return change

    def update_network(self, changes):
        
        for theta, grad in changes.items():
            self.network[theta] -= self.learn_r * grad

    def accuracy(self, X_test, y_test):

        predict = []

        for x, y in zip(X_test, y_test):
            output = self.forward(x)
            pred = np.argmax(output)
            predict.append(pred == np.argmax(y))
            mean = np.mean(predict)
        
        return mean

    def train_network(self, X_train, y_train, X_test, y_test):
        
        for i in range(self.epochs):
            for x,y in zip(X_train, y_train):
                output = self.forward(x)
                changes = self.backward(y, output)
                self.update_network(changes)
                
                self.Epochs.append(i)
            
            accuracy = self.accuracy(X_test, y_test)
            print("Epoch: ", i + 1, " - Accuracy: ", accuracy * 100, "%")
            
    def printErrors(self):
        print(self.Errors.shape)
        print(self.Errors[0])
        print(self.Errors[1])
          
            
NN = Task2(layers = [784, 256, 128, 10], epochs = 10, learn_r = 0.001)

NN.train_network(X_train, y_train, X_test, y_test)




# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                   Plotting 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

print(NN.Errors[0][0])
print(NN.Errors[1])
print(NN.Errors[2])
print(len(NN.Errors))

print(len(NN.Epochs))
print(NN.Epochs)

print(len(NN.Cost))
print(NN.Cost)

Error = NN.Errors[::3]
Epoch = NN.Epochs
iterations = []

for i in range(len(Epoch)):
    iterations.append(i)

print(len(Error))
print(len(iterations))
del Error[-1]

import matplotlib.pyplot as plt

plt.xlabel('EPOCH')
plt.ylabel('ERROR')
for i in range(len(Error[0])):
    plt.plot(iterations, [pt[i] for pt in Error], '.')
plt.legend()
plt.show()

print(Error[1])
print(iterations[1])

x = Error[0:5]
y = iterations[0:5]

print(x)
print(y)

print(len(x[0]))
print(len(y))



plt.xlabel('EPOCH')
plt.ylabel('ERROR')
for i in range(len(Error[0])):
    plt.plot(iterations, [pt[i] for pt in Error], '.')
plt.legend()
plt.show()



out = [0.05801755, 0.06268149, 0.13551731, 0.16804708,
       0.15930986, 0.06539832,0.08839717, 0.09125258,
       0.10061969, 0.07075896]

out2 = [0.05167471, 0.08869379, 0.11409307, 0.18247032, 0.12676151, 0.08487161,
 0.0851985,  0.08396755, 0.12000909, 0.06225987]


def loss(ytrain, output):
    L_sum = np.sum(np.multiply(ytrain, np.log(output)))
    m = ytrain.shape[1]
    L = -(1/m) * L_sum
    return L


print(loss(y_train, out))

print(loss(y_train, out2))



def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return s





















