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



class Task2():
    
    def __init__(self, layers, epochs, learn_r):
        self.layers = layers
        self.epochs = epochs
        self.learn_r = learn_r
        
        self.network = self.instantiate()

# Sigmoid function to be applied to input/hidden layers 
# Output 1 as x -> +∞ | 0 as x -> -∞
    def sigmoid(self, z, isDeriv = False):
            
        if isDeriv: # When backpropagating
            return (np.exp(-z)) / ((np.exp(-z) + 1 )**2)
        
        # Forward feed
        return 1 / (1 + np.exp(-z))
    
# Relu function can also be applied to input/hidden layers
    def relu(self, z):
        
        r = np.maximum(0, z)
        return r
    
# Softmax function to be applied to the output layer 
    def softmax(self, z, isDeriv = False):
        expo = np.exp(z - z.max())
        
        if isDeriv: # When backpropagating
            return expo / np.sum(expo, axis=0) * (1 - expo / np.sum(expo, axis=0))
        
        # Forward feed
        return expo / np.sum(expo, axis = 0) 

    def instantiate(self):
        
        # Input Layer, Hidden Layer, Output Layer
        inputLayer = self.layers[0]
        hiddenLayer = self.layers[1]
        outputLayer = self.layers[2]
        
        network = {
                    'W1' : np.random.randn(hiddenLayer, inputLayer) * np.sqrt(1.0/hiddenLayer),
                    'W2' : np.random.randn(outputLayer, hiddenLayer) * np.sqrt(1.0/hiddenLayer)
        }

        return network

    def forward(self, X_train):
        network = self.network

        # Input Layer
        network['A0'] = X_train
        
        # Input -> Hidden
        network['Z1'] = np.dot(network["W1"], network['A0'])
        network['A1'] = self.sigmoid(network['Z1'])

        # Hidden -> Output
        network['Z2'] = np.dot(network["W2"], network['A1'])
        network['A2'] = self.softmax(network['Z2'])
        
        # Return output 
        return network['A2']


    def backward(self, y_train, output):

        network = self.network
        change = {}

        # Update from Softmax Output 
        error = 2 * (output - y_train) / output.shape[0] * self.softmax(network['Z2'], isDeriv = True)
        change['W2'] = np.outer(error, network['A1'])

        # Update from Hidden Layer
        error = np.dot(network['W2'].T, error) * self.sigmoid(network['Z1'], isDeriv = True)
        change['W1'] = np.outer(error, network['A0'])

        return change

    def update_network(self, changes):
        
        for k, v in changes.items():
            self.network[k] -= self.learn_r * v

    def accuracy(self, X_test, y_test):

        predict = []

        for x, y in zip(X_test, y_test):
            output = self.forward(x)
            pred = np.argmax(output)
            predict.append(pred == np.argmax(y))
        
        return np.mean(predict)

    def train_network(self, X_train, y_train, X_test, y_test):
        
        for i in range(self.epochs):
            for x,y in zip(X_train, y_train):
                output = self.forward(x)
                changes = self.backward(y, output)
                self.update_network(changes)
            
            accuracy = self.accuracy(X_test, y_test)
            print("Epoch: ", i + 1, " - Accuracy: ", accuracy * 100, "%")
          
            
NN = Task2(layers = [784, 146, 10], epochs = 10, learn_r = 0.001)

NN.train_network(X_train, y_train, X_test, y_test)


































