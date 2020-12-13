# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 13:57:10 2020

@author: Husam
"""
import numpy as np
from sklearn.datasets import fetch_openml
import abc as ABC
mnist = fetch_openml('mnist_784')
X, y = mnist["data"], mnist["target"]

print(X.shape)
print(y.shape)

from keras.utils.np_utils import to_categorical
y_new = to_categorical(y)
print(y_new[2])

split = 60000
split_test = X.shape[0] - split

print(split_test)

X_train = X[:split]
y_train = y_new[:split]

X_test = X[split:]
y_test = y_new[split:]

print(X_train.shape)
print(y_train.shape)

class SGD:
    def __init__(self, learning_rate =1.e-2, momentum = 0.0):
        self.learning_rate = learning_rate
        self.momentum = momentum
    
    def update(self, params, gradient):
        if not hasattr(self, 'delta_params'):
            

class Layer(object):
    def crate_layer(self, input_shape, rng):
        """ Creates layer with parameter at __init__()"""
        pass
    
    def forward(self, input):
        """ Calculate layer output for given input (forward propagation). """
        pass

    def backpropagation(self, input):
        pass
    
class NeuralNetwork():
    
    def __init__(self, sizes, epochs, learn_rate):
        self.sizes = sizes

        self.epochs = epochs
        self.learn_r = learn_rate

# Sigmoid function to be applied to input/hidden layers 
# Output 1 as x -> +∞ | 0 as x -> -∞
    def sigmoid(self, z, isDeriv):
            
        if isDeriv: # When backpropagating
            return (np.exp(-z)) / ((np.exp(-z) + 1) **2)
        
        # Forward feed
        return 1 / (1 + np.exp(-z))
  
   def sigmoid_d(self, z):
       s = self.sigmoid(z, isDeriv=False)
       return s*(1-s)
   
# Relu function can also be applied to input/hidden layers
    def relu(self, z):
        
        r = np.maximum(0, z)
        return r
    # Gradient of the Relu function 
    def relu_d(self, z):
      
        dz = np.zeros(z.shape)
        dz[z >= 0] = 1
    return dz
# Softmax function to be applied to the output layer 
    def softmax(self, z, isDeriv):
        expo = np.exp(z - z.max())
        
        if isDeriv: # When backpropagating
            return expo / np.sum(expo, axis = 0) * (1 - expo / np.sum(expo, axis = 0))
        
        # Forward feed
        return expo / np.sum(expo, axis = 0)    
    
  #Tanh function to be applied to the input/output layer  
    def tanh(self, z):
        """Returns the tanh of x """
    return np.tanh(z)
    
  #Gradient of tanh function 
    def tanh_d(self, z):
        """Returns the gradiend of tanh(x) function """
        e = np.exp(2*z)
        return (e - 1)/(e + 1)
    
    def instantiate(self):
        
        # 3 Layered Neural Network 
        inputLayer = self.sizes[0]
        hiddenLayer = self.sizes[1]
        outputLayer = self.sizes[2]
        
        network = {
                    'W1' : np.random.randn(hiddenLayer, inputLayer) * np.sqrt(1.0/hiddenLayer)
                    #,'b1' : np.zeros(hiddenLayer, 1) * np.sqrt(1.0/ inputLayer)
                    ,'W2' : np.random.randn(outputLayer, hiddenLayer) * np.sqrt(1.0/hiddenLayer)
                    #,'b2' : np.zeros(y_train.shape[1], 1) * np.sqrt(1.0/ hiddenLayer)
                 }
        
        
        return network


    def forward(self, X_train):
        
        network = self.network
        
        # Input Layer
        network['A0'] = X_train
        
        # Input -> Hidden
        network['Z1'] = np.dot(network["W1"], network['A0'])
        network['A1'] = self.sigmoid(network['Z1'], False)

        # Hidden -> Output
        network['Z2'] = np.dot(network["W2"], network['A1'])
        network['A2'] = self.sigmoid(network['Z2'], False)
        
        # Return output from froward pass
        return network['A2']

    def backward(self, y_train, output):
        
        network = self.network
        
        # Empty Dictionary to contain backpropagation changes
        changes = {}
        
        error = 2 * (output - y_train) / output.shape[0] * self.softmax(network['Z2'], True)
        changes['W2'] = np.outer(error. params['A1'])

        error = np.dot(network['W2'].T, error) * self.sigmoid(network['Z1'], True)
        changes['W1'] = np.outer(error. network['A0'])
        
        return changes

































































