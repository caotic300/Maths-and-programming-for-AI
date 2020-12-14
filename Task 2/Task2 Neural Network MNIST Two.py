# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 13:57:10 2020

@author: Husam
"""
import numpy as np
from sklearn.datasets import fetch_openml
from abc import ABC, abstractmethod
mnist = fetch_openml('mnist_784')
X, y = mnist["data"], mnist["target"]

print(X.shape)
print(y.shape)

from keras.utils.np_utils import to_categorical


#def to_categorical(labels):
    #n_classes = labels.max() + 1
    
    #y = np.zeros((labels.shape[0], n_classes))
    
    #for i in range(labels.shape[0]):
        #y[i, label[i]] = 1
    
    #return y


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


class Optimizer(ABC):
    
    def __init__(self, learning_rate = 0.01, momentum = 0.0):
        self.learning_rate = learning_rate
        self.momentum = momentum
        
    @abstractmethod
    def step(self):
        pass
    
    @abstractmethod
    def update(self, params, gradient):
        pass
    
class SGD(Optimizer):
    def __init__(self, learning_rate =0.01, momentum = 0.0):
        super().__init__(learning_rate=learning_rate, momentum= momentum)
    
    def update(self, params, gradient, has_delta=False):
        if not has_delta:
                
                self.delta = np.zeros_like(params)

        self.delta = self.momentum
       
    def step(self):
        pass
        #if self.first: #if  

    def update(self, params, gradient):
        pass



class Layer(ABC):
    
    """ A layer of neurons in a neural network"""
    
    def __init__(self, neurons: int):
        self.neurons = neurons
        self.is_initialized = True
        self.gradients_param = []
        self.operations = []
        self.seed = 1
    @abstractmethod 
    def create_layer(self, inputs, seed):
        """ Creates layer """
        pass
    
    @abstractmethod
    def forward(self, inputs):
        """ Calculate layer output using forward propagation for given input. """
        pass

    @abstractmethod 
    def backward(self, ouputs):
        pass
    

class DenseLayer(Layer):
    
    def __init__(self, neurons: int, activation):
        super().__init__(neurons)
        self.activation = activation
        
    def create_layer(self, inputs, seed):
        
        if isinstance(seed, int):
            self.seed = np.random.seed(seed)
        self.params = []
        
        #Add weights
        self.params.append(np.random.randn(inputs.shape[1], self.neurons))
        
        #bias
        self.params.append(np.random.randn(1, self.neurons))
        
        self.operations = [WeightMultiply(self.params[0]),
                           BiasAdd(self.params[1]),
                           self.activation]
class Activation(Layer):
    

    """Class of an operation in a neural network such as forward and backward"""
    def __init__(self, type_func):
        if type_func == 'sigmoid':
            self.act_func = self.sigmoid
            self.act_func_d = self.sigmoid_d
        elif type_func == 'relu':
            self.act_func = self.relu
            self.act_func_d = self.relu_d
        elif type_func == 'tanh':
            self.act_func = self.tanh
            self.act_func_d = self.tanh_d
        elif type_func == 'softmax':
            self.act_func = self.softmax
            self.act_func_d = self.softmax_d
        else:
            raise ValueError('Invalid activation function.')

    def forward(self, inputs):
       self.last_input = inputs
       return self.act_func(inputs)

    
    def backward(self, output):
        return output * self.act_func_d(self.last_input)

    def sigmoid(self, z):
            
        #if isDeriv: # When backpropagating
            #return (np.exp(-z)) / ((np.exp(-z) + 1) **2)
        
        # Forward feed
        return 1 / (1 + np.exp(-z))
  
    def sigmoid_d(self, z):
       s = self.sigmoid(z)
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

    def softmax_d(self, x):
        s = self.softmax(x)
        ds = np.stack([np.diag(s[i, :]) for i in range(s.shape[0])])
  #Tanh function to be applied to the input/output layer  
    def tanh(self, z):
        """Returns the tanh of x """
        return np.tanh(z)
    
  #Gradient of tanh function 
    def tanh_d(self, z):
        """Returns the gradiend of tanh(x) function """
        e = np.exp(2*z)
        return (e - 1)/(e + 1)  
    
   
class NeuralNetwork(object):
    """The class defining a neural network"""
    
    def __init__(self, layers, loss, optimizer, seed_gen):
        self.layers = layers
        self.loss = loss
        self.seed_gen = seed_gen
        self.optimizer = optimizer

        if seed_gen:
            for layer in self.layers:
                setattr(layer, "seed", self.seed)
    
    def forward(self, X_batch):
         """Sends the data forward through the layers."""
         
         x_output = X_batch
        
         for layer in self.layers: #for all the layers use forward propagation
            x_output = layer.forward(x_output)
        
         return x_output
    
    
    def backward(self, gradient):
        """Sends the data backward through the layers."""
        grad = gradient
        
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
            
    def train(self, x_batch, y_batch):
        """Returns the loss by computing the predictive value using forward and backward propagation"""
        pred = self.forward(x_batch)
        
        loss = self.loss.forward(pred, y_batch)
        self.backward(self.loss.backward())
        
        return loss
    
    def params(self):
        """Gets the parameters for the network."""
        
        for layer in self.layers:
            yield from layer.params
    

    def params_grads(self):
        """Gets the gradiend of the loss of the parameters of the network"""
        
        for layer in self.layers:
            yield from layer.param_grads
            
"""class NeuralNetwork():
    
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

    def softmax_d(self, x):
        s = softmax(x, False)
        ds = np.stack([np.diag(s[i, :]) for i in range(s.shape[0])])
  #Tanh function to be applied to the input/output layer  
    def tanh(self, z):
        """ 'Returns the tanh of x' """
        return np.tanh(z)
    
  #Gradient of tanh function 
    def tanh_d(self, z):
        #"""'Returns the gradiend of tanh(x) function' """
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
        
        return changes"""

































































