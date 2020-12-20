# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 15:39:40 2020

@author: Husam
"""

import torch
import torch.nn as nn
import torchvision
import numpy as np
from sklearn import datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import math

# Configuring Main Pamaeters
input_size = 784
hidden_size = 128
hidden_size_two = 64
num_classes = 10
num_epochs = 10
batch_size = 100
learning_rate = 0.01

train_dataset = torchvision.datasets.MNIST(root='./data', train=True,transform=transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False,transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

examples = iter(train_loader)
samples, labels = examples.next()
#print(samples.shape)

class Loss(nn.Module):
    def __init__(self, criterion):
        super(Loss, self).__init__()
        self.criterion = self.loss_function(criterion)
    
    def current_loss_function(self):
        return self.criterion

    def loss_function(self, criterion):
        if criterion == 'cross_entropy':
            return nn.CrossEntropyLoss()
        elif criterion == 'mse':
            return nn.MSELoss()
        elif criterion == 'mae':
            return nn.L1Loss()

class Activation(nn.Module):
    def __init__(self, activation_type):
        super(Activation, self).__init__()
        self.activation_type = self.activation_func(activation_type)
     
    def activation_func(self, activation_type):
        if activation_type == 'relu':
            return nn.ReLU()
        elif activation_type == 'sigmoid':
            return nn.Sigmoid()
        elif activation_type == 'tanh':
            return nn.Tanh()
        elif activation_type == 'leaky_relu':
            return nn.LeakyReLU()
        elif activation_type == 'softmax':
            return nn.Softmax()
        
    def current_activation_function(self):
        return self.activation_type

class Optimizer(object):
    def __init__(self, model, optimizer_type, learning_rate):
        super(Optimizer, self).__init__()
        self.learning_rate = learning_rate
        self.model_params = model.parameters()
        self.optimizer_type = self.optimizer(optimizer_type)
    
    def current_optimizer(self):
        return self.optimizer_type
    
    def optimizer(self, optimizer_type):
        if optimizer_type == 'Adam':
            return torch.optim.Adam(self.model_params, lr=self.learning_rate)   
        elif optimizer_type == 'SGD':
            return torch.optim.SGD(self.model_params, lr=self.learning_rate)

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size,
                 hidden_size_two, num_classes, act_function):
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, hidden_size) 
        self.act_func = act_function
        self.l2 = nn.Linear(hidden_size, hidden_size_two)  
        self.act_func = act_function
        self.l3 = nn.Linear(hidden_size_two, num_classes)
        
    def forward(self, x):
        out = self.l1(x)
        out = self.act_func(out)
        out = self.l2(out)
        out = self.act_func(out)
        out = self.l3(out)
        return out # no activation and no softmax at the end



#device = torch.device('cpu')

# Key Functions used in Training
act_func = Activation('sigmoid').current_activation_function()
model = NeuralNet(input_size, hidden_size, hidden_size_two, num_classes, act_func)#.to(device)
criterion = Loss('cross_entropy').current_loss_function()
optimizer = Optimizer(model, 'Adam', learning_rate=learning_rate).current_optimizer()
Loss_ = [] # Track the loss for plotting

# Train the model
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):  
        images = images.reshape(-1, 28*28)#.to(device)    # origin shape: [100, 1, 28, 28]
        labels = labels#.to(device)                       # resized: [100, 784]
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        Loss_.append(loss)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
predlist=torch.zeros(0,dtype=torch.long, device='cpu')
lbllist=torch.zeros(0,dtype=torch.long, device='cpu')
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28*28)#.to(device)
        labels = labels#.to(device)
        outputs = model(images)
        # max returns (value ,index)
        _, predicted = torch.max(outputs.data, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()
        
        #conf_mat = # Append batch prediction results
        predlist=torch.cat([predlist,predicted.view(-1).cpu()])
        lbllist=torch.cat([lbllist,labels.view(-1).cpu()])

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network : {acc} %') 


from sklearn.metrics import confusion_matrix
import seaborn as sns; sns.set_style()
# Confusion matrix
conf_mat=confusion_matrix(lbllist, predlist)
print(conf_mat)
sns.heatmap(conf_mat, annot=True, fmt='d', cmap=plt.cm.Blues)


plt.plot(Loss_, 'r-')
plt.xlabel('Epochs', fontsize=15)
plt.ylabel('Loss', fontsize=15)



































