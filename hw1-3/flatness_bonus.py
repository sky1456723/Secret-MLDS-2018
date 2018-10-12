#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 17:56:16 2018

@author: jeffchen
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.utils.data
import collections
import torch.nn as nn
import torch.autograd

# Training MNIST
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = np.reshape(x_train, (-1,784)) / 255
x_test = np.reshape(x_test, (-1,784)) / 255
#batch = range(1000, 10000, 1000)'
batch = [pow(10, i) for i in range(2, 7)]
epoch = 4
model_number = len(batch)

tensor_train_x = torch.Tensor(x_train)
tensor_train_y = torch.LongTensor(y_train)
train_dataset = torch.utils.data.TensorDataset(tensor_train_x, tensor_train_y)

tensor_test_x = torch.Tensor(x_test)
tensor_test_y = torch.LongTensor(y_test)
test_dataset = torch.utils.data.TensorDataset(tensor_test_x, tensor_test_y)

class myModule(nn.Module):
    def __init__(self, input_dim, batch_size):
        super(myModule, self).__init__()
        self.layer1 = torch.nn.Linear(input_dim, 40)
        self.activation = torch.nn.ReLU()
        self.layer2 = torch.nn.Linear(40, 40)
        self.layer3 = torch.nn.Linear(40, 40)
        self.layer4 = torch.nn.Linear(40, 10)
        self.activation2 = torch.nn.Softmax(dim=1)
    def forward(self,x):
        out1 = self.layer1(x)
        act1 = self.activation(out1)
        out2 = self.layer2(act1)
        act2 = self.activation(out2)
        out3 = self.layer3(act2)
        act3 = self.activation(out3)
        out4 = self.layer4(act3)
        output = self.activation2(out4)
        return output



train_loss_list, test_loss_list = [], []

train_acc_list, test_acc_list = [], []

sharpness_list = []

loss = torch.nn.CrossEntropyLoss()

for model_num in range(model_number):
    print("Model", model_num+1, "\b, batch size = ", batch[model_num])
    
    train_dataloader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = batch[model_num], shuffle = True)
    test_dataloader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = batch[model_num], shuffle = True)

    model = myModule(784, batch[model_num])
    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    optimizer.zero_grad()
    for epoch_num in range(epoch):
        print("\tEpoch", epoch_num+1, end = '\t')
        epoch_loss = 0
        epoch_acc = 0
        for x, y in train_dataloader:
            y_pred = model.forward(x)
            ans = y_pred.argmax(dim=1)
           
            cross_entropy = loss(y_pred, y)
            
            epoch_loss += cross_entropy.item()
            
            epoch_acc += torch.sum(torch.eq(y, ans), dim =0).item() / batch[model_num]
            cross_entropy.backward()
            optimizer.step()
            optimizer.zero_grad()
            
        print("\b\tloss: ",epoch_loss/ (len(x_train)/batch[model_num]), end='\t' )
        print("acc: ",epoch_acc/ (len(x_train)/batch[model_num]))

    weight = model.state_dict()

    train_loss = 0
    train_acc = 0
    for x, y in train_dataloader:
        y_pred = model(x)
        train_loss += loss(y_pred, y).item()
        ans = y_pred.argmax(dim=1)
        train_acc += torch.sum(torch.eq(y, ans), dim=0).item() / batch[model_num]
    train_acc /= (len(x_train) / batch[model_num])
    train_loss /= (len(x_train) / batch[model_num])
    train_loss_list.append(train_loss)
    train_acc_list.append(train_acc)
    
    test_loss = 0
    test_acc = 0
    for x, y in test_dataloader:
        y_pred = model(x)
        test_loss += loss(y_pred, y).item()
        ans = y_pred.argmax(dim=1)
        test_acc += torch.sum(torch.eq(y, ans), dim=0).item() / batch[model_num]
    test_acc /= (len(x_test)/batch[model_num])
    test_loss /= (len(x_test)/batch[model_num])
    test_loss_list.append(test_loss)
    test_acc_list.append(test_acc)
    
    


    
    epsilon = 1e-4
    train_acc_s, train_loss_s = train_acc, train_loss
    train_loss0 = train_loss
    
    samples = []
    for w in weight:
        dims = weight[w].size()
        w_ = torch.zeros_like(weight[w]).reshape(-1,1)
        for t in range(len(w_))[:100]:
            wi = weight.copy()
            _ = w_.clone()
            _[t] += epsilon
            
            wi[w] = weight[w] + _.reshape(dims)
            samples.append(wi)
            
            
    for j in samples:
        new_model = myModule(784, None)
        new_model.load_state_dict(j)

        samp_loss, samp_acc = 0, 0
    
        for x, y in train_dataloader:
            y_pred = new_model(x)
            train_loss_s += loss(y_pred, y).item()
            ans = y_pred.argmax(dim=1)
            train_acc += torch.sum(torch.eq(y, ans), dim=0).item() / batch[model_num]
        train_acc_s /= (len(x_train) / batch[model_num])
        train_loss_s /= (len(x_train) / batch[model_num])
        
        if train_loss_s > train_loss0:
            train_loss0 = train_loss_s
    
    sharpness = (train_loss0 - train_loss) / (1 + train_loss)
    sharpness_list.append(sharpness)
    

    


print("Start plotting")
figure, axes = plt.subplots()
axes2 = axes.twinx()
axes.plot(batch, train_loss_list, 'b-', label = "train")
axes.plot(batch, test_loss_list, 'b--', label = "test")
axes2.plot(batch, sharpness_list, 'r-', label = "train")
axes2.plot(batch, sharpness_list, 'r--', label = "test")
axes.set_xlabel("batch_size", fontsize = 16)
axes.set_xscale('log')
axes.set_ylabel("cross_entropy", fontsize = 16, color = 'b')
axes2.set_ylabel("sharpness", fontsize = 16, color = 'r')
plt.legend(loc = 'upper right', fontsize=16)
plt.savefig("flatness1.png")
plt.show()

figure, axes = plt.subplots()
axes2 = axes.twinx()
axes.plot(batch, train_acc_list, 'b-', label = "train")
axes.plot(batch, test_acc_list, 'b--', label = "test")
axes2.plot(batch, sharpness_list, 'r-', label = "train")
axes2.plot(batch, sharpness_list, 'r--', label = "test")
axes.set_xlabel("batch_size", fontsize = 16)
axes.set_xscale('log')
axes.set_ylabel("acc", fontsize = 16, color = 'b')
axes2.set_ylabel("sharpness", fontsize = 16, color = 'r')
plt.legend(loc = 'upper right', fontsize=16)
plt.savefig("flatness2.png")
plt.show()
#'''
