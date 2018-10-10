#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 20:53:27 2018

@author: jimmy
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.utils.data
import collections



mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = np.reshape(x_train, (-1,784)) / 255
x_test = np.reshape(x_test, (-1,784)) / 255
batch = [64, 1024]
epoch = 10
model_number = 2
tensor_train_x = torch.Tensor(x_train)
tensor_train_y = torch.LongTensor(y_train)
dataset = torch.utils.data.TensorDataset(tensor_train_x, tensor_train_y)

tensor_test_x = torch.Tensor(x_test)
tensor_test_y = torch.LongTensor(y_test)
test_dataset = torch.utils.data.TensorDataset(tensor_test_x, tensor_test_y)

class myModule(torch.nn.Module):
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

weight_list = []
loss = torch.nn.CrossEntropyLoss()
interpolation_factor = np.linspace(-1,2,num=50)

for model_num in range(model_number):
    print("Model",model_num)
    dataloader = torch.utils.data.DataLoader(dataset = dataset, batch_size = batch[model_num], shuffle = True)
    test_dataloader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = batch[model_num], shuffle = True)
    
    model = myModule(784, batch[model_num])
    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    optimizer.zero_grad()
    for epoch_num in range(epoch):
        epoch_loss = 0
        epoch_acc = 0
        for x, y in dataloader:
            y_pred = model.forward(x)
            ans = y_pred.argmax(dim=1)
           
            cross_entropy = loss(y_pred, y)
            
            epoch_loss += cross_entropy.item()
            
            epoch_acc += torch.sum(torch.eq(y, ans), dim =0).item() / batch[model_num]
            cross_entropy.backward()
            optimizer.step()
            optimizer.zero_grad()
            
        print("loss: ",epoch_loss/ (len(x_train)/batch[model_num]) )
        print("acc: ",epoch_acc/ (len(x_train)/batch[model_num]))
    
    weight_list.append(model.state_dict())

train_loss_list = []
train_acc_list = []
test_loss_list = []
test_acc_list = []
dataloader = torch.utils.data.DataLoader(dataset = dataset, batch_size = 32, shuffle = True)
test_dataloader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = 32, shuffle = True)

for alpha in interpolation_factor:
    new_weight = collections.OrderedDict()
    for keys in weight_list[0].keys():
        new_weight[keys] = weight_list[0][keys]*alpha + weight_list[1][keys]*(1-alpha)
    new_model = myModule(784, None)
    new_model.load_state_dict(new_weight)
    
    train_loss = 0
    train_acc = 0
    for x, y in dataloader:
        y_pred = new_model(x)
        train_loss += loss(y_pred, y).item()
        ans = y_pred.argmax(dim=1)
        train_acc += torch.sum(torch.eq(y, ans), dim=0).item() / 32
    train_acc /= (len(x_train) / 32)
    train_loss /= (len(x_train) / 32)
    train_loss_list.append(train_loss)
    train_acc_list.append(train_acc)
    
    test_loss = 0
    test_acc = 0
    for x, y in test_dataloader:
        y_pred = new_model(x)
        test_loss += loss(y_pred, y).item()
        ans = y_pred.argmax(dim=1)
        test_acc += torch.sum(torch.eq(y, ans), dim=0).item() / 32
    test_acc /= (len(x_test)/32)
    test_loss /= (len(x_test)/32)
    test_loss_list.append(test_loss)
    test_acc_list.append(test_acc)

print("Start plotting")
figure, axes = plt.subplots()
axes2 = axes.twinx()
axes.plot(interpolation_factor, train_loss_list, 'r-', label = "train")
axes.plot(interpolation_factor, test_loss_list, 'r--', label = "test")
axes2.plot(interpolation_factor, train_acc_list, 'b-', label = "train")
axes2.plot(interpolation_factor, test_acc_list, 'b--', label = "test")
axes.set_xlabel("interpolation factor", fontsize = 16)
axes.set_ylabel("cross_entropy", fontsize = 16, color = 'r')
axes2.set_ylabel("acc", fontsize = 16, color = 'b')
plt.legend(loc = 'upper right', fontsize=16)
plt.savefig("flatness1.png")
plt.show()

    
    
