#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 16:13:15 2018

@author: jimmy
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.utils.data


mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = np.reshape(x_train, (-1,784)) / 255

simulate_func_x = np.reshape( np.array(list(range(0,1000))) / 100, (-1,1))
simulate_func_y = np.sinc(simulate_func_x)


batch = 32
epoch = 10
tensor_train_x = torch.Tensor(x_train)
tensor_train_y = torch.LongTensor(y_train)

tensor_train_x2 = torch.Tensor(simulate_func_x)
tensor_train_y2 = torch.Tensor(simulate_func_y)

dataset = torch.utils.data.TensorDataset(tensor_train_x, tensor_train_y)
dataset2 = torch.utils.data.TensorDataset(tensor_train_x2, tensor_train_y2)

dataloader = torch.utils.data.DataLoader(dataset = dataset, batch_size = batch, shuffle = True)
dataloader2 = torch.utils.data.DataLoader(dataset = dataset2, batch_size = batch, shuffle = True)


class myModule(torch.nn.Module):
    def __init__(self, input_dim, batch_size):
        super(myModule, self).__init__()
        self.layer1 = torch.nn.Linear(input_dim, 40)
        self.activation = torch.nn.ReLU()
        self.layer2 = torch.nn.Linear(40, 40)
        self.layer3 = torch.nn.Linear(40, 10)
        self.activation2 = torch.nn.Softmax(dim=1)
    def forward(self,x):
        out1 = self.layer1(x)
        act1 = self.activation(out1)
        out2 = self.layer2(act1)
        act2 = self.activation(out2)
        out3 = self.layer3(act2)
        output = self.activation2(out3)
        return output

#====================================
#MNIST

model = myModule(784, batch)
loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
optimizer.zero_grad()

loss_list = []
grad_list = []
acc_list = []
epoch_num_list = list(range(epoch)) 
iter_num_list = list(range(epoch*(len(x_train)//batch ) ))
for epoch_num in range(epoch):
    acc = 0
    epoch_loss = 0
    for input_data, input_label in dataloader:
        model_pred = model.forward(input_data)
        acc += torch.sum(torch.eq(input_label, model_pred.argmax(dim=1)), dim=0).item()/batch
        cross_entropy = loss(model_pred, input_label)
        epoch_loss += cross_entropy.item()
        cross_entropy.backward()
        optimizer.step()
        
        grad_all = 0
        for param in model.parameters():
            grad = 0
            if param.grad is not None:
                grad = (param.grad.cpu().data.numpy() **2).sum()
            grad_all += grad
        grad_norm = grad_all ** 0.5
        grad_list.append(grad_norm)
        
        optimizer.zero_grad()
    acc /= (np.shape(x_train)[0] / batch)
    epoch_loss /= (np.shape(x_train)[0] / batch)
    #print("weight norm: ", model.layer1.weight.norm().item())
    #norm_list.append(model.layer1.weight.norm().item())
    
    print("accuracy: ", acc)
    print("loss: ", epoch_loss)
    loss_list.append(epoch_loss)
    acc_list.append(acc)

#===========================================
#simulate sinc function
model2 = torch.nn.Sequential(
        torch.nn.Linear(1, 40),
        torch.nn.ReLU(),
        torch.nn.Linear(40, 40),
        torch.nn.ReLU(),
        torch.nn.Linear(40, 1)
        
        )
loss2 = torch.nn.MSELoss()
optimizer2 = torch.optim.Adam(model2.parameters())
optimizer2.zero_grad()
iter_num_list2 = list(range(epoch*(len(simulate_func_x)//batch+1 ) ))
loss_list2 = []
grad_list2 = []
for epoch_num in range(epoch):
    epoch_loss = 0
    for x, y in dataloader2:
        y_pred = model2.forward(x)
        MSE = loss2(y_pred, y)
        epoch_loss += MSE.item()
        MSE.backward()
        optimizer2.step()
        
        grad_all = 0
        for param in model2.parameters():
            grad = 0
            if param.grad is not None:
                grad = (param.grad.cpu().data.numpy() **2).sum()
            grad_all += grad
        grad_norm = grad_all ** 0.5
        grad_list2.append(grad_norm)
        
        optimizer2.zero_grad()
    epoch_loss /= (np.shape(simulate_func_x)[0]//batch + 1)
    print("loss: ", epoch_loss)
    loss_list2.append(epoch_loss)
#=============================================
#plot-MNIST
plt.plot(iter_num_list, grad_list)
plt.xlabel("iteration", fontsize=16)
plt.ylabel("norm", fontsize=16)
plt.show()

plt.plot(epoch_num_list, loss_list)
plt.xlabel("epoch", fontsize=16)
plt.ylabel("cross_entropy", fontsize=16)
plt.show()

plt.plot(epoch_num_list, acc_list)
plt.xlabel("epoch", fontsize=16)
plt.ylabel("acc", fontsize=16)
plt.show()

#plot-simulate a function
plt.plot(iter_num_list2, grad_list2)
plt.xlabel("iteration", fontsize=16)
plt.ylabel("norm", fontsize=16)
plt.show()

plt.plot(epoch_num_list, loss_list2)
plt.xlabel("epoch", fontsize=16)
plt.ylabel("MSE", fontsize=16)
plt.show()