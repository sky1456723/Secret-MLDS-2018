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
#y_train = np.reshape(y_train, (-1,1))


batch = 32
epoch = 10
tensor_train_x = torch.Tensor(x_train)
tensor_train_y = torch.LongTensor(y_train)
dataset = torch.utils.data.TensorDataset(tensor_train_x, tensor_train_y)
dataloader = torch.utils.data.DataLoader(dataset = dataset, batch_size = batch, shuffle = False)


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

model = myModule(784, batch)
loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
optimizer.zero_grad()

loss_list = []
grad_list = []
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


plt.plot(iter_num_list, grad_list)
plt.xlabel("epoch", fontsize=16)
plt.ylabel("norm", fontsize=16)
plt.legend(loc = 'upper right', fontsize=16)
plt.show()

plt.plot(epoch_num_list, loss_list)
plt.xlabel("epoch", fontsize=16)
plt.ylabel("cross_entropy", fontsize=16)
plt.legend(loc = 'upper right', fontsize=16)