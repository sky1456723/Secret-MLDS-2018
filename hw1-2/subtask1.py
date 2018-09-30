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
import sklearn.decomposition


mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = np.reshape(x_train, (-1,784)) / 255
x_test = np.reshape(x_test, (-1,784)) / 255
batch = 32
epoch = 30
tensor_train_x = torch.Tensor(x_train)
tensor_train_y = torch.LongTensor(y_train)
dataset = torch.utils.data.TensorDataset(tensor_train_x, tensor_train_y)
dataloader = torch.utils.data.DataLoader(dataset = dataset, batch_size = batch, shuffle = True)

tensor_test_x = torch.Tensor(x_test)
tensor_test_y = torch.LongTensor(y_test)
test_dataset = torch.utils.data.TensorDataset(tensor_test_x, tensor_test_y)
test_dataloader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = 10, shuffle = True)

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
acc_list = []
loss_list = []

layer4_list = []
whole_model_list = []
model_num = 8
for i in range(model_num):
    acc_list.append([])
    loss_list.append([])
    model = myModule(784, batch)
    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    optimizer.zero_grad()
    weight_all = None
    
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
            
            optimizer.zero_grad()
            
        acc /= (np.shape(x_train)[0] / batch)
        epoch_loss /= (np.shape(x_train)[0] / batch)
        print("training_acc: ", acc)
        print("loss: ", epoch_loss)
        acc_list[i].append(acc)
        loss_list[i].append(epoch_loss)
        
        if(epoch_num%3==0):
            l1 = model.layer1.weight.clone()
            l1 = l1.cpu().data.numpy()
            l1_shape = np.shape(l1)
            l1 = np.reshape(l1, (l1_shape[0]*l1_shape[1],))
            
            l2 = model.layer2.weight.cpu().data.numpy()
            l2_shape = np.shape(l2)
            l2 = np.reshape(l2, (l2_shape[0]*l2_shape[1],))
            
            l3 = model.layer3.weight.cpu().data.numpy()
            l3_shape = np.shape(l3)
            l3 = np.reshape(l3, (l3_shape[0]*l3_shape[1],))
            
            l4 = model.layer4.weight.clone().cpu().data.numpy()
            l4_shape = np.shape(l4)
            l4 = np.reshape(l4, (l4_shape[0]*l4_shape[1],))
            layer4_list.append(l4)
            weight_all = np.concatenate((l1,l2,l3,l4))   
            whole_model_list.append(weight_all)
    
layer4_list = np.array(layer4_list)
whole_model_list = np.array(whole_model_list)
acc_list = np.array(acc_list)
epoch_num = np.array(list(range(epoch)))

pca = sklearn.decomposition.PCA(n_components=2)
new_point_layer4 = pca.fit_transform(layer4_list)
new_point_whole = pca.fit_transform(whole_model_list)

index = epoch//3

plt.subplot(2,2,1)
for i in range(model_num):
    plt.scatter(new_point_layer4[i*index:(i+1)*index,0], new_point_layer4[i*index:(i+1)*index,1])
plt.title("layer4", fontdict={'fontsize': 16})


plt.subplot(2,2,3)
for i in range(model_num):
    plt.scatter(new_point_whole[i*index:(i+1)*index,0], new_point_whole[i*index:(i+1)*index,1])
plt.title("whole model", fontdict={'fontsize': 16})


plt.subplot(2,2,2)
for i in range(model_num):
    plt.plot(epoch_num, loss_list[i])
plt.title("loss", fontdict={'fontsize': 16})

plt.subplot(2,2,4)
for i in range(model_num):
    plt.plot(epoch_num, acc_list[i])
plt.title("acc", fontdict={'fontsize': 16})
plt.show()