#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 23:28:13 2018

@author: jason
"""


import keras
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.utils.data
#import collections

mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = np.reshape(x_train, (-1,784)) / 255
x_test = np.reshape(x_test, (-1,784)) / 255
batch = 128
epoch = 15

model_size = np.arange(14,50,2)
#model_size = [20]

tensor_train_x = torch.Tensor(x_train)
tensor_train_y = torch.LongTensor(y_train)
data = torch.utils.data.TensorDataset(tensor_train_x, tensor_train_y)

tensor_test_x = torch.Tensor(x_test)
tensor_test_y = torch.LongTensor(y_test)
test_data = torch.utils.data.TensorDataset(tensor_test_x, tensor_test_y)

class myModule(torch.nn.Module):
    def __init__(self, input_dim, model_scale):
        super(myModule, self).__init__()
        self.layer1 = torch.nn.Linear(input_dim, model_scale)
        self.activation = torch.nn.ReLU()
        self.layer2 = torch.nn.Linear(model_scale, model_scale)
        self.layer3 = torch.nn.Linear(model_scale, model_scale)
        self.layer4 = torch.nn.Linear(model_scale, 10)
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

loss = torch.nn.CrossEntropyLoss()

training_loss_list=[]
training_acc_list=[]
testing_loss_list=[]
testing_acc_list=[]
model_weight_number=[]

for model_k in model_size:
    print("model ",model_k)
    model = myModule(784, model_k)
    #model = model.cuda()
    dataloader = torch.utils.data.DataLoader(dataset = data, batch_size = batch, shuffle = True )
    test_dataloader = torch.utils.data.DataLoader(dataset = test_data, batch_size = 100, shuffle = True )

    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    optimizer.zero_grad()
    for epoch_num in range(epoch):
        epoch_loss = 0
        epoch_acc = 0
        
        for x, y in dataloader:
            # x = x.cuda()
            # y = y.cuda()
            y_pred = model(x)
            ans = y_pred.argmax(dim = 1)
            cross_entropy = loss(y_pred, y)
            epoch_loss += cross_entropy.item()
            epoch_acc += torch.sum(torch.eq(ans, y), dim = 0).item() / batch
            cross_entropy.backward()
            optimizer.step()
            optimizer.zero_grad()
        print("acc: ", epoch_acc/(len(data)/batch))
        print("loss: ", epoch_loss/(len(data)/batch))
    print("Finish training")

    training_loss = 0
    training_acc = 0
    for x, y in dataloader:
        # x = x.cuda()
        # y = y.cuda()
        y_pred = model(x)
        ans = y_pred.argmax(dim = 1)
        cross_entropy = loss(y_pred, y)
        training_loss += cross_entropy.item()
        training_acc += torch.sum(torch.eq(ans, y), dim = 0).item() / batch
    training_acc /= (len(data)/batch)
    training_loss /= (len(data)/batch)
    training_acc_list.append(training_acc)
    training_loss_list.append(training_loss)
    
    testing_loss = 0
    testing_acc = 0
    grad_to_input_norm = 0
    for x, y in test_dataloader:
        model.batch = 100
        # x = x.cuda()
        # x.requires_grad_()
        # y = y.cuda()
         
        y_pred = model(x)
        ans = y_pred.argmax(dim = 1)
        cross_entropy = loss(y_pred, y)
        cross_entropy.backward()
        
        # grad_to_input_norm += torch.sqrt( torch.sum(x.grad**2) )
        testing_loss += cross_entropy.item()
        testing_acc += torch.sum(torch.eq(ans, y), dim = 0).item() / 100
    testing_acc /= (len(test_data)/100)
    testing_loss /= (len(test_data)/100)
    testing_acc_list.append(testing_acc)
    testing_loss_list.append(testing_loss)
    params = list(model.parameters())
    count = 0
    for p in model.parameters():
        count += p.data.nelement()
    model_weight_number.append(count)

print("Start plotting")

plt.scatter(model_weight_number, training_loss_list, label = "training_loss")
plt.scatter(model_weight_number, testing_loss_list, label = "testing_loss")

plt.xlabel("number of parameters", fontsize = 16)
plt.ylabel("loss", fontsize = 16)

plt.legend(loc = 'upper right', fontsize=16)
plt.show()

plt.scatter(model_weight_number, training_acc_list, label = "train_accuracy")
plt.scatter(model_weight_number, testing_acc_list, label = "testing_accuracy")
plt.xlabel("number of parameters", fontsize = 16)
plt.ylabel("accuracy", fontsize = 16)
plt.show()

















