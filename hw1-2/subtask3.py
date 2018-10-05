#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 06:12:31 2018

@author: jeffchen
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.utils.data
import torch.autograd
from torch.autograd import Variable

sinc_x = np.linspace(0, 10, 41)[:-1].reshape(-1, 1)
sinc_y = np.sinc(sinc_x)

batch, epoch = 32, 10

tsr_trn_x = Variable(torch.Tensor(sinc_x))
tsr_trn_y = Variable(torch.Tensor(sinc_y))

torch.set_grad_enabled(True)

dataset = torch.utils.data.TensorDataset(tsr_trn_x, tsr_trn_y)

dataloader = torch.utils.data.DataLoader(dataset = dataset, batch_size=batch, shuffle=True)

layers = [40,40,10]

class myModule(torch.nn.Module):
    def __init__(self, input_dim, batch_size):
        super.__init__()
        self.layer1 = torch.nn.Linear(input_dim, layers[0])
        self.activation1 = torch.nn.ReLU()
        self.layer2 = torch.nn.Linear(layers[0], layers[1])
        self.layer3 = torch.nn.Linear(layers[1], layers[2])
        self.activation2 = torch.nn.Linear()
    def forward(self, x_input):
        out1 = self.layer1(x_input)
        act1 = self.activation1(out1)
        out2 = self.layer2(act1)
        act2 = self.activation1(out2)
        out3 = self.layer3(act2)
        y_output = self.activation2(out3)
        return y_output        

model = torch.nn.Sequential(
        torch.nn.Linear(1, layers[0]),
        torch.nn.ReLU(),
        torch.nn.Linear(layers[0], layers[1]),
        torch.nn.ReLU(),
        torch.nn.Linear(layers[1], layers[2]),
        torch.nn.ReLU(),
        torch.nn.Linear(layers[2], 1)
        )

loss = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())
optimizer.zero_grad()
iteration_number_list, loss_list, grad_list = list(range((len(sinc_x)//batch+1)*epoch)), [], []

for _ in range(epoch):
    epoch_loss = 0
    for (x, y) in dataloader:
        y_learned = model.forward(x)
        MSE =loss(y_learned, y)
        epoch_loss += MSE.item()
        
        MSE.backward()
        optimizer.step()
        
        grad_all = 0
        for paras in model.parameters():
            grad = 0
            if paras.grad is not None:
                grad = (paras.grad.cpu().data.numpy() **2).sum()
            grad_all += grad
        grad_norm = grad_all ** (1/2)
        grad_list.append(grad_norm)
        loss_list.append(MSE.item())
        
        optimizer.zero_grad()
    epoch_loss /= (np.shape(sinc_x)[0] // (batch+1))
    print("loss = ", epoch_loss)
    
print("Gradient")
    
loss = torch.autograd.grad(tsr_trn_y, tsr_trn_x)
optimizer = torch.optim.Adam(model.parameters())
optimizer.zero_grad()

for _ in range(epoch):
    epoch_loss = 0
    for (x, y) in dataloader:
        y_learned = model.forward(x)
        gradient_ =loss(y_learned, y)
        epoch_loss += gradient_.item()
        
        MSE.backward()
        optimizer.step()
        
        grad_all = 0
        for paras in model.parameters():
            grad = 0
            if paras.grad is not None:
                grad = (paras.grad.cpu().data.numpy() **2).sum()
            grad_all += grad
        grad_norm = grad_all ** (1/2)
        grad_list.append(grad_norm)
        loss_list.append(MSE.item())
        
        optimizer.zero_grad()
    epoch_loss /= (np.shape(sinc_x)[0] // (batch+1))
    print("loss = ", epoch_loss)


plt.subplot(211)
plt.plot(iteration_number_list, grad_list)
plt.xlabel("iteration")
plt.ylabel("norm")

plt.subplot(212)
plt.plot(iteration_number_list, loss_list)
plt.xlabel("iteration")
plt.ylabel("MSE")

plt.show()

def difff(y,x):
    samples = len(x)
    dist = x[-1]-x[0]
    return np.gradient(y.reshape(1,-1)[0]*samples/dist)

