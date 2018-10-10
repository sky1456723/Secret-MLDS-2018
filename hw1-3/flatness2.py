#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 20:29:46 2018

@author: jimmy
"""

import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt

batch = 10
epoch = 50
model_num = 5
data = torchvision.datasets.CIFAR10("./",download=True, transform = 
                        torchvision.transforms.Compose([ torchvision.transforms.ToTensor()]))
test_data = torchvision.datasets.CIFAR10("./",download=True, train=False, transform = 
                        torchvision.transforms.Compose([ torchvision.transforms.ToTensor()]))
mean = data[0][0].data.clone().numpy()
for i in range(1,len(data)):
    mean += data[i][0].data.numpy()
mean /= len(data)
stddev = (data[0][0].data.clone().numpy() - mean)**2
for i in range(1,len(data)):
    stddev += (data[i][0].data.clone().numpy() - mean)**2
stddev = np.sqrt(stddev/len(data))

mean = torch.Tensor(mean)
stddev = torch.Tensor(stddev)
transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Normalize(mean, stddev)])

data = torchvision.datasets.CIFAR10("./",download=True, transform = transform)
test_data = torchvision.datasets.CIFAR10("./",download=True, train=False, transform = transform)


class myModule(torch.nn.Module):
    def __init__(self, input_dim, batch):
        super(myModule, self).__init__()
        self.batch = batch
        self.layer1 = torch.nn.Conv2d(3, 20 ,3, padding = 1)
        self.pooling = torch.nn.MaxPool2d(2,padding=1)
        self.layer2 = torch.nn.Conv2d(20, 20 ,3, padding = 1)
        self.layer3 = torch.nn.Linear(20*9*9,100)
        self.layer4 = torch.nn.Linear(100,100)
        self.layer5 = torch.nn.Linear(100,10)
        self.activation = torch.nn.ReLU()
        self.output_activation = torch.nn.Softmax()
    def forward(self,x):
        out1 = self.pooling( self.activation(self.layer1(x)) )
        out2 = self.pooling( self.activation(self.layer2(out1)) )
        out2 = out2.view(self.batch, -1)
        out3 = self.activation( self.layer3(out2) )
        out4 = self.activation( self.layer4(out3) )
        output = self.output_activation( self.layer5(out4) )
        return output

training_loss_list=[]
training_acc_list=[]
testing_loss_list=[]
testing_acc_list=[]
norm_list=[]
for model_k in range(model_num):
    print("model ",model_k)
    model = myModule((32, 32), batch)
    model = model.cuda()
    dataloader = torch.utils.data.DataLoader(dataset = data, batch_size = batch, shuffle = True )
    test_dataloader = torch.utils.data.DataLoader(dataset = test_data, batch_size = 100, shuffle = True )

    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    optimizer.zero_grad()
    for epoch_num in range(epoch):
        epoch_loss = 0
        epoch_acc = 0
        
        for x, y in dataloader:
            x = x.cuda()
            y = y.cuda()
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
    grad_to_input_norm = 0

    for x, y in dataloader:
        x = x.cuda()
        x.requires_grad_()
        y = y.cuda()
        y_pred = model(x)
        ans = y_pred.argmax(dim = 1)
        cross_entropy = loss(y_pred, y)
        cross_entropy.backward()
        grad_to_input_norm += torch.sqrt( torch.sum(x.grad**2) )
        training_loss += cross_entropy.item()
        training_acc += torch.sum(torch.eq(ans, y), dim = 0).item() / batch
    training_acc /= (len(data)/batch)
    training_loss /= (len(data)/batch)
    training_acc_list.append(training_acc)
    training_loss_list.append(training_loss)
    norm_list.append(grad_to_input_norm / (len(data)/batch) )
    
    testing_loss = 0
    testing_acc = 0
    for x, y in test_dataloader:
        model.batch = 100
        x = x.cuda()
        y = y.cuda()
         
        y_pred = model(x)
        ans = y_pred.argmax(dim = 1)
        cross_entropy = loss(y_pred, y)
        testing_loss += cross_entropy.item()
        testing_acc += torch.sum(torch.eq(ans, y), dim = 0).item() / 100
    testing_acc /= (len(test_data)/100)
    testing_loss /= (len(test_data)/100)
    testing_acc_list.append(testing_acc)
    testing_loss_list.append(testing_loss)
    batch *= 5
 
fig1, axes = plt.subplots()
axes2 = axes.twinx()
axes.plot([10, 50, 250, 1250, 6250], norm_list, 'r-')
axes.set_xlabel("batch_size", fontsize = 16)
axes.set_ylabel("sensitivity", fontsize = 16)
axes2.plot([10, 50, 250, 1250, 6250], training_loss_list, 'b-', label = "train")
axes2.plot([10, 50, 250, 1250, 6250], testing_loss_list, 'b--', label = "test")
axes2.set_ylabel("loss", fontsize = 16)
plt.legend(loc = 'upper right', fontsize=16)
plt.savefig("sensitivity_loss.png")
plt.show()

fig2, axes = plt.subplots()
axes2 = axes.twinx()
axes.plot([10, 50, 250, 1250, 6250], norm_list, 'r-')
axes.set_xlabel("batch_size", fontsize = 16)
axes.set_ylabel("sensitivity", fontsize = 16)
axes2.plot([10, 50, 250, 1250, 6250], training_acc_list, 'b-', label = "train")
axes2.plot([10, 50, 250, 1250, 6250], testing_acc_list, 'b--', label = "test")
axes2.set_ylabel("accuracy", fontsize = 16)
plt.legend(loc = 'upper right', fontsize=16)
plt.savefig("sensitivity_acc.png")
plt.show()
    
            
            

        
        




        
