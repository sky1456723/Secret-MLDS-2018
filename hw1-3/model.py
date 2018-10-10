"""
Define a model and run it on CIFAR-10
Shuffle its labels to test 'generalization' ability of models
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import time
start_time = time.time()


### DEVICE CONFIGURATION ###

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


### HYPERPARAMETERS ###

num_epochs = 20
num_classes = 10
batch_size = 20
learning_rate = 0.001
momentum = 0.9


### DATA PREPROCESSING ###

class RandLabel():
    def __init__(self):
        pass

    def __call__(self, dp):
        dp = np.random.randint(10)
        return dp

train_dataset = torchvision.datasets.CIFAR10(root='./',
    train=True,
    transform=transforms.ToTensor(),
    target_transform=RandLabel(),
    download=True
)

test_dataset = torchvision.datasets.CIFAR10(root='./',
    train=False,
    transform=transforms.ToTensor(),
    target_transform=RandLabel()
)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True
)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
    batch_size=batch_size,
    shuffle=False
)


### MODEL ###

class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc = nn.Sequential(
            nn.Linear(8*8*32, 1024),
            nn.Dropout(p=0.5),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        #return nn.Softmax(out, dim=1) <-- nn.CrossEntropyLoss() already contains nn.LogSoftmax()
        return out

model = ConvNet(num_classes).to(device)


### OPTIMIZATION TARGET ###

criterion = nn.CrossEntropyLoss()
opt = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)


### TRAIN ###

def train_session():
    loss_sum = 0.0

    # trigger train mode
    model.train()

    for i, (img, lbl) in enumerate(train_loader):
        # iterate through 10000 images
        img, lbl = img.to(device), lbl.to(device)

        # forward
        opt.zero_grad()
        out = model(img)
        loss = criterion(out, lbl)

        #backward
        loss.backward()
        opt.step()

        loss_sum += loss.item()

    # return average loss of this epoch
    return loss_sum / len(train_loader)


### TEST ###

def test_session():
    loss_sum = 0.0
    model.eval()
    with torch.no_grad():
        for i, (img, lbl) in enumerate(test_loader):
            img, lbl = img.to(device), lbl.to(device)
            out = model(img)
            loss_sum += criterion(out, lbl)

    return loss_sum / len(test_loader)


### MAIN ###

x = range(num_epochs)
yt = []
yv = []

for i in range(num_epochs):
    lt = train_session()
    lv = test_session()

    yt.append(lt)
    yv.append(lv)
    print('Epoch: %d, train loss = %.4f, validation loss = %.4f' % (i, lt, lv))

plt.figure()
plt.plot(x, yt, label = 'Train')
plt.plot(x, yv, label = 'Test')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('Loss.png')
plt.show()

print("Finished training %d epochs.\n%.4f minutes elapsed." % (num_epochs, (time.time() - start_time) / 60))
