import gym
import math
import random
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

# if gpu is to be used
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(device)

class DQN(nn.Module):

    def __init__(self, h, w):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=5, stride=2)   # 31x36
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=2)  # 14x16
        self.bn2 = nn.BatchNorm2d(32)
        

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(w))   
        convh = conv2d_size_out(conv2d_size_out(h))
        
        linear_input_size = convw * convh * 64
        self.head = nn.Sequential(nn.Linear(linear_input_size, 128),
                                  nn.LeakyReLU(),
                                  nn.BatchNorm1d(128),
                                  nn.Linear(128,3)
                                  ) 

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        return self.head(x.view(x.size(0), -1))
    ## return a list, each index means the action value(0,1,2...) and 
    ## the value of the list at each index means the cumulated reward
    
def test_model():
    im = Image.open('gpu.jpg')
    im = torch.tensor([np.array(im).transpose(2,0,1)]*10,dtype = torch.float).to(device)
    _, _, screen_height, screen_width = im.shape
    
    policy_net = DQN(screen_height, screen_width).to(device)
    target_net = DQN(screen_height, screen_width).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    
    with torch.no_grad():
        target = target_net(im)
        policy = policy_net(im)
    print('shape of the output of target net:',target.shape,'target[0]:',target[0])
    print('shape of the output of policy net:',policy.shape,'policy[0]:',policy[0])
    print('target.max(1) contains the rewards and actions')
    print('target.max(1):',target.max(1))
    ### max(1)[0]: reward, max(1)[1]: action

if __name__ == '__main__':
    test_model()