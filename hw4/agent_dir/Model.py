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

class NoisyLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias = True, sigma_0 = 0.5):
        super(NoisyLinear, self).__init__(in_features, out_features, bias = True)
        #mean is weight and bias in nn.Linear
        self.sigma_w = nn.Parameter(nn.init.uniform_(torch.rand(out_features, in_features),
                                                      a=-1/in_features, b=1/out_features))
        self.sigma_b = nn.Parameter(torch.Tensor(out_features).fill_(sigma_0/in_features))
        self.register_buffer("epsilon_input", torch.zeros(1, in_features))
        self.register_buffer("epsilon_output", torch.zeros(out_features, 1))
    def forward(self, x, fixed_noise = False):
        if not fixed_noise:
            torch.randn(self.epsilon_input.size(), out=self.epsilon_input)
            torch.randn(self.epsilon_output.size(), out=self.epsilon_output)
        func = lambda x: torch.sign(x) * torch.sqrt(torch.abs(x))
        
        epsilon_in = func(self.epsilon_input)
        epsilon_out = func(self.epsilon_output)
        epsilon_w = torch.mul(epsilon_out, epsilon_in)
        epsilon_b = epsilon_out.squeeze(dim=1)
        
        return F.linear(x, self.weight + self.sigma_w*epsilon_w,
                           bias = self.bias + self.sigma_b*epsilon_b)
    def sample_noise(self):
        torch.randn(self.epsilon_input.size(), out=self.epsilon_input)
        torch.randn(self.epsilon_output.size(), out=self.epsilon_output)
    def remove_noise(self):
        torch.zeros(self.epsilon_input.size(), out=self.epsilon_input)
        torch.zeros(self.epsilon_output.size(), out=self.epsilon_output)

class DQN(nn.Module):

    def __init__(self, h, w, dueling, noisy):
        super(DQN, self).__init__()
        
        self.dueling = dueling
        self.noisy = noisy
        
        self.conv1 = nn.Conv2d(4, 32, kernel_size=7, stride=2)   # 29x35
        #self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=2)  # 13x16
        #self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=4, stride=2)  # 5x7
        #self.bn3 = nn.BatchNorm2d(64)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w, kernel_size = 7)), kernel_size = 4)   
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h, kernel_size = 7)), kernel_size = 4)
        
        linear_input_size = convw * convh * 64
        if not self.noisy:
            self.linear_1 = nn.Sequential(nn.Linear(linear_input_size, 512),
                                      nn.ReLU(),
                                      )
            self.advantage = nn.Linear(512,3)
        else:
            self.linear_1 = nn.Sequential(NoisyLinear(linear_input_size, 512),
                                      nn.ReLU(),
                                      )
            self.advantage = NoisyLinear(512,3)
                                
        if self.dueling:
            if not self.noisy:
                self.value = nn.Linear(512,1)
            else:
                self.value = NoisyLinear(in_features = 512, out_features = 1)

    def forward(self, x, fixed_noise = False):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        if not self.dueling:
            return self.advantage(self.linear_1(x.view(x.size(0), -1)))
        else:
            x = self.linear_1(x.view(x.size(0), -1))
            A = self.advantage(x)
            A = A - torch.mean(A, dim = 1, keepdim = True)
            if self.noisy:
                V = self.value(x, fixed_noise)
            return A+V
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