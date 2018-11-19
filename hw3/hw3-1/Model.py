import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as SN

'''
Model:
    Generator:
        Input: (100) # Contains n-dim categorical codes & continuous latent codes & noise variables
        Linear(128*16*16)
        ReLU
        ConvTranspose2d(128, kernel=4)
        ReLU
        ConvTranspose2d(64, kernel=4)
        ReLU
        ConvTranpose2d(3, kernel=4)
        tanh -> (3, 64, 64) fake image
        
    Discriminator:
        Input: (3, 64, 64)
        Conv2d(32, kernel=4)
        ReLU > LeakyReLU
        Conv2d(64, kernel=4)
        ZeroPadding
        ReLU > LeakyReLU
        Conv2d(128, kernel=4)
        ReLU > LeakyReLU
        Conv2d(256, kernel=4)
        ReLU > LeakyReLU
        Flatten
        Dense(1) -> raw score
        Sigmoid -> 0~1 score
        
Conv2d:
    Input: (batch, channel_in, height_in, width_in)
    Output: (batch, channel_out, height_out, width_out)
                 height_in + 2*padding - dilation * (kernel_size - 1) - 1
    height_out = -------------------------------------------------------- + 1
                                         stride
    Default values:
        stride = 1
        padding = 0
        dilation = 1

ConvTranspose2d:
    Input: (batch, channel_in, height_in, width_in)
    Output: (batch, channel_out, height_out, width_out)
    height_out = (height_in - 1) * stride - 2*padding + kernel_size + output_padding
    Default values:
        stride = 1
        padding = 0
        output_padding = 0

InfoGAN paper: https://arxiv.org/pdf/1606.03657.pdf
'''

class Generator(nn.Module):
    def __init__(self, z_dim, c_dim):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim

        self.linear = nn.Sequential(
            nn.Linear(z_dim + c_dim, 128*16*16)
        )
        self.seq = nn.Sequential(
            nn.BatchNorm2d(128*16*16),
            nn.ReLU(),
            nn.ConvTranspose2d(128*16*16, 128, 4),    # (batch, 128, 4, 4)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=3), # (batch, 64, 13, 13)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, stride=5),   # (batch, 3, 64, 64)
            nn.Tanh()
        )

    def forward(self, z, c):
        z_c = torch.cat((z, c), -1) # shape (batch, dim_z + dim_c)
        z_c = self.linear(z_c) # shape (batch, 128*16*16)
        return self.seq(z_c.view(z_c.shape[0], -1, 1, 1))

class Discriminator(nn.Module):
    def __init__(self, c_dim):
        super(Discriminator, self).__init__()
        
        self.linear1 = nn.Linear(512*52*52, 512)
        self.linear2 = nn.Linear(512, 1)
        self.reconstr = nn.Linear(512, c_dim)
        self.sig = nn.Sigmoid()
        self.seq = nn.Sequential(
            SN(nn.Conv2d(3, 32, 4)),    # (batch, 32, 61, 61)
            nn.LeakyReLU(0.1),
            SN(nn.Conv2d(32, 64, 4)),   # (batch, 64, 58, 58)
            nn.LeakyReLU(0.1),
            SN(nn.Conv2d(64, 128, 4)),  # (batch, 128, 55, 55)
            nn.LeakyReLU(0.1),
            SN(nn.Conv2d(128, 256, 4)), # (batch, 512, 52, 52)
            nn.LeakyReLU(0.1)
        )
    
    def forward(self, x):
        raw = self.seq(x).view(-1, 512*52*52) # (batch, 512*52*52)
        raw = self.linear1(raw)               # (batch, 512)
        raw_score = self.linear2(raw)
        sig_score = self.sig(raw_score)
        c_reconstr = self.reconstr(raw)       # (batch, c_dim)
        return raw_score, sig_score, c_reconstr

# Wrapper class

class InfoGAN(nn.Module):
    def __init__(self, z_dim, c_dim):
        super(InfoGAN, self).__init__()
        
        self.G = Generator(z_dim, c_dim).cuda()
        self.D = Discriminator(c_dim).cuda()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.train_D = False
    
    def forward(self, x, z=None, c=None):
        assert x.shape[1] == 3 and x.shape[2] == 64 and x.shape[3] == 64, \
            "Error: expected input shape of (batch size, 3, 64, 64), got {}".format(x.shape)
        if type(z) is torch.Tensor:
            assert z.shape[1] == self.z_dim, \
                "Error: wrong input z dim. Expected {}, got {}".format(self.z_dim, z.shape[1])
        if type(c) is torch.Tensor:
            assert c.shape[1] == self.c_dim, \
                "Error: wrong input c dim. Expected {}, got {}".format(self.c_dim, c.shape[1])
        
        return_dict = {}
        
        if z is None:
            z = torch.randn((x.shape[0], self.z_dim), dtype=torch.float32).cuda()
        if c is None:
            c = torch.randn((x.shape[0], self.c_dim), dtype=torch.float32).cuda()
        
        if self.train_D:
            # feed D with x
            # feed G with automatically generates z and c, produces x', feed D with x'
            # returns:
            #     1) x'
            #     2) score of true data (D(x))
            #     3) score of false data (D(x'))
            #     4) c and c' = Q(x')
            return_dict = {}
            raw, sig, _ = self.D(x)
            return_dict['raw_true'] = raw
            return_dict['sig_true'] = sig
            
            x_g = self.G(z=z, c=c)
            raw, sig, crec = self.D(x_g)
            return_dict['false_data'] = x_g
            return_dict['raw_false'] = raw
            return_dict['sig_false'] = sig
            return_dict['reconstructed_code'] = crec
            
            return return_dict
        
        else:
            # feed G with automatically generated z and c, produces x'
            # feed D with x', produces score
            # returns:
            #     1) z + c
            #     2) score of false data (D(x'))
            x_g = self.G(z=z, c=c)
            raw, sig, _ = self.D(x_g)
            return_dict['noise'] = torch.cat((z, c), dim=-1)
            return_dict['raw_false'] = raw
            return_dict['sig_false'] = sig
            return return_dict
    
    def train_discriminator(self):
        self.train_D = True
        
    def train_generator(self):
        self.train_D = False
    
    def parameters(self):
        return {'discriminator': self.D.parameters(), 'generator': self.G.parameters}

def main():
    model = InfoGAN(128, 5)
    print("Model: ")
    print(model)
    print("")
    
    test_input = torch.randn((50, 3, 64, 64), dtype=torch.float32).cuda()
    test_z = torch.randn((50, 128), dtype=torch.float32).cuda()
    test_c = torch.randn((50, 5), dtype=torch.float32).cuda()
    
    model.train_discriminator()
    print(model(test_input).keys())
    model.train_generator()
    print(model(test_input).keys())
    
    model.train_discriminator()
    print(model(test_input, z=test_z, c=test_c).keys())
    model.train_generator()
    print(model(test_input, z=test_z, c=test_c).keys())
    
    print(model.parameters()['generator'])

if __name__ == '__main__':
    main()