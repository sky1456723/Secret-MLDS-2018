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
TA's parameters: https://docs.google.com/presentation/d/1nDvL6YFUQBqXauOOpm8uQhs4klcNPOtMKeu5-SSdcBA/edit#slide=id.p3
'''

class Generator(nn.Module):
    def __init__(self, z_dim, c_dim):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim

        self.linear = nn.Sequential(
            nn.Linear(z_dim + c_dim, 256*16*16)
        )
        self.seq = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            #nn.ReLU(),
            nn.ConvTranspose2d(256, 256, 4, stride = 2, padding = 1),    # (batch, 128, 32, 32)
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            #nn.ReLU(),
            nn.Conv2d(256, 128, 3, stride = 1, padding = 1), # (batch, 128, 32, 32) #changed kernel size
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            #nn.ReLU(),
            nn.ConvTranspose2d(128, 128, 4, stride = 2, padding = 1), # (batch, 128, 64, 64)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            #nn.ReLU(),
            nn.Conv2d(128, 64, 3, stride = 1, padding = 1), # (batch, 64, 64, 64)
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            #nn.ReLU(),
            nn.Conv2d(64, 3, 3, stride = 1, padding = 1),   # (batch, 3, 64, 64)
            nn.Sigmoid()
        )

    def forward(self, z, c):
        z_c = torch.cat((z, c), -1) # shape (batch, dim_z + dim_c)
        z_c = self.linear(z_c) # shape (batch, 128*16*16)
        z_c = z_c.view(z_c.shape[0], -1, 16, 16) # 128 channels, 16x16 image
        output = self.seq(z_c)
        #print(output.shape)
        return output

class Discriminator(nn.Module):
    def __init__(self, c_dim):
        super(Discriminator, self).__init__()
        
        self.linear1 = nn.Linear(128*15*15, 1)
        self.linear2 = nn.Linear(100, 1)
        self.reconstr = nn.Linear(64*15*15, c_dim) # change temporarily
        self.sig = nn.Sigmoid()
        self.seq = nn.Sequential(
            SN(nn.Conv2d(3, 32, kernel_size = 5, padding = 2)),    # (batch, 32, 64, 64)
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            #nn.ReLU(),
            nn.AvgPool2d(kernel_size = 2), # (batch, 32, 32, 32)
            #nn.MaxPool2d(kernel_size = 2),
            SN(nn.Conv2d(32, 64, kernel_size = 4, padding = 1)),   # (batch, 64, 31, 31)
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1)
            #nn.ReLU(),
        )
        self.seq_score = nn.Sequential(
            SN(nn.Conv2d(64, 128, kernel_size = 4, padding = 1)),  # (batch, 128, 30, 30)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            #nn.ReLU(),
            nn.AvgPool2d(kernel_size = 2), # batch, 128, 15, 15)
            #nn.MaxPool2d(kernel_size = 2),
            SN(nn.Conv2d(128, 128, kernel_size = 3, padding = 1)), # (batch, 128, 15, 15)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            #nn.ReLU()
        )
        self.seq_recon = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size = 4, padding = 1),  # (batch, 64, 30, 30)
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            #nn.ReLU(),
            nn.AvgPool2d(kernel_size = 2), # batch, 64, 15, 15)
            #nn.MaxPool2d(kernel_size = 2),
            nn.Conv2d(64, 64, kernel_size = 3, padding = 1), # (batch, 64, 15, 15)
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            #nn.ReLU()
        )
    
    def forward(self, x):
        raw = self.seq(x)
        raw2score = self.seq_score(raw)
        raw2recon = self.seq_recon(raw)
        #print(raw.shape)
        raw2score = raw2score.view(-1, 128*15*15)         # (batch, 256*16*16)
        raw2recon = raw2recon.view(-1, 64*15*15)
        raw_score = self.linear1(raw2score)         # reduce parameter
        sig_score = self.sig(raw_score)
        c_reconstr = self.reconstr(raw2recon)       # (batch, c_dim)
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
            z = torch.randn((x.shape[0], self.z_dim), dtype=torch.float32).cuda() # x.shape[0]:batch size
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
            #x_g = x_g + 1e-5*torch.rand_like(x_g)
            raw, sig, crec = self.D(x_g)
            return_dict['false_data'] = x_g
            return_dict['raw_false'] = raw
            return_dict['sig_false'] = sig
            return_dict['latent_code'] = c
            return_dict['reconstructed_code'] = crec
            
            return return_dict
        
        else:
            # feed G with automatically generated z and c, produces x'
            # feed D with x', produces score
            # returns:
            #     1) z + c
            #     2) score of false data (D(x'))
            x_g = self.G(z=z, c=c)
            #x_g = x_g + 1e-5*torch.rand_like(x_g)
            raw, sig, crec = self.D(x_g)
            return_dict['noise'] = torch.cat((z, c), dim=-1)
            return_dict['raw_false'] = raw
            return_dict['sig_false'] = sig
            return_dict['latent_code'] = c
            return_dict['reconstructed_code'] = crec
            return return_dict
    
    def train_discriminator(self):
        self.train_D = True
        
    def train_generator(self):
        self.train_D = False
    
    def parameters(self):
        return {'all': super(InfoGAN, self).parameters(), 'discriminator': self.D.parameters(), 'generator': self.G.parameters()}

def main():
    model = InfoGAN(128, 5)
    print("Model: ")
    print(model)
    print("")
    
    test_input = torch.randn((1, 3, 64, 64), dtype=torch.float32).cuda()
    test_z = torch.randn((1, 128), dtype=torch.float32).cuda()
    test_c = torch.randn((1, 5), dtype=torch.float32).cuda()
    
    model.train_discriminator()
    print(model(test_input).keys())
    model.train_generator()
    print(model(test_input).keys())
    
    model.train_discriminator()
    print(model(test_input, z=test_z, c=test_c).keys())
    model.train_generator()
    print(model(test_input, z=test_z, c=test_c).keys())
    
    print(model.parameters()['all'])
    print(model.parameters()['generator'])

if __name__ == '__main__':
    main()