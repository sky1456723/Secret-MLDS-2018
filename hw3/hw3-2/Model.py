import torch
from torch import nn
import torch.nn.utils.spectral_norm as SN
import torch.nn.functional as F
import numpy as np

'''
Model of 3-2: ACGAN

Model:
    Generator:
        noise_input = (100,);
        text_input = (119,);
        # num of (hair, eyes) pairs
        text_emb = Dense(256,‘relu’)(text_input);
        concatenate([noise_input, text_emb]);
        Dense(4*4*512); Reshape((4, 4, 512));
        Batchnorm(mom=0.9); Relu;
        Conv2DTranspose(256, kernel=5);
        Batchnorm(mom=0.9); Relu;
        Conv2DTranspose(128, kernel=5);
        Batchnorm(mom=0.9); Relu;
        Conv2DTranspose(64, kernel=5);
        Batchnorm(mom=0.9); Relu;
        Conv2DTranspose(3, kernel=5);
        Tanh;

    Discriminator:
        image_input = (64,64,3);
        text_input = (119,);
        text_emb = Dense(256,’relu’)(text_input);
        text_emb = Reshape((1,1,256))(text_emb);
        tiled_emb = tile(text_emb, [1,4,4,1]);
        Conv2D(64 ,kernel=5)(image_input); LeakyRelu;
        Conv2D(128, kernel=5);
        Batchnorm(mom=0.9); LeakyRelu;
        Conv2D(256, kernel=5);
        Batchnorm(mom=0.9); LeakyReLu;
        Conv2D(512, kernel=5);
        Batchnorm(mom=0.9);
        image_feat = LeakyRelu;
        concatenate([image_feat, tiled_emb]);
        Conv2D(512, kernel=1, strides=(1,1));
        Flatten;
        Dense(1, ‘sigmoid’);

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

ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1):
    Input: (batch, channel_in, height_in, width_in)
    Output: (batch, channel_out, height_out, width_out)
    height_out = (height_in - 1) * stride - 2*padding + kernel_size + output_padding
    Default values:
        stride = 1
        padding = 0
        output_padding = 0

BatchNorm2d(num_features, eps=1e-5, momentum=0.1):
    Input: (N,C,H,W)


TA's parameters: https://drive.google.com/file/d/1W0mtkqckzWiKqvLphfcXP5e-7cAx_9Jl/view
'''

leaky = 0.1

class Generator(nn.Module):
    def __init__(self, z_dim, c_dim, leaky, momentum):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim

        self.linear_c = nn.Linear(c_dim, 256)
        self.linear_z_c = nn.Linear(z_dim + c_dim, 16*16*256)
        self.seq = nn.Sequential(
            nn.BatchNorm2d(256, momentum=momentum),
            nn.LeakyReLU(leaky),
            nn.ConvTranspose2d(256, 200, 4, stride=2, padding=1),  # (batch, 256, 8, 8)
            
            nn.BatchNorm2d(200, momentum=momentum),
            nn.LeakyReLU(leaky),
            nn.ConvTranspose2d(200, 100, 4, stride=2, padding=1),  # (batch, 256, 16, 16)
            
            nn.BatchNorm2d(100, momentum=momentum),
            nn.LeakyReLU(leaky),
            nn.Conv2d(100, 64, 3, stride=1, padding=1),   # (batch, 64, 32, 32)
            
            nn.BatchNorm2d(64, momentum=momentum),
            nn.LeakyReLU(leaky),
            nn.ConvTranspose2d(64, 3, 3, stride=1, padding=1),     # (batch, 3, 64, 64)
            nn.Sigmoid()
        ) # Deconvolute an image from (batch, 512, 4, 4) to (batch, 3, 64, 64)

    def forward(self, z, c):
        #c = self.linear_c(c)                     # (batch, 100)
        z_c = torch.cat((z, c), -1)              # (batch, dim_z + 100)
        z_c = self.linear_z_c(z_c)               # (batch, 4*4*512)
        z_c = z_c.view(z_c.shape[0], -1, 16, 16)   # (batch, 512, 4, 4)
        return self.seq(z_c)                     # (batch, 3, 64, 64)


class Discriminator(nn.Module):
    def __init__(self, c_dim, leaky, momentum):
        super(Discriminator, self).__init__()
        '''
        self.linear_c = nn.Linear(c_dim, 100)
        self.linear_mat = nn.Sequential(
            # (batch, 512) -> matchedness score (batch, 1)
            nn.Linear(256*15*15, 1)
        )
        '''
        self.linear_mat1 = nn.Sequential(
            # (batch, 512) -> matchedness score (batch, 1)
            nn.Linear(256*15*15, 12)
        )
        self.linear_mat2 = nn.Sequential(
            # (batch, 512) -> matchedness score (batch, 1)
            nn.Linear(256*15*15, 10)
        )
        
        self.linear_gen = nn.Sequential(
            # (batch, 256) -> genuineness score (batch, 1)
            nn.Linear(256*15*15, 1)
        )
        '''
        self.linear_recon = nn.Sequential(
            # (batch, 256) -> genuineness score (batch, 1)
            nn.Linear(256, 100)  # 100 means z_dim
        )
        '''
        self.seq = nn.Sequential(
            # (batch, 3, 64, 64) -> (batch, 512, 4, 4)
            nn.Conv2d(3, 32, 5, stride=1, padding=2),      # (batch, 32, 64, 64)
            nn.BatchNorm2d(32, momentum=momentum),
            nn.LeakyReLU(leaky),
            nn.AvgPool2d(2, stride=2),                         # (batch, 64, 32, 32)
            
            nn.Conv2d(32, 64, 4, stride=1, padding=1),    # (batch, 64, 31, 31)
            nn.BatchNorm2d(64, momentum=momentum),
            nn.LeakyReLU(leaky),
            
            nn.Conv2d(64, 128, 4, stride=1, padding=1),   # (batch, 128, 30, 30)
            nn.BatchNorm2d(128, momentum=momentum),
            nn.LeakyReLU(leaky),
            
            nn.Conv2d(128, 256, 3, stride=1, padding=1),   # (batch, 256, 30, 30)
            nn.BatchNorm2d(256, momentum=momentum),
            nn.LeakyReLU(leaky),
            nn.AvgPool2d(2, stride=2),                         # (batch, 256, 15, 15)
        )
        '''
        self.seq_mat = nn.Sequential(
            # (batch, 768, 4, 4) -> (batch, 512, 1, 1)
            nn.Conv2d(356, 256, 4),
            
        )
        self.seq_gen = nn.Sequential(
            # (batch, 512, 4, 4) -> (batch, 256, 1, 1)
            nn.Conv2d(256, 256, 4),          
        )
        self.seq_recon = nn.Sequential(
            # (batch, 512, 4, 4) -> (batch, 256, 1, 1)
            nn.Conv2d(256, 256, 4),          
        )
        '''
    def forward(self, x, c):
        # conditional GAN discriminator structure:
        # https://www.youtube.com/watch?v=LpyL4nZSuqU
        x = self.seq(x)                             # (batch, 256, 4, 4)
        #c = self.linear_c(c)                        # (batch, 256)
        #c = c.repeat([4, 4, 1, 1]).transpose(0, 2).transpose(1, 3) # adjusted by Jeff.
        #c = c.view(-1,100,4,4)
        
                                                    # (batch, 4, 4, 256) -> (batch, 256, 4, 4)
        #raw = torch.cat((x, c), 1)                  # (batch, 768, 4, 4), raw is used for matchedness here
        raw2 = x.view(-1,256*15*15)                     # (batch, 512, 1, 1)
        #raw2 = raw2.view(-1, 256)                     # (batch, 512)
        
        matchness12 = self.linear_mat1(raw2)          # (batch, 1)
        matchness10 = self.linear_mat2(raw2) 
        
        #matchness = self.linear_mat(raw2)
        #raw = self.seq_gen(x)                       # (batch, 1024, 1, 1), raw is used for genuineness here
        #raw = raw.view(-1, 256)                     # (batch, 256)
        genuineness = self.linear_gen(raw2)          # (batch, 1)
        
        #raw3 = self.seq_recon(x)
        #raw3 = raw3.view(-1, 256)
        #recon_err = self.linear_recon(raw3)
        
        return genuineness, matchness12, matchness10 


def two_hot(batch_size, cat_dim):
    # the category code tensor contains exactly 2 1's in each batch element
    # returns a tensor
    
    
    
    out = np.zeros((batch_size, cat_dim))
    for i in range(batch_size):
        vec = np.zeros((cat_dim,))
        vec[np.random.randint(0,11)]=1
        vec[np.random.randint(12,21)]=1
        out[i] = vec # assign shuffled two-hot
    
    return torch.Tensor(out).cuda()


# Wrapper class

class ACGAN(nn.Module):
    def __init__(self, z_dim, c_dim, gen_leaky=0.1, dis_leaky=0.1, gen_momentum=0.9, dis_momentum=0.9):
        super(ACGAN, self).__init__()
        
        self.G = Generator(z_dim, c_dim, gen_leaky, gen_momentum).cuda()
        self.D = Discriminator(c_dim, dis_leaky, dis_momentum).cuda()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.train_D = False
    
    def forward(self, x=None, c=None, z=None, c_z=None, batch_size=None):
        assert x is not None or batch_size is not None, \
            "Error: you must provide batch size."
        batch_size = x.shape[0] if x is not None else batch_size
        
        # Check noise (z)
        if type(z) is torch.Tensor:
            assert z.shape[1] == self.z_dim, \
                "Error: wrong input z dim. Expected {}, got {}".format(self.z_dim, z.shape[1])
        elif z is None:
            # sample z from gaussian
            z = torch.randn((batch_size, self.z_dim), dtype=torch.float32)
            z = (z/torch.norm(z,dim=1,keepdim=True)).cuda()
        else:
            raise TypeError("Type of noise (z) should be either None or Tensor, got {}".format(type(z)))
        
        # Check noise category code (c_z)
        if type(c_z) is torch.Tensor:
            assert c_z.shape[1] == self.c_dim, \
                "Error: wrong input c_z dim. Expected {}, got {}".format(self.c_dim, c_z.shape[1])
            assert c_z.sum() == 2 * c_z.shape[0], \
                "Error: wrong number of 1's. Expected sum of category code to be {}, got {}".format(2*c_z.shape[0], c_z.sum())
        elif c_z is None:
            # randomly generate category code
            c_z = two_hot(batch_size, self.c_dim)
        else:
            raise TypeError("Type of noise category code (c_z) should be either None or Tensor, got {}".format(type(c_z)))
        
        # This model automatically generates generator input data, thus no data is necessary
        # except for batch size.
        
        
        if self.train_D:
            # Feed G with automatically generates z and c, produces x'
            # Feed D with x, c, x', and c', produces genuineness score (D(x), D(x')) and
            #     matchedness score (Q(x, c), Q(x', c'))
            # returns:
            #     1) x': fake image
            #     2) D(x): genuineness of x
            #     3) D(x'): genuineness of x'
            #     4) Q(x, c): matchedness of x and c
            #     5) Q(x', c'): matchedness of x' and c'
            #     6) c': category code of x'
            
            assert type(x) is torch.Tensor, \
                "Input data (x) is necessary when training discriminator!"
            assert x.shape[1] == 3 and x.shape[2] == 64 and x.shape[3] == 64, \
                "Error: expected input shape of (batch size, 3, 64, 64), got {}".format(x.shape)
            assert type(c) is torch.Tensor, \
                "Input category code (c) is necessary when training discriminator!"
            assert c.shape[1] == self.c_dim, \
                "Error: wrong input c dim. Expected {}, got {}".format(self.c_dim, c.shape[1])
            assert c.sum() == 2 * c.shape[0], \
                "Error: wrong number of 1's. Expected sum of category code to be {}, got {}".format(2*c.shape[0], c.sum())
            '''
            x_g = self.G(z, c_z)
            d_x_g, q_x_g, recon = self.D(x_g, c_z)
            d_x, q_x, _ = self.D(x, c)
            
            return_dict = {
                'false_data': x_g,
                'genuineness_true': d_x,
                'genuineness_true_sig': torch.sigmoid(d_x),
                'genuineness_false': d_x_g,
                'genuineness_false_sig': torch.sigmoid(d_x_g),
                'matchedness_true': q_x,
                'matchedness_true_sig': torch.sigmoid(q_x),
                'matchedness_false': q_x_g,
                'matchedness_false_sig': torch.sigmoid(q_x_g),
                'category_code_false': c_z,
                #'reconstruction_code' : recon,
                'noise' : z
            }
            '''
            x_g = self.G(z, c_z)
            d_x_g, match12_x_g, match10_x_g = self.D(x_g, c_z)
            d_x, match12_x, match10_x = self.D(x, c)
            
            return_dict = {
                'false_data': x_g,
                'genuineness_true': d_x,
                'genuineness_true_sig': torch.sigmoid(d_x),
                'genuineness_false': d_x_g,
                'genuineness_false_sig': torch.sigmoid(d_x_g),
                'matchedness12_true':  match12_x,
                'matchedness10_true':  match10_x,
                'matchedness12_false':  match12_x_g,
                'matchedness10_false':  match10_x_g,
                'category_code_false': c_z
            }
            
            return return_dict
        
        else:
            # feed G with automatically generated z and c_z, produces x'
            # feed D with x' and c_z, produces genuineness and matchedness
            # returns:
            #     1) z + c_z
            #     2) D(x'): genuineness of generator output
            #     3) Q(x', c_z): matchedness of x' and c_z
            '''
            x_g = self.G(z, c_z)
            d_x_g, q_x_g, recon = self.D(x_g, c_z)
            
            return_dict = {
                'noise': z,
                'category_code_false': c_z,
                'genuineness_false': d_x_g,
                'genuineness_false_sig': torch.sigmoid(d_x_g),
                'matchedness_false': q_x_g,
                'matchedness_false_sig': torch.sigmoid(q_x_g),
                'reconstruction_code' : recon
            }
            '''
            x_g = self.G(z, c_z)
            d_x_g, match12_x_g, match10_x_g= self.D(x_g, c_z)
            
            return_dict = {
                'noise': z,
                'category_code_false': c_z,
                'genuineness_false': d_x_g,
                'genuineness_false_sig': torch.sigmoid(d_x_g),
                'matchedness12_false':  match12_x_g,
                'matchedness10_false':  match10_x_g,
                'category_code_false': c_z
            }
            
            return return_dict
    def infer(self, c_z=None, batch_size=None, noise_z=None):
        # input a category code into generator to generate corresponding images
        # generate as much images as batch size of c_z
        if type(c_z) is torch.Tensor:
            assert c_z.shape[1] == self.c_dim, \
                "Error: wrong input c_z dim. Expected {}, got {}".format(self.c_dim, c_z.shape[1])
            assert c_z.sum() == 2 * c_z.shape[0], \
                "Error: wrong number of 1's. Expected sum of category code to be {}, got {}". \
                format(2*c_z.shape[0], c_z.sum())
        elif c_z is None:
            # randomly generate category code
            assert batch_size is not None, \
                "Error: you must provide batch size."
            c_z = two_hot(batch_size, self.c_dim)
        else:
            raise TypeError("Type of noise category code (c_z) should be either None or Tensor, got {}".format(type(c_z)))
        
        self.eval()
        with torch.no_grad():
            if noise_z is None:
                z = torch.randn((c_z.shape[0], self.z_dim), dtype=torch.float32).cuda()
            else:
                z = noise_z
            out = self.G(z, c_z)
        return out
    
    def train_discriminator(self):
        self.train_D = True
        
    def train_generator(self):
        self.train_D = False
    
    def parameters(self):
        return {'all': super(ACGAN, self).parameters(), 'discriminator': self.D.parameters(), 'generator': self.G.parameters()}


def main():
    ### PARAMETER ###
    batch_size = 10
    noise_dim = 256
    cat_dim = 128
    
    ### INPUT ###
    test_x = torch.randn((batch_size, 3, 64, 64), dtype=torch.float32).cuda()
    test_z = torch.randn((batch_size, noise_dim), dtype=torch.float32).cuda()
    test_c = two_hot(batch_size, cat_dim)
    test_c_z = two_hot(batch_size, cat_dim)
    
    ### MODEL ###
    model = ACGAN(256, 128, gen_leaky=0, dis_leaky=0)
    print("Model: ")
    print(model)
    print()
    
    model.train_discriminator()
    print(model(x=test_x, c=test_c).keys())
    
    model.train_generator()
    print(model(batch_size=batch_size).keys())
    
    model.train_discriminator()
    print("\nGenuineness score:")
    print(model(x=test_x, z=test_z, c=test_c, c_z=test_c_z)['genuineness_true'])
    
    model.train_generator()
    print("\nCategory code:")
    print(model(x=test_x, z=test_z, c=test_c, c_z=test_c_z)['category_code_false'])
    
    print("\nInfer:")
    print(model.infer(test_c_z))
    
    print("\nInfer:")
    print(model.infer(batch_size=batch_size))
        
    # invalid input
    test_c_z[-1] = 1
    print("\nInfer:")
    print(model.infer(test_c_z))
    
    print(model.parameters()['all'])
    print(model.parameters()['generator'])

if __name__ == '__main__':
    main()