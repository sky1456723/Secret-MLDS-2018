import torch
import numpy as np
import Model
import matplotlib.pyplot as plt
import argparse

def save_imgs(generator, output_path, z1, z2):
    #import matplotlib.pyplot as plt
    np.random.seed(179)
    r, c = 5, 5
    noise_z = np.random.normal(0, 1, (r * c, z1))
    noise_c = np.repeat( np.random.normal(0, 1, (1, z2)), [r*c], axis = 0)
    noise_z = torch.FloatTensor(noise_z)
    noise_c = torch.FloatTensor(noise_c)
    # gen_imgs should be shape (25, 64, 64, 3)
    generator = generator.cpu()
    gen_imgs = generator(noise_z, noise_c) * 255
    gen_imgs = gen_imgs.detach().cpu().numpy().astype(np.int32)
    gen_imgs = np.transpose(gen_imgs, [0,2,3,1])
    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i,j].imshow(gen_imgs[cnt, :,:,:])
            axs[i,j].axis('off')
            cnt += 1
    fig.savefig(output_path+".png")
    plt.close()

def save_imgs_info(generator, output_path, z1, z2, index, diff):
    #import matplotlib.pyplot as plt
    np.random.seed(179)
    r, c = 5, 5
    noise_z = np.repeat( np.random.normal(0, 1, (1, z1)), [r*c], axis = 0)
    noise_c = np.random.normal(0, 1, (1, z2))
    d = np.zeros(z2)
    d[index] = diff
    
    inter_c = []
    for i in range(r*c):
        inter_c.append(noise_c[0]+ d*(i/(r*c-1)) )
        
 
    noise_z = torch.FloatTensor(noise_z)
    inter_c = torch.FloatTensor(np.array(inter_c))
    #print(inter_c)
    # gen_imgs should be shape (25, 64, 64, 3)
    generator = generator.cpu()
    gen_imgs = generator(noise_z, inter_c) * 255
    gen_imgs = gen_imgs.detach().cpu().numpy().astype(np.int32)
    gen_imgs = np.transpose(gen_imgs, [0,2,3,1])
    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i,j].imshow(gen_imgs[cnt, :,:,:])
            axs[i,j].axis('off')
            cnt += 1
    fig.savefig(output_path+".png")
    plt.close()

def main(args):
    model = torch.load(args.model, map_location='cpu')
    model = model.eval()
    generator = model.G
    save_imgs(generator, args.output, model.z_dim, model.c_dim)
    #save_imgs_info(generator, args.output+"_inter", model.z_dim, model.c_dim, 6, 10)
    return 0 
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GAN HW3-1')
    parser.add_argument("model", type=str)
    parser.add_argument("output", type=str)
    args = parser.parse_args()
    main(args)
