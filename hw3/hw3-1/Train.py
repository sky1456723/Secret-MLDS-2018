#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 16:27:09 2018

@author: jimmy
"""

import torch
import numpy as np
import argparse
import os
import Model
import Data
import time

start_time = time.time()

### DEVICE CONFIGURATION ###
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


### CRITERION ###
def criterion(pred, flag, train_D, batch_size):
    if flag == "Base":
        if train_D:
            #expected pred[] : (batch, 1)
            #output : (1)
            loss = -1 * (torch.sum( torch.log(pred['sig_true']+1e-8)) +
                         torch.sum( torch.log(1.0 - pred['sig_false']+1e-8) ) )
            return loss/batch_size
        else:
            loss = -1*torch.sum( torch.log(pred['sig_false']+1e-8) )
            return loss/batch_size
    elif flag == "Info":
        if train_D:
            #expected pred[] : (batch, 1)
            #output : (1)
            loss = -1 * (torch.sum( torch.log(pred['sig_true']+1e-8)) +
                         torch.sum( torch.log(1.0 - pred['sig_false']+1e-8) ) )
            
            #print(loss)
            reconstrcut_err = torch.sum( (pred['latent_code']-pred['reconstructed_code'])**2 )
            loss += reconstrcut_err
            #print(loss)
            return loss/batch_size
        else:
            loss = -1*torch.sum( torch.log(pred['sig_false']+1e-8) )
            return loss/batch_size
    elif flag == "LSGAN":
        a = -1
        b = 1
        c = 1
        if train_D:
            #expected pred[] : (batch, 1)
            #output : (1)
            real = (pred['raw_true']-b)**2
            gen = (pred['raw_false']-a)**2
            loss = (torch.sum(real) + torch.sum(gen))
            return loss/batch_size
        else:
            gen = (pred['raw_false']-c)**2
            loss = torch.sum(gen)
            return loss/batch_size
    elif flag == "WGAN":
        if train_D:
            #expected pred[] : (batch, 1)
            #output : (1)
            real = pred['raw_true']
            gen = pred['raw_false']
            loss = -1*(torch.sum(real) - torch.sum(gen))
            return loss/batch_size
        else:
            gen = pred['raw_false']
            loss = -1*torch.sum(gen)
            return loss/batch_size
    elif flag == "WGAN-GP":
        pass
    
    
def main(args):
    ### DATA ###
    print("Loading Data")
    epoch = args.epoch_number
    batch_size = args.batch_size
    update_D = args.k
    if args.num_dataset == 1:
        real_dataset = Data.GetDataset()
        real_dataloader = torch.utils.data.DataLoader(dataset = real_dataset,
                                                      batch_size = batch_size,
                                                      shuffle = True)
    else:
        pass
    print("Finish Loading Data")
    ### MODEL CREATION ###
    print(args.model_type)
    print("WGAN? ", args.model_type == 'WGAN')
    model_name = args.model_name+".pkl"
    G_optim_name = args.model_name+"_G.optim"
    D_optim_name = args.model_name+"_D.optim"
    model = None
    G_optimizer = None
    D_optimizer = None
    if args.new_model:
        if os.path.isfile(model_name):
            print("Model exists, please change model_name.")
            exit()
        else:
            model = Model.InfoGAN(90, 10).to(device)
            G_optimizer = torch.optim.Adam(model.G.parameters(),
                                           lr=0.0002, betas = (0.5,0.999))
            D_optimizer = torch.optim.Adam(model.D.parameters(),
                                           lr=0.0002, betas = (0.5,0.999))
    elif args.load_model:
        if not os.path.isfile(model_name):
            print("Model doesn't exist!")
            exit()
        elif (not os.path.isfile(G_optim_name) or
              not os.path.isfile(D_optim_name)):
            print("Optim file doesn't exist!")
            exit()
        else:
            model = torch.load("./model/"+model_name).to(device)
            G_optimizer = torch.optim.Adam(model.G.parameters(),
                                           lr=0.0002, betas = (0.5,0.999))
            D_optimizer = torch.optim.Adam(model.D.parameters(),
                                           lr=0.0002, betas = (0.5,0.999))
            G_optimizer.load_state_dict(torch.load("./model/"+G_optim_name))
            D_optimizer.load_state_dict(torch.load("./model/"+D_optim_name))
    ### TRAIN ###
    print("Start Training")
    print("k value: ", update_D)  # k value is the update times of discriminator
    WGAN_c = 0.01  # use for weight clipping of WGAN
    model = model.train()
    for e in range(epoch):
        print("Epoch :", e+1)
        true_data_iter = iter(real_dataloader)
        d_loss = 0
        g_loss = 0
        batch_count = 1
        while True:
            batch_d_loss = 0
            batch_g_loss = 0
            try:
                true_data, feature1, feature2 = true_data_iter.next()
                true_data = true_data.to(torch.float32).to(device)
                
            except:
                break
            model.train_discriminator()
            
            for d in range(update_D):
                D_optimizer.zero_grad()
                prediction = model(true_data)
                
                loss = criterion(prediction, args.model_type,
                                 model.train_D, batch_size)
                batch_d_loss = loss.item()
                loss.backward()
                D_optimizer.step()
                if args.model_type == 'WGAN':
                    for param in model.D.parameters():
                        param = torch.clamp(param, -WGAN_c, WGAN_c)
            model.train_generator()
            G_optimizer.zero_grad()
            prediction_G = model(true_data)
            loss_G = criterion(prediction_G, args.model_type, model.train_D, batch_size)
            batch_g_loss = loss_G.item()
            loss_G.backward()
            G_optimizer.step()
            
            print("Batch %d / %d: D loss: %4f, G loss: %4f" %
                  (batch_count, len(real_dataloader), batch_d_loss, batch_g_loss)
                  , end='\r')
            d_loss += batch_d_loss
            g_loss += batch_g_loss
            batch_count += 1
        print('\n')
        print("Epoch loss: D: %4f, G: %4f" % (d_loss/len(real_dataloader), g_loss/len(real_dataloader)))
            
        torch.save(model, model_name)
    torch.save(G_optimizer.state_dict(), G_optim_name)
    torch.save(D_optimizer.state_dict(), D_optim_name)
                
    return 0    
   
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GAN HW3-1')
    parser.add_argument('num_dataset', type=int) # required
    parser.add_argument('--model_name', type=str, default='GAN_default')
    parser.add_argument('--model_type', type=str, default='Base')
    parser.add_argument('--epoch_number', '-e', type=int, default=20)
    parser.add_argument('--batch_size', '-b', type=int, default=50)
    parser.add_argument('-k', type=int, default=1)
    mutex = parser.add_mutually_exclusive_group(required = True)
    mutex.add_argument('--load_model', '-l', action='store_true', help='load a pre-existing model')
    mutex.add_argument('--new_model', '-n', action='store_true', help='create a new model')

    args = parser.parse_args()
    main(args)

