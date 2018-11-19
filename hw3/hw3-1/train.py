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
def criterion(pred, flag, model_param, train_D):
    if flag == "Base":
        if train_D:
            #expected pred[] : (batch, 1)
            #output : (1)
            real = torch.log(pred['sig_true'])
            gen = torch.log(1-pred['sig_false'])
            loss = -1*(torch.mean(real, dim = 0) + torch.mean(gen, dim = 0))
            return loss
        else:
            gen = torch.log(pred['sig_false'])
            loss = torch.mean(gen, dim = 0)
            return loss
    elif flag == "LSGAN":
        a = -1
        b = 1
        c = 1
        if train_D:
            #expected pred[] : (batch, 1)
            #output : (1)
            real = (pred['raw_true']-b)**2
            gen = (pred['raw_false']-a)**2
            loss = (torch.mean(real, dim = 0) + torch.mean(gen, dim = 0))
            return loss
        else:
            gen = (pred['sig_false']-c)**2
            loss = torch.mean(gen, dim = 0)
            return loss
    elif flag == "WGAN":
        if train_D:
            #expected pred[] : (batch, 1)
            #output : (1)
            real = pred['raw_true']
            gen = pred['raw_false']
            loss = -1*(torch.mean(real, dim = 0) - torch.mean(gen, dim = 0))
            return loss
        else:
            gen = -1*pred['sig_false']
            loss = torch.mean(gen, dim = 0)
            return loss
    elif flag == "WGAN-GP":
        pass
    
    
def main(args):
    ### DATA ###
    epoch = args.epoch_number
    batch_size = args.batch_size
    update_D = args.k
    if args.num_dataset == 1:
        real_dataset = ""
        real_dataloader = torch.utils.data.DataLoader(dataset = real_dataset,
                                                      batch_size = batch_size,
                                                      shuffle = True)
    else:
        pass
    
    ### MODEL CREATION###
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
            model = Model.InfoGAN(50, 50)
            G_optimizer = torch.optim.Adam(model.G.parameters(),
                                         lr=0.0002,
                                         beta = 0.5)
            D_optimizer = torch.optim.Adam(model.D.parameters(),
                                         lr=0.0002,
                                         beta = 0.5)
    elif args.load_model:
        if not os.path.isfile(model_name):
            print("Model doesn't exist!")
            exit()
        elif (not os.path.isfile(G_optim_name) or
              not os.path.isfile(D_optim_name)):
            print("Optim file doesn't exist!")
            exit()
        else:
            model = torch.load("./model/"+model_name)
            G_optimizer = torch.optim.Adam(model.G.parameters(),
                                         lr=0.0002,
                                         beta = 0.5)
            D_optimizer = torch.optim.Adam(model.D.parameters(),
                                         lr=0.0002,
                                         beta = 0.5)
            G_optimizer.load_state_dict(torch.load("./model/"+G_optim_name))
            D_optimizer.load_state_dict(torch.load("./model/"+D_optim_name))
    ### TRAIN ###
    
    WGAN_c = 5
    
    for e in range(epoch):
        true_data_iter = iter(real_dataloader)
        while True:
            try:
                true_data, true_label = true_data_iter.next()
            except:
                break
            
            model.train_discriminator()
            for d in range(update_D):
                D_optimizer.zero_grad()
                prediction = model(true_data)
                loss = criterion(prediction, args.model_type,
                                 model.parameters(), model.train_D)
                loss.backward()
                D_optimizer.step()
                if args.model_type == 'WGAN':
                    for param in model.D.parameters():
                        param = torch.clamp(param, -WGAN_c, WGAN_c)
                
            model.train_generator()
            G_optimizer.zero_grad()
            prediction = model(true_data)
            loss = criterion(prediction, args.model_type, model.parameters())
            loss.backward()
            G_optimizer.step()
                
        
   
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

