#!/usr/bin/env python3
# -*- coding: utf-8
"""
Created on Mon Dec 10  8:34:55 2018
 Redone on Sun Dec 16 23:02:55 2018
 
@author: Jeff
"""

import torch
import numpy as np
import argparse
import os
import Model
import Data
import time
import torch.nn.functional as F

start_time = time.time()

### DEVICE CONFIGURATION ###
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

### CRITERION ###
def criterion(prediction, flag, train_D, batch_size, epsilon=0, feat1=None, feat2=None):
    if flag == "Base":
        if train_D:
            #expected prediction[] : (batch, 1)
            #output : (1)
            loss = -1 * (torch.sum( torch.log(prediction['sig_true']+epsilon)) +
                         torch.sum( torch.log(1.0 - prediction['sig_false']+epsilon) ) )
            return loss/batch_size
        else:
            loss = -1 * torch.sum( torch.log(prediction['sig_false']+epsilon) )
            return loss/batch_size
    elif flag == "Info":
        if train_D:
            #expected prediction[] : (batch, 1)
            #output : (1)
            loss = -1 * (torch.sum( torch.log(prediction['sig_true']+epsilon)) +
                         torch.sum( torch.log(1.0 - prediction['sig_false']+epsilon) ) )
            
            #print(loss)
            reconstrcut_err = torch.sum( (prediction['latent_code']-prediction['reconstructed_code'])**2 )
            loss += reconstrcut_err
            #print(loss)
            return loss/batch_size
        else:
            loss = -1 * torch.sum( torch.log(prediction['sig_false']+epsilon) )
            reconstrcut_err = torch.sum( (prediction['latent_code']-prediction['reconstructed_code'])**2 )
            loss += reconstrcut_err
            return loss/batch_size
    elif flag == "LSGAN":
        a = -1
        b = 1
        c = 1
        if train_D:
            #expected prediction[] : (batch, 1)
            #output : (1)
            real = (prediction['raw_true']-b)**2
            gen = (prediction['raw_false']-a)**2
            loss = (torch.sum(real) + torch.sum(gen))
            return loss/batch_size
        else:
            gen = (prediction['raw_false']-c)**2
            loss = torch.sum(gen)
            return loss/batch_size
    elif flag == "WGAN":
        if train_D:
            #expected prediction[] : (batch, 1)
            #output : (1)
            real = prediction['raw_true']
            gen = prediction['raw_false']
            loss = -1 * (torch.sum(real) - torch.sum(gen))
            return loss/batch_size
        else:
            gen = prediction['raw_false']
            loss = -1 * torch.sum(gen)
            return loss/batch_size
    elif flag == "WGAN-GP":
        pass
    ###*** Part modified by Jeff begins ***###
    elif flag == "ACGAN":
        if train_D:
            #expected prediction[] : (batch, 1)
            #output : (1)
            loss = -1/batch_size * (torch.sum(torch.log(prediction['genuineness_true_sig']+epsilon)) +
                         torch.sum(torch.log(1.0 - prediction['genuineness_false_sig']+epsilon)))
            loss = -1/batch_size * (torch.sum(torch.log(prediction['matchedness_true_sig']+epsilon)) +
                         torch.sum(torch.log(1.0 - prediction['matchedness_false_sig']+epsilon)))
            #loss += F.mse_loss(prediction['reconstruction_code'], prediction['noise'], )
            
            return loss
        else: # train_G
            loss = -1/batch_size * (torch.sum(torch.log(prediction['genuineness_false_sig']+epsilon)))
            loss = -1/batch_size * (torch.sum(torch.log(prediction['matchedness_false_sig']+epsilon)))
            #loss += F.mse_loss(prediction['reconstruction_code'], prediction['noise'])
            
            return loss
    ###***  Part modified by Jeff ends  ***###
    elif flag == "ACGAN2":
        
        if train_D:
            #expected prediction[] : (batch, 1)
            #output : (1)
            G_category_code = prediction['category_code_false']
            loss = -1/batch_size * (torch.sum(torch.log(prediction['genuineness_true_sig']+epsilon)) +
                                     torch.sum(torch.log(1.0 - prediction['genuineness_false_sig']+epsilon)))
            loss += F.cross_entropy(prediction['matchedness12_true'],torch.argmax(feat1, dim = 1))
            loss += F.cross_entropy(prediction['matchedness10_true'],torch.argmax(feat2, dim = 1))
            loss += F.cross_entropy(prediction['matchedness12_false'],torch.argmax(G_category_code[:,:12], dim = 1))
            loss += F.cross_entropy(prediction['matchedness10_false'],torch.argmax(G_category_code[:,12:], dim = 1))
            return loss
        else: # train_G
            G_category_code = prediction['category_code_false']
            loss = -1/batch_size * (torch.sum(torch.log(prediction['genuineness_false_sig']+epsilon)))
            loss += F.cross_entropy(prediction['matchedness12_false'],torch.argmax(G_category_code[:,:12], dim = 1))
            loss += F.cross_entropy(prediction['matchedness10_false'],torch.argmax(G_category_code[:,12:], dim = 1))
            
            return loss

def main(args):
    ### DATA ###
    print("Loading Data")
    epoch = args.epoch_number
    batch_size = args.batch_size
    update_D = args.update_d
    update_G = args.update_g
    real_dataset = Data.GetDataset()
    real_dataloader = torch.utils.data.DataLoader(
            dataset = real_dataset,
            batch_size = batch_size,
            shuffle = True)
    print("Finish Loading Data")

    ### MODEL CREATION ###
    print(args.model_type)
    # print("WGAN? ", args.model_type == 'WGAN')
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
            model = Model.ACGAN(z_dim = 100, c_dim = 12 + 10, gen_leaky = args.gen_lk, dis_leaky = args.dis_lk,
                               gen_momentum = 0.9, dis_momentum = 0.9).to(device)
#             model = Model.ACGAN(90, 12 + 10, gen_leaky=0, dis_leaky=0).to(device)
            G_optimizer = torch.optim.Adam(model.G.parameters(),
                                           lr=args.lr, betas = (0.5,0.999))
            G_scheduler = torch.optim.lr_scheduler.StepLR(G_optimizer, step_size = 10, gamma = 0.8)
            
            D_optimizer = torch.optim.Adam(model.D.parameters(),
                                           lr=args.lr, betas = (0.5,0.999))
            
            D_scheduler = torch.optim.lr_scheduler.StepLR(D_optimizer, step_size = 10, gamma = 0.8)
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
                                           lr=args.lr, betas = (0.5,0.999))
            
            D_optimizer = torch.optim.Adam(model.D.parameters(),
                                           lr=args.lr, betas = (0.5,0.999))
            
            G_optimizer.load_state_dict(torch.load("./model/"+G_optim_name))
            D_optimizer.load_state_dict(torch.load("./model/"+D_optim_name))
    ### TRAIN ###
    print("Start Training")
    print("k value: ", update_D)  # k value is the update times of discriminator
    #WGAN_c = 0.01  # use for weight clipping of WGAN
    model = model.train()
    for e in range(epoch):
        print("Epoch :", e + 1)
        G_scheduler.step(e)
        D_scheduler.step(e)
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
                feature1 = feature1.to(device)
                feature2 = feature2.to(device)
                
            except:
                break
                
            model.train_discriminator()
            for d in range(update_D):
                D_optimizer.zero_grad()
                prediction_D = model(x=true_data, c = torch.cat((feature1, feature2), dim=1).to(torch.float32).to(device))
                loss_D = criterion(prediction_D, args.model_type, model.train_D, batch_size, epsilon=args.epsilon,
                                  feat1 = feature1, feat2 = feature2)
                batch_d_loss = loss_D.item()
                loss_D.backward()
                D_optimizer.step()
            
            model.train_generator()
            for g in range(update_G):
                G_optimizer.zero_grad()
                prediction_G = model(x=true_data, c=torch.cat((feature1, feature2), dim=1).to(torch.float32).to(device))
                loss_G = criterion(prediction_G, args.model_type, model.train_D, batch_size, epsilon=args.epsilon,
                                  feat1 = feature1, feat2 = feature2)
                batch_g_loss = loss_G.item()
                loss_G.backward()
                G_optimizer.step()
            
            print("Batch %d / %d: D loss: %4f, G loss: %4f" %
                  (batch_count, len(real_dataloader), batch_d_loss, batch_g_loss), end='\r')
            d_loss += batch_d_loss
            g_loss += batch_g_loss
            batch_count += 1
        print(); print()
        print("Epoch loss: D: %4f, G: %4f" % (d_loss/len(real_dataloader), g_loss/len(real_dataloader)))
            
        torch.save(model, model_name)
    torch.save(G_optimizer.state_dict(), G_optim_name)
    torch.save(D_optimizer.state_dict(), D_optim_name)
                
    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Conditional GAN HW3-2')
    parser.add_argument('--model_name', type=str, default='GAN_default')
    parser.add_argument('--model_type', type=str, default='ACGAN2')
    parser.add_argument('--epoch_number', '-e', type=int, default=20)
    parser.add_argument('--batch_size', '-b', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--update_d', type=int, default=1)
    parser.add_argument('--update_g', type=int, default=1)
    parser.add_argument('--gen_lk', type=float, default=0.1)
    parser.add_argument('--dis_lk', type=float, default=0.1)
    parser.add_argument('--epsilon', type=float, default=0)
    mutex = parser.add_mutually_exclusive_group(required = True)
    mutex.add_argument('--load_model', '-l', action='store_true', help='load a pre-existing model')
    mutex.add_argument('--new_model', '-n', action='store_true', help='create a new model')

    args = parser.parse_args()
    main(args)
