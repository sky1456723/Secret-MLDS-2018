#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 14:51:07 2018

@author: jeffchen
"""

import numpy as np
import torch
import torch.utils.data as data
import json
"""
import S2VT



mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = np.reshape(x_train, (-1,784)) / 255
x_test = np.reshape(x_test, (-1,784)) / 255
batch = [64, 1024]
epoch = 10
model_number = 2
tensor_train_x = torch.Tensor(x_train)
tensor_train_y = torch.LongTensor(y_train)
dataset = torch.utils.data.TensorDataset(tensor_train_x, tensor_train_y)

tensor_test_x = torch.Tensor(x_test)
tensor_test_y = torch.LongTensor(y_test)
test_dataset = torch.utils.data.TensorDataset(tensor_test_x, tensor_test_y)



#ans_linux = np.load(featfile)
#ans_mac = np.fromfile(featfile)[-80*4096:].reshape(80,4096)
"""

def sent2words(sent):
    res_para = [j.split(' ') for j in (chr(ord(sent[0])+32) + sent[1:-1]).split(',')]
    res_tok = res_para.pop(0)
    while len(res_para) != 0:
        res_tok.extend([','] + res_para.pop(0)[1:])
    res_tok += ['.']
    for j in range(len(res_tok)):
        if res_tok[j][-2:] == "'s" and res_tok[j] != "'s":
            res_tok[j] = res_tok[j][:-2]
            res_tok.insert(j+1,"'s")
    return res_tok

data_direc = "/Users/jeffchen/Desktop/Seq2seq/MLDS_hw2_1_data/"
# Training part
train_labels, train_feat_dict = [], {}
with open(data_direc + "training_label.json", "r") as f_train:
    train_labels = json.loads(f_train.read())
    for i in train_labels:
        train_feat_dict[ i['id'] ] = np.fromfile(data_direc + 'training_data/feat/' + i['id'] + '.npy')[-80*4096:].reshape(80,4096)

        
# Testing part
test_labels, test_feat_dict = [], {}
with open(data_direc + "testing_label.json", "r") as f_test:
    test_labels = json.loads(f_test.read())
    for i in test_labels:
        test_feat_dict[ i['id'] ] = np.fromfile(data_direc + 'testing_data/feat/' + i['id'] + '.npy')[-80*4096:].reshape(80,4096)




