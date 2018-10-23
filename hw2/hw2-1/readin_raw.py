#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 14:51:07 2018

@author: jeffchen
"""

import numpy as np
import torch
import torch.utils.data
import json

###split a sentence 
def sent2words(sent):
    res_para = [j.split(' ') for j in (chr(ord(sent[0])+32) + sent[1:-1]).split(',')]
    res_tok = res_para.pop(0)
    while len(res_para) != 0:
        res_tok.extend([','] + res_para.pop(0)[1:])
    #res_tok += ['.']
    for j in range(len(res_tok)):
        if res_tok[j][-2:] == "'s" and res_tok[j] != "'s":
            res_tok[j] = res_tok[j][:-2]
            res_tok.insert(j+1,"'s")
    return res_tok



def generate_dataloader(directory = "./MLDS_hw2_1_data/", batch_size = 64):
    ###Load data
    data_direc = directory
    # Training part
    train_labels, train_feat_dict = [], {}
    with open(data_direc + "training_label.json", "r") as f_train:
        train_labels = json.loads(f_train.read())
        for i in train_labels:
            train_feat_dict[ i['id'] ] = np.fromfile(data_direc + 'training_data/feat/' + i['id'] + '.npy')[-80*4096:].reshape(80,4096)
    
    #train_labels = train_labels[:len(train_labels)//100]
            
    # Testing part
    test_labels, test_feat_dict = [], {}
    with open(data_direc + "testing_label.json", "r") as f_test:
        test_labels = json.loads(f_test.read())
        for i in test_labels:
            test_feat_dict[ i['id'] ] = np.fromfile(data_direc + 'testing_data/feat/' + i['id'] + '.npy')[-80*4096:].reshape(80,4096)
    
    ###Build word dictionary and pop some words
    word_dict = {}
    
    for i in range(len(train_labels)):
        caption_list = train_labels[i]['caption']
        for j in range(len(caption_list)):
            tokens = sent2words(caption_list[j])
            for k in tokens:
                if k not in word_dict.keys():
                    word_dict[k] = 1
                else:
                    word_dict[k] += 1
            tokens.append('<EOS>')
            train_labels[i]['caption'][j] = tokens
    to_pop = []
    for i in word_dict.keys():
        if word_dict[i] < 3:
            to_pop.append(i)
    for i in to_pop:
        word_dict.pop(i)
    
    one_hot_len = len(word_dict)+3
    
    for num, key in enumerate(word_dict.keys()):
        one_hot = [0]*one_hot_len
        one_hot[num] = 1
        word_dict[key] = one_hot
    
    one_hot = [0]*one_hot_len
    one_hot[-1] = 1
    word_dict['<EOS>'] = one_hot
    one_hot = [0]*one_hot_len
    one_hot[-2] = 1
    word_dict['<BOS>'] = one_hot
    one_hot = [0]*one_hot_len
    one_hot[-3] = 1
    word_dict['<UNK>'] = one_hot
    #Finish build dictionary
    
    max_len = 0
    train_x = []
    train_y = []
    sentence_len = []
    for data_num in range(len(train_labels)):
        caption = train_labels[data_num]['caption']
        video_id = train_labels[data_num]['id']
        for phrase in caption:
            if len(phrase) > max_len:
                max_len = len(phrase)
            phrase_list = [] #contain one-hot vectors of words
            for word in phrase:
                try:
                    phrase_list.append(word_dict[word])
                except:
                    phrase_list.append(word_dict['<UNK>'])
            train_x.append(train_feat_dict[video_id])
            train_y.append(phrase_list)
            sentence_len.append(len(phrase))
    
    for i in range(len(train_y)):
        pad_len = max_len - sentence_len[i]
        for j in range(pad_len):
            train_y[i].append([0]*one_hot_len)
    
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    sentence_len = np.array(np.reshape(sentence_len, (-1,1)))
    tensor_train_x = torch.Tensor(train_x)
    tensor_train_y = torch.LongTensor(train_y)
    tensor_seq_len = torch.LongTensor(sentence_len)
    
    dataset = torch.utils.data.TensorDataset(tensor_train_x, tensor_train_y, tensor_seq_len)
    dataloader = torch.utils.data.DataLoader(dataset = dataset, batch_size = batch_size, shuffle=True)
    return dataloader, one_hot_len, max_len
    




