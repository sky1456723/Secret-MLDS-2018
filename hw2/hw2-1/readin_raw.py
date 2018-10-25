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
    print("Loading Data")
    ###Load data
    data_direc = directory
    # Training part
    train_labels, train_feat_dict = [], {}
    with open(data_direc + "training_label.json", "r") as f_train:
        train_labels = json.loads(f_train.read())
        for i in train_labels:
            train_feat_dict[ i['id'] ] = np.load(data_direc + 'training_data/feat/' + i['id'] + '.npy')
    #train_labels = train_labels[:len(train_labels)//100]
    print(len(train_labels))
    '''        
    # Testing part
    test_labels, test_feat_dict = [], {}
    with open(data_direc + "testing_label.json", "r") as f_test:
        test_labels = json.loads(f_test.read())
        for i in test_labels:
            test_feat_dict[ i['id'] ] = np.load(data_direc + 'testing_data/feat/' + i['id'] + '.npy')
    '''
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
    print("Finish count words")
    one_hot_len = len(word_dict)+3
    
    for num, key in enumerate(word_dict.keys()):
        one_hot = [0]*one_hot_len
        one_hot[num] = 1
        word_dict[key] = one_hot
    
    one_hot =  [0]*one_hot_len
    one_hot[-1] = 1
    word_dict['<EOS>'] = one_hot
    one_hot =  [0]*one_hot_len
    one_hot[-2] = 1
    word_dict['<BOS>'] = one_hot
    one_hot =  [0]*one_hot_len
    one_hot[-3] = 1
    word_dict['<UNK>'] = one_hot
    #Finish build dictionary
    
    max_len = 0
    train_x = []
    train_y = []
    sentence_len = []
    for data_num in range(len(train_labels)):
        print("Process data ",data_num," to list")
        caption = train_labels[data_num]['caption']
        video_id = train_labels[data_num]['id']
        choose_caption = []
        choose_length = []
        for phrase in caption:
            if len(phrase) > max_len:
                max_len = len(phrase)
            phrase_list = [] #contain one-hot vectors of words
            for word in phrase:
                try:
                    phrase_list.append(word_dict[word])
                except:
                    phrase_list.append(word_dict['<UNK>'])
            choose_caption.append(phrase_list)
            choose_length.append(len(phrase))
        train_y.append(choose_caption)
        sentence_len.append(choose_length)
        train_x.append(train_feat_dict[video_id]) 
    
    for i in range(len(train_y)):
        for j in range(len(train_y[i])):
            pad_len = max_len - sentence_len[i][j]
            for k in range(pad_len):
                pad = [0]*one_hot_len
                train_y[i][j].append(pad)
    
    print("Max len = ", max_len )
    
    training_caption = []
    training_caption_len = []
    for i in range(len(train_y)):
        choose = np.random.randint(len(train_y[i]))
        training_caption.append(train_y[i][choose])
        training_caption_len.append(sentence_len[i][choose])
    
    train_x = np.array(train_x)
    training_caption = np.array(training_caption)
    training_caption_len = np.array(np.reshape(training_caption_len, (-1,1)))
    print(training_caption.shape)
    print(training_caption_len.shape)
    print("Convert to Tensor")
    tensor_train_x = torch.Tensor(train_x)
    tensor_train_y = torch.Tensor(training_caption)
    tensor_seq_len = torch.LongTensor(training_caption_len)
    print("Load to dataset")
    dataset = torch.utils.data.TensorDataset(tensor_train_x, tensor_train_y, tensor_seq_len)
    dataloader = torch.utils.data.DataLoader(dataset = dataset, batch_size = batch_size, shuffle=True)
    return dataloader, one_hot_len, max_len
    




