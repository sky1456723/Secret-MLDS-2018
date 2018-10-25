#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 14:31:18 2018

@author: jimmy
"""

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

def generate_word_dict(directory = "./MLDS_hw2_1_data/"):
    print("Loading Data")
    ###Load data
    data_direc = directory
    # Training part
    train_labels, train_feat_dict = [], {}
    with open(data_direc + "training_label.json", "r") as f_train:
        train_labels = json.loads(f_train.read())
        for i in train_labels:
            train_feat_dict[ i['id'] ] = np.load(data_direc + 'training_data/feat/' + i['id'] + '.npy')
    
    print(len(train_labels))
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
    int_to_dict = {}
    for num, key in enumerate(word_dict.keys()):
        int_to_dict[num] = key
    return int_to_dict

def generate_test_dataloader(directory = "./MLDS_hw2_1_data/", batch_size = 64):
    print("Loading Data")
    ###Load data
    data_direc = directory
      
    # Testing part
    test_labels, test_feat_dict = [], {}
    with open(data_direc + "testing_label.json", "r") as f_test:
        test_labels = json.loads(f_test.read())
        for i in test_labels:
            test_feat_dict[ i['id'] ] = np.load(data_direc + 'testing_data/feat/' + i['id'] + '.npy')
    print(len(test_labels))
    ###Build word dictionary and pop some words
    
    test_x = []
    video_id_list = []
    for data_num in range(len(test_labels)):
        print("Process data ",data_num," to list")

        video_id = test_labels[data_num]['id']
        video_id_list.append(video_id)
        test_x.append(test_feat_dict[video_id]) 
 
    test_x = np.array(test_x)
    
    print("Convert to Tensor")
    tensor_test_x = torch.Tensor(test_x)
    return tensor_test_x, video_id_list

def prediction(model_dir, input_x, id_list, int_to_word):
    file = open("output.txt",'w')
    model = torch.load(model_dir)
    for i in range(len(input_x)):
        sentence = ""
        ans = model(input_x[i])
        for j in range(len(ans)):
            one_word = int_to_word[np.argmax(ans[j])]
            if(one_word == '<EOS>'):
                break
            sentence += one_word
        file.write(id_list[i]+','+sentence)
    file.close()


        
        