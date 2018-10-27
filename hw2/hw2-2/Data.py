# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 14:38:06 2018

@author: Jimmy
"""

import torch
import torch.utils.data
import numpy as np

class ChatbotDataset(torch.utils.data.Dataset):
    def __init__(self, data_x, data_y, from_file = False):
        #Expected data shape like:(data_num, data_len)
        #Can be a list of numpy array or something like this
        if from_file:
            self.data = np.load(data_x)
            self.label = np.load(data_y)
        else:
            self.data = data_x
            self.label = data_y
    def __getitem__(self, index):
        data_seq_len = len(self.data[index])
        label_seq_len = len(self.label[index])
        return [self.data[index], self.label[index],
                data_seq_len, label_seq_len]
    def __len__(self):
        return len(self.data)
    
#In dataloader, when we iter in it,
#It will execute __next__ of DataLoaderIter,
#And __next__ will call Dataset[index] (__getitem__)
#and get #batch time to save the return in a list
def collate_fn(batch_data):
    #batch_data should be a list of data
    #data would be a list of [list/np.array, list/np.array,
    #                         int, int]
    #each list/np.array shape like : (seq_len, word_vec_size)
    #collate_fn return: Tensor of (x, y,
    #                              unpadded_data_seq_len,
    #                              unpadded_label_seq_len)
    #shape of x : (batch_size, padded_data_seq_len, word_vec_size)
    #shape of y : (batch_size, padded_label_seq_len, one_hot_size)
    data_type = str(type(batch_data[0][0]))
    if data_type == "<class 'numpy.ndarray'>":
        #padding data
        one_hot_size = batch_data[0][1].shape[1]
        word_vec_size = batch_data[0][0].shape[1]
        max_data_len = max(batch_data, key=lambda x:x[2])[2]
        max_label_len = max(batch_data, key=lambda x:x[3])[3]
        for data_num in range(len(batch_data)):
            data_len = batch_data[data_num][2]
            if max_data_len - data_len > 0:
                pad = [[0]*word_vec_size]*(max_data_len - data_len)
                pad = np.array(pad)
                unpadded = batch_data[data_num][0]
                batch_data[data_num][0] = np.concatenate((unpadded, pad),
                                                         axis = 0)
        for label_num in range(len(batch_data)):
            label_len = batch_data[label_num][3]
            if max_label_len - label_len > 0:
                pad = [[0]*one_hot_size]*(max_label_len - label_len)
                pad = np.array(pad)
                unpadded = batch_data[label_num][1]
                batch_data[label_num][1] = np.concatenate((unpadded, pad),
                                                         axis = 0)
        
        batch_x = [torch.Tensor(data[0]) for data in batch_data]
        batch_x = torch.stack(batch_x) #shape:(batch, seq_len, max_data_len)
        batch_y = [torch.Tensor(data[1]) for data in batch_data]
        batch_y = torch.stack(batch_y) #shape:(batch, seq_len, max_label_len)
        data_seq_len = [torch.Tensor([data[2]]) for data in batch_data]
        label_seq_len = [torch.Tensor([data[3]]) for data in batch_data]
        data_seq_len = torch.stack(data_seq_len) #shape : (batch, 1)
        label_seq_len = torch.stack(label_seq_len) #shape : (batch, 1)
        return batch_x, batch_y, data_seq_len, label_seq_len
    elif data_type == "<class 'list'>":
        #padding data
        #batch[0][0] : list of shape (seq_len, word_vec_size) 
        #batch[0][1] : list of shape (seq_len, one_hot_size)
        
        one_hot_size = len(batch_data[0][1][0])
        word_vec_size = len(batch_data[0][0][0])
        max_data_len = max(batch_data, key=lambda x:x[2])[2]
        max_label_len = max(batch_data, key=lambda x:x[3])[3]
        for data_num in range(len(batch_data)):
            data_len = batch_data[data_num][2]
            for i in range(max_data_len - data_len):
                batch_data[data_num][0].append([0]*word_vec_size)
        for label_num in range(len(batch_data)):
            label_len = batch_data[label_num][3]
            for i in range(max_label_len - label_len):
                batch_data[label_num][1].append([0]*one_hot_size)
        
        batch_x = [torch.Tensor(data[0]) for data in batch_data]
        batch_x = torch.stack(batch_x) #shape:(batch, seq_len, max_data_len)
        batch_y = [torch.Tensor(data[1]) for data in batch_data]
        batch_y = torch.stack(batch_y) #shape:(batch, seq_len, max_label_len)
        data_seq_len = [torch.Tensor([data[2]]) for data in batch_data]
        label_seq_len = [torch.Tensor([data[3]]) for data in batch_data]
        data_seq_len = torch.stack(data_seq_len) #shape : (batch, 1)
        label_seq_len = torch.stack(label_seq_len) #shape : (batch, 1)
        return batch_x, batch_y, data_seq_len, label_seq_len
    
###TEST CODE###
'''
data = [np.array([[0,1],[0,2],[0,3]]), np.array([[0,4],[0,5]]),
        np.array([[0,1],[0,2],[0,3],[0,4]]), np.array([[9,9]])]
label = [np.array([[0,0,0,0,1],[0,0,0,0,2],[0,0,0,0,3],[0,0,0,0,4]]),
         np.array([[0,0,0,0,5]]),
         np.array([[0,0,0,0,1], [0,0,0,0,2]]),
         np.array([[0,0,0,0,3], [0,0,0,0,4],[0,0,0,0,5]])]

list_data = [ [[0,1],[0,2]], [[0,3],[0,4],[0,5]],
              [[0,1],[0,2],[0,3],[0,4]], [[9,9]] ]
list_label = [ [[0,0,0,0,1],[0,0,0,0,2],[0,0,0,0,3],[0,0,0,0,4]],
               [[0,0,0,0,5]],
               [[0,0,0,0,1], [0,0,0,0,2]],
               [[0,0,0,0,3], [0,0,0,0,4],[0,0,0,0,5]] ]
dataset = ChatbotDataset(data, label)
dataloader = torch.utils.data.DataLoader(dataset, batch_size = 3,
                                         collate_fn = collate_fn)
for x,y,z,v in dataloader:
    print(x.shape)
    print(y.shape)
'''
    