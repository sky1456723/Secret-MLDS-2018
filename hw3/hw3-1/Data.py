# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 20:38:00 2018

@author: Jason
"""
import torch
import torch.utils.data
import numpy as np
import glob
import cv2
import pandas as pd

class GANDataset(torch.utils.data.Dataset):
    """
    __len__ will return the # of data
    """
    def __init__(self, data_x, data_y, from_file = False):
        if from_file:
            self.data = np.load(data_x)
            self.label = np.load(data_y)
        else:
            self.data = data_x
            self.label = data_y
    def __getitem__(self, index):
        x = torch.tensor(self.data[index])
        feature1 = torch.tensor(self.label[index][0])
        feature2 = torch.tensor(self.label[index][1])
        return [x,feature1,feature2]
    def __len__(self):
        return len(self.data)

def GetDataset():
    # put the img path into glob to get all the img name
    imgname = glob.glob(r'./data/extra_data/images/*.jpg')

    img_list = []
    for item in imgname:
        im = cv2.imread(item)
        img_list.append(np.transpose(im , (2,0,1)) )

    tag = np.array(pd.read_csv('./data/extra_data/tags.csv'))
    tag_head = np.array(['aqua hair aqua eyes'])
    tags = np.insert(tag[:,1], 0, values=tag_head, axis=0)

    # get two feature of img: eye color and hair color
    hair_set = set()
    eye_set = set()
    for it in tags:
        tmp = it.split(' ')
        hair_set.add(tmp[0])
        eye_set.add(tmp[2])
    hair_dict = {}
    eye_dict = {}
    i = 0
    for it in hair_set:
        hair_dict[it] = i
        i = i + 1
    i = 0
    for it in eye_set:
        eye_dict[it] = i
        i = i + 1

    # img_list:
    # shape:36740*(64, 64, 3), type:np.ndarray
    # feature:
    # shape: (36740,2), type:np.ndarray
    feature = []
    for it in tags:
        tmp = it.split(' ')
        hair = np.zeros(len(hair_dict))
        hair[hair_dict[tmp[0]]] = 1
        eye = np.zeros(len(eye_dict))
        eye[eye_dict[tmp[2]]] = 1
        feature.append([hair,eye])
    feature = np.array(feature)
    # print(feature[0],len(feature),type(feature),feature.shape)
    img_list = np.array(img_list)
    img_list = img_list/255.0
    # print(len(img_list),type(img_list),img_list.shape)

    dataset = GANDataset(img_list, feature)
    return dataset
    #dataloader = torch.utils.data.DataLoader(dataset, batch_size = 4)
    '''
    for i, (x,f1,f2) in enumerate(dataloader):
        print(i)
        print(type(x),x.shape)
        print(type(f1),f1.shape)
        print(type(f2),f2.shape)
    #     if(i>2):
    #         break
    '''

    # in each iteration, dataloader will return:
    # x: batch of images
    # <class 'torch.Tensor'> torch.Size([batch size, 64, 64, 3])
    # f1: hair color 
    # <class 'torch.Tensor'> torch.Size([batch size, 12])
    # f2: eye colar 
    # <class 'torch.Tensor'> torch.Size([batch size, 10])


