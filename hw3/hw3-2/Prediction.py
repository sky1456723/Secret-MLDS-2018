# -*- coding: utf-8 -*-
"""
Created on Sat Dec 22 01:20:22 2018

@author: Jeff
"""
import torch
import numpy as np
import glob
import os
import cv2
import pandas as pd
import json
import argparse
import Model

def get_feature_two_hots(descr):
    if os.path.isfile('./feature_dicts.json'):
        with open('./feature_dicts.json', 'r') as f:
            hair_dict, eye_dict = json.load(f)
    else:
        tag = np.array(pd.read_csv('../hw3-1/data/extra_data/tags.csv'))
        tag_head = np.array(['aqua hair aqua eyes'])
        tags = np.insert(tag[:,1], 0, values=tag_head, axis=0)
        print(tags)
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
        with open('feature_dicts.list', 'w') as f:
            json.dump([hair_dict, eye_dict], f)
    
    tmp = descr.split(' ')
    assert tmp[1] == 'hair' and tmp[3] == 'eyes'
    hair = np.zeros(len(hair_dict))
    hair[hair_dict[tmp[0]]] = 1
    eye = np.zeros(len(eye_dict))
    eye[eye_dict[tmp[2]]] = 1
    return [hair,eye]

def save_imgs(model, output_path):
    import matplotlib.pyplot as plt
    plt.switch_backend('agg')

    r, c = 5, 5
    # z = torch.randn((c_z.shape[0], self.z_dim), dtype=torch.float32).cuda()
    noise = np.random.normal(0, 1, (r * c, model.z_dim))
    noise = torch.Tensor(noise).cuda()
    
    # gen_imgs should be shape (25, 64, 64, 3)

    with open('testing_tags.txt', 'r') as f:
        Ttags = f.readlines()
        Ttags = [i.split('\n')[0].split(',') for i in Ttags]
        Ttags = dict([int(j[0]), j[1]] for j in Ttags)

    _feat = []
    for p in range(25):
        _hair, _eyes = get_feature_two_hots(Ttags[p + 1])
        _hair, _eyes = torch.Tensor(_hair), torch.Tensor(_eyes)
        _feat.append(torch.cat((_hair, _eyes)))
    _feat = torch.stack(_feat).cuda()
    gen_imgs = model.infer(c_z=_feat, noise_z=noise)
    gen_imgs = gen_imgs.permute(0, 2, 3, 1)
    gen_imgs = gen_imgs.cuda()
    
    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i,j].imshow(gen_imgs[cnt, :,:,:])
            axs[i,j].axis('off')
            cnt += 1
    plt.show()
    fig.savefig(output_path)
    plt.close()

def main(args):
    model = torch.load(args.model)
    if os.path.isfile(args.output):
        os.remove(args.output)
    save_imgs(model, args.output) 
    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Conditional GAN HW3-2')
    parser.add_argument("model", type=str)
    parser.add_argument("output", type=str)
    args = parser.parse_args()
    main(args)