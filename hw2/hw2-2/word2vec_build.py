# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 02:44:57 2018
First created on Sat Oct 27 17:35:25 2018

@author: Jeff Chen
"""

from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec
import numpy as np
#import matplotlib.pyplot as plt
import codecs

directory = "/home/jovyan/ntulee/b05901009/hw2-2/mlds_hw2_2_data/"

# Sec. 1: Loading coversation
with open(directory + "clr_conversation.txt", "r") as f:
    sentences = [dialogue.split('\n') for dialogue in f.read().split('+++$+++')]
contents = []
for i in sentences:
    if i[0] == '':
        i = i[1:]
    if i[-1] == '':
        i = i[:-1]
    contents.extend(i)
contents = [sentence.split() for sentence in contents]
print("Start training model")
# Sec. 2: Generating models
# dictionary = gensim.corpora.Dictionary(contents)
# path = get_tmpfile("word2vec.model")
model = Word2Vec(contents, size=250, window=5, min_count=3, workers=4)
model.save(directory + "word2vec_run_by_Wei_Tsung.model")
# model_try.train(["我們", "該", "讓", "他們", "回到", "家鄉"], total_examples=1, epochs=1)
vocab = list(dict(model.wv.vocab.items()).keys())
print('The size of vocabulary =', len(vocab))
# for j in vocab:
#     print(j, model.wv[j])
# modelp = Word2Vec.load(directory + "word2vec.model")
# modelp.wv['say']