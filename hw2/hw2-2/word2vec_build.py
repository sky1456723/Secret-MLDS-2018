# -*- coding: utf-8 -*-
"""
Created         on Sun Oct 29 02:44:57 2018
First created   on Sat Oct 27 17:35:25 2018
Second modified on Fri Nov  9 17:15:44 2018
Last modified   on Fri Nov  9 17:48:49 2018

@author: Jeff Chen
"""

from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec
import numpy as np
import codecs
from collections import defaultdict
from pprint import pprint

directory = "/home/jovyan/ntulee/b05901009/hw2-2/mlds_hw2_2_data/"
unknown_threshold = 5
iterations = 10
print(">>> current directory: \"{}\"<<<".format(directory))
print("### PROCESS  STARTED ###")

# Sec. 1: Loading conversations
print("Loading conversations...")
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

print("Adding <UNK> and <EOS>...")
# Sec. 2: Low frequent words elimination and attachment of sentence endings
freq = defaultdict(int)
for text in contents:
    for token in text:
        freq[token] += 1
texts = []
for sent in contents:
    _sent = []
    for token in sent:
        if not freq[token] >= unknown_threshold:
            _sent.append("<UNK>")
        else:
            _sent.append(token)
    texts.append(_sent + ['<EOS>'])
contents = texts
del texts, _sent

print("Start training...")
# Sec. 3: Generating models
model = Word2Vec(contents, size=250, window=5, min_count=unknown_threshold, workers=4, iter=iterations)
print("Saving the model.")
model.save("./word2vec_run_by_Jeff.model")
vocab = list(dict(model.wv.vocab.items()).keys())
print('The size of vocabulary =', len(vocab), ".")

print("###PROCESS TERMINATED###")
