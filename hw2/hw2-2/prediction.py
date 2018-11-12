#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 15:51:04 2018

@author: jimmy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import gensim
import numpy as np
import Seq2seqModel as seq2seq
import argparse
import time

start_time = time.time()


### DEVICE CONFIGURATION ###

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

### MODEL ###

class Model(nn.Module):
    def __init__(self, encoder_layer=1, decoder_layer=1, w2v_size=250, one_hot_size=71347, optimizer='NONE'):
        super(Model, self).__init__()
        self.global_epoch = 0
        self.w2v_size = w2v_size
        self.hidden_size = 128
        self.one_hot_size = one_hot_size
        self.optimizer = optimizer
        self.batch_size = 20
        
        self.encoder_embedding = nn.Linear(self.w2v_size, self.hidden_size).to(device)
        self.decoder_embedding = nn.Linear(self.one_hot_size, self.hidden_size).to(device)
        
        self.encoder = seq2seq.EncoderRNN(hidden_size=self.hidden_size,
                                          embedding=self.encoder_embedding,
                                          n_layers=encoder_layer,
                                          dropout=0.5).to(device)
        self.decoder = seq2seq.LuongAttnDecoderRNN(attn_model='dot',
                                                   hidden_size=self.hidden_size,
                                                   embedding=self.decoder_embedding,
                                                   output_size=self.one_hot_size,
                                                   n_layers=decoder_layer).to(device)    

    def forward(self, x, x_len, y):
        # x: (batch size, max length, w2v dim)
        # x_len: (batch_size, 1)
        # input: (seq length, batch size, one hot size)
        # encoder_outputs: (max length, batch size, hidden size) - bidirectional
        # decoder_output: (batch size, one hot size)
        x = x.transpose(0, 1) # encoder requires (max_length, batch_size, w2v_dim)
        x_len = x_len.long().squeeze() # to int64
        y = y.long()
        pred_max_len = y.shape[1]
        decoder_outputs_list = []
        
        ### ENCODER ###
        encoder_outputs, encoder_hidden = self.encoder(x, x_len)

        ### INPUT TENSORS ###
        empty_input = torch.zeros((1, self.batch_size, self.one_hot_size), dtype=torch.float32).to(device)
        randn_input = torch.randn((1, self.batch_size, self.one_hot_size), dtype=torch.float32).to(device)
        
        ### FIRST DECODER INPUT ###
        decoder_output, decoder_hidden = self.decoder(randn_input, encoder_hidden[0:1], encoder_outputs)
        decoder_outputs_list.append(decoder_output)
        
        ### ONE-HOT TEACHER FORCING ###
        one_hot_sentence_list = []
        for i in range(self.batch_size):
            # convert index of words into one-hot vectors
            one_hot_list = []
            for j in range(pred_max_len):
                one_hot = torch.zeros((self.one_hot_size), dtype=torch.float32).to(device)
                one_hot[y[i, j].item()] = 1.0
                one_hot_list.append(one_hot)
            one_hot_sentence = torch.stack(one_hot_list, dim=0) # tensor of shape (max_length, one_hot_size)
            one_hot_sentence_list.append(one_hot_sentence)
        teacher_forcing_input = torch.stack(one_hot_sentence_list, dim=1) # shape (max_length, batch_size, one_hot_size)
        
        ### FOLLOWING DECODER INPUT ###
        for step in range(pred_max_len - 1):
            # length of prediction = max length of answer
            decoder_output, decoder_hidden = self.decoder(teacher_forcing_input[step:(step+1), :, :], decoder_hidden, encoder_outputs)
            decoder_outputs_list.append(decoder_output)
        
        # Turn a list of tensors into a tensor, e.g. n tensors of shape (x, y, z) stacked at dim=0
        # returns a tensor of shape (n, x, y, z)
        # Lastly, make sure that we return a tensor of shape (batch_size, max_length, one_hot_size)
        return torch.stack(decoder_outputs_list, dim=1)


def main(arg):
    model = torch.load(arg.model).to(device)
    wv = gensim.models.KeyedVectors.load(arg.word2vec)
    one_hot_size = len(wv.index2entity)
    ### Process Data ###
    if arg.direct:
        data = arg.input
    else:
        file = open(arg.txtdata)
        data = file.readlines()
        file.close()
        for sentence_num in range(len(data)):
            data[sentence_num] = data[sentence_num].replace('\n', '')
            data[sentence_num] = data[sentence_num].split()
            for word_num in range(len(data[sentence_num])):
                word = data[sentence_num][word_num]
                try:
                    data[sentence_num][word_num] = wv[word]
                except:
                    #print("Has <UNK>")
                    data[sentence_num][word_num] = wv['<UNK>']
            data[sentence_num] = np.array(data[sentence_num])
    
    ### Decoder First Input ###
    
    empty_input = torch.zeros((1, 1, one_hot_size), dtype=torch.float32).to(device)
    randn_input = torch.randn((1, 1, one_hot_size), dtype=torch.float32).to(device)
    
    ### Predict ###
    ans_list = []
    if arg.direct:
        pass
    else:
        for one_data in data:
            max_ans_len = 10
            input_data = torch.Tensor(one_data).reshape(1, len(one_data), -1).transpose(0,1).to(device)
            data_len = torch.Tensor([len(one_data)]).to(device).long()
            ### ENCODE ###
            encoder_outputs, encoder_h = model.encoder(input_data, data_len)
            
            ### FIRST DECODER INPUT ###
            output_words = []
            decoder_output, decoder_hidden = model.decoder(empty_input, encoder_h[0:1], encoder_outputs)
            one_hot_index = decoder_output.argmax(dim=1).item()
            output_words.append(wv.index2entity[one_hot_index])
            count_len = 1
            while output_words[-1] != '<EOS>' and count_len<max_ans_len:
                ### MUST change
                #last_word = torch.Tensor(wv[output_words[-1]]).reshape(1,1,-1).to(device)
                last_word = decoder_output.reshape(1,1,-1)
                decoder_output, decoder_hidden = model.decoder(last_word, 
                                                               decoder_hidden, 
                                                               encoder_outputs)
                one_hot_index = decoder_output.argmax(dim=1).item()
                output_words.append(wv.index2entity[one_hot_index])
                count_len+=1
            ans_list.append(output_words)
        output_file = open(arg.output, 'w')
        for sentence in ans_list:
            for word in sentence:
                output_file.write(word)
            output_file.write('\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='chatbot')
    parser.add_argument('--model', type=str, default = './model/model_800000.pkl')
    parser.add_argument('--txtdata', type=str)
    parser.add_argument('-d','--direct', action='store_true')
    parser.add_argument('--input', type=str)
    parser.add_argument('--output', type=str)
    parser.add_argument('--word2vec', type=str,
                        default = './word2vec/word2vec_wv_Jeff.wv')
    
    arg = parser.parse_args()
    main(arg)

