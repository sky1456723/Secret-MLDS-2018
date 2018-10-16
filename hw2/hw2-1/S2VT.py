#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 09:19:24 2018

@author: jimmy
"""

import torch



class S2VT(torch.nn.Module):
    def __init__(self, input_dim, batch, encoder_unit, decoder_unit,
                 encoder_layer, decoder_layer):
        super(S2VT, self).__init__()
        self.batch_size = batch
        self.input_size = input_dim
        self.output_size = decoder_unit
        #To change between encoding & decoding
        self.encoder_stage = True
        self.decoder_stage = False
        
        #left to implement attention and schedule sampling
        self.schedule_sample_rate = 0
        self.attention_vector = None
        
        #save after encoding
        self.encoding_code = None
        self.encoding_c = None
        self.encoder_h_size = encoder_unit
        self.decoder_h = None
        self.decoder_c = None
        self.decoder_h_size = decoder_unit
        ################
        #Can change initialize
        ################
        
        # shape = (num_layer, batch, hidden_size)
        self.encoder_h0 = torch.nn.init.xavier_uniform_(
                          torch.Tensor(encoder_layer, batch, encoder_unit)) 
        
        self.encoder_c0 = torch.nn.init.xavier_uniform_(
                          torch.Tensor(encoder_layer, batch, encoder_unit))
        self.encoder = torch.nn.LSTM(input_size = input_dim,
                                     hidden_size = encoder_unit,
                                     num_layers = encoder_layer,
                                     dropout = 0)
        
        self.decoder_input_size = (decoder_unit + encoder_unit)
        ################
        #Can change initialize
        #shape : (num_layer, batch, hidden_size)
        ################
        self.decoder_h0 = torch.nn.init.xavier_uniform_(
                          torch.Tensor(decoder_layer, batch, decoder_unit)) 
        self.decoder_c0 = torch.nn.init.xavier_uniform_(
                          torch.Tensor(decoder_layer, batch, decoder_unit))
        
        self.decoder = torch.nn.LSTM(input_size = self.decoder_input_size,
                                     hidden_size = decoder_unit,
                                     num_layers = decoder_layer,
                                     dropout = 0)
    def forward(self, input_data):
        #input_data shape: (sequence_len, batch, input_size)
        #output_data shape: (sequence_len, batch, input_size)
        if self.encoder_stage:
            print("Encoding")
            encoder_output, (encoder_h_n, encoder_c_n) = self.encoder(input_data, (self.encoder_h0, self.encoder_c0))
            self.encoding_code = encoder_h_n
            self.encoding_c = encoder_c_n
            
            padding_tag = torch.zeros((80, self.batch_size, self.decoder_h_size), 
                                      dtype= torch.float32)
            
            decoder_input = torch.cat( (encoder_output, padding_tag), dim = 2)
            
            decoder_output, (decoder_h_n, decoder_c_n) = self.decoder(decoder_input, (self.decoder_h0, self.decoder_c0))
            self.decoder_h = decoder_h_n
            self.decoder_c = decoder_c_n
            
            return decoder_output
        
        elif self.decoder_stage:
            print("Decoding")
            encoder_output, (encoder_h_n, encoder_c_n) = self.encoder(input_data, (self.encoding_code, self.encoding_c))
            
            ###############
            #to add bos tag
            #shape : (1, batch_size, feature)
            ###############
            bos_tag = torch.zeros((1, self.batch_size, self.decoder_h_size), 
                                      dtype= torch.float32) 
            
            temp_h = None
            temp_c = None
            
            First_flag = True
            decoder_output = []
            for one_time in encoder_output:
                if First_flag:
                    
                    one_time = one_time.reshape((1, self.batch_size, -1)) #-1 -> hidden_size
                    decoder_input = torch.cat((one_time, bos_tag), dim = 2)
                    output, (h, c) = self.decoder(decoder_input, (self.decoder_h, self.decoder_c))
                    temp_h = h
                    temp_c = c
                    ###########################
                    #Can do something to output
                    ###########################
                    decoder_output.append(output)
                    First_flag = False
                else:
                    one_time = one_time.reshape((1, self.batch_size, -1))
                    
                    decoder_input = torch.cat((one_time, decoder_output[-1]), dim = 2)
                    output, (h, c) = self.decoder(decoder_input, (temp_h, temp_c))
                    temp_h = h
                    temp_c = c
                    ###########################
                    #Can do something to output
                    ###########################
                    decoder_output.append(output)
            return decoder_output
    
test_data = torch.zeros((80,32,4096), dtype=torch.float32)
test = S2VT(4096, 32, 256, 256, 1, 1)            
test(test_data)
test.encoder_stage = False
test.decoder_stage = True
ans = test(test_data)
            