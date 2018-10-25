#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 09:19:24 2018

@author: jimmy
"""

import torch

class Attention(torch.nn.Module):
    # Takes necessary information of dimensions of vectors
    # returns a dict of embedding networks
    # b: batch size
    # e_l: layers of encoder
    # e_u: units of encoder, thus hidden of encoder = (e_l, b, e_u) assuming one direction lstm
    # max: maximum sequence length, defaults to 1024
    # p: word size, thus prediction of decoder at one time step = (b, p)
    def __init__ (self, e_l, e_u, d_l, d_u, b, max, p):
        super(Attention, self).__init__()

        self.dropout = torch.nn.Dropout(0.2)
        self.linear_l1du_max = torch.nn.Linear((d_l + 1) * d_u, max)
        self.linear_2du_du = torch.nn.Linear(2 * d_u, d_u)
        self.softmax = torch.nn.Softmax(dim=1)
        self.relu = torch.nn.PReLU(num_parameters=d_l)

    def concat(self, x, y):
        # concatenate two tensors along last axis (which is usually num of units)
        assert len(x.shape) == len(y.shape), 'Attention.concat(): no matching tensor ranks.'
        return torch.cat((x, y), -1)

    def bvm(self, BS, SBH):
        # eliminate S from both tensors, returns LBH
        B1S = BS.unsqueeze(1)
        BSH = SBH.transpose(0, 1)
        B1H = torch.bmm(B1S, BSH)
        return B1H.squeeze(1)

    def forward(self, h, word, E):
        # h: (layer, batch, units)
        # word: (1, batch, units)
        # E: (seq_max or input_seq_len, batch, units)
        assert len(h.shape) == 3, 'Attention.forward(): hidden state vector is of rank {}, not 3.'.format(len(h.shape))
        assert word.shape[0], 'Attention.forward(): sequence of words (len={}) are not accepted.'.format(word.shape[0])

        i = h[0]
        for _ in range (h.shape[0] - 1):
            i = self.concat(i, h[_ + 1])
        a = self.concat(i, word.squeeze(0))         # (batch, (layer + 1)*units)
        w = self.softmax(self.linear_l1du_max(a))   # (batch, max)
        f = self.bvm(w, E)                          # (batch, units)
        c = self.concat(word.squeeze(0), f)         # (batch, 2*units)
        o = self.relu(self.linear_2du_du(c))        # (batch, units)
        return o.unsqueeze(0)                       # (1, batch, units)


class S2VT(torch.nn.Module):
    def __init__(self, input_dim, batch, encoder_unit, decoder_unit,
                 encoder_layer, decoder_layer, one_hot, use_attention):
        super(S2VT, self).__init__()

        ### PARAMETERS ###
        #Need to add embedding size
        self.one_hot_size = one_hot
        self.batch_size = batch
        self.encoder_h_size = encoder_unit
        self.decoder_h_size = decoder_unit
        self.encoder_input_size = input_dim
        self.decoder_input_size = (decoder_unit + encoder_unit)
        self.encoder_layer = encoder_layer
        self.decoder_layer = decoder_layer
        self.output_softmax = torch.nn.Softmax(dim=1)
        self.use_attention = use_attention

        #Need to change this ratio by model.schedule_sample_rate
        self.schedule_sample_rate = 1

        ### SUBMODULES ###
        # shape = (num_layer, batch, hidden_size)
        self.encoder_h = torch.zeros((encoder_layer, batch, encoder_unit),
                                      dtype = torch.float32).cuda()
        self.encoder_c = torch.zeros((encoder_layer, batch, encoder_unit),
                                      dtype = torch.float32).cuda()
        self.decoder_h = torch.zeros((decoder_layer, batch, decoder_unit),
                                      dtype = torch.float32).cuda()
        self.decoder_c = torch.zeros((decoder_layer, batch, decoder_unit),
                                      dtype = torch.float32).cuda()

        self.encoder = torch.nn.LSTM(input_size = self.encoder_input_size,
                                     hidden_size = encoder_unit,
                                     num_layers = encoder_layer,
                                     dropout = 0)
        self.decoder = torch.nn.LSTM(input_size = self.decoder_input_size,
                                     hidden_size = decoder_unit,
                                     num_layers = decoder_layer,
                                     dropout = 0)

        self.input_embedding = torch.nn.Linear(self.one_hot_size, self.decoder_h_size)
        self.output_embedding = torch.nn.Linear(self.decoder_h_size, self.one_hot_size)

        self.attn = Attention(e_l=encoder_layer,
                              e_u=encoder_unit,
                              d_l=decoder_layer,
                              d_u=decoder_unit,
                              b=batch,
                              max=80,
                              p=one_hot)

    def forward(self, input_data, correct_answer, max_len, real_ans_length):
        # correct_answer : whole sequence of answer
        # input_data shape: (sequence_len, batch, input_size)
        # encoded_sequence shape: (sequence_len, batch, input_size)
        # First the encoder is fed with input sequence, then
        # the decoder is fed with one word from the result
        # sequence of encoder at a step.
        # When the encoder is receiving the sequence, the decoder also
        # receives its output; when the decoder is decoding, the encoder
        # does nothing.

        #print("Encoding")
        encoded_sequence, (encoder_hn, encoder_cn) = \
            self.encoder(input_data, (self.encoder_h, self.encoder_c))

        padding_tag = torch.zeros((len(input_data), self.batch_size, self.decoder_h_size),
                                  dtype=torch.float32).cuda()

        decoder_input = torch.cat((encoded_sequence, padding_tag), dim = 2)
        decoder_output, (decoder_hn, decoder_cn) = \
            self.decoder(decoder_input, (self.decoder_h, self.decoder_c))

        #print("Decoding")
        decoder_output_words = []
        bos_tag = torch.zeros((1, self.batch_size, self.one_hot_size),
                                  dtype= torch.float32).cuda()
        bos_tag[:,:,-2] = 1
        padding_input = torch.zeros((max_len, self.batch_size, self.encoder_input_size),
                                    dtype=torch.float32).cuda()

        encoded_padding, (encoder_hn, encoder_cn) = self.encoder(padding_input, (encoder_hn, encoder_cn))

        for time in range(max_len):
            context = None
            sample = None
            if time == 0:
                # sample a <bos> at first time step
                bos_embedding = self.input_embedding(bos_tag)
                sample = bos_embedding
                context = self.attn(decoder_hn, sample, encoded_sequence)  + (encoded_padding[time]).unsqueeze(0) if self.use_attention \
                    else  (encoded_padding[time]).unsqueeze(0)

            else:
                #some change by Wei-Tsung
                # time > 0
                # randomly choose previous prediction or correct answer as input
                sample = self.input_embedding(correct_answer[time].unsqueeze(0)) \
                    if np.random.random() < self.schedule_sample_rate \
                    else decoder_output

                # process sample with attention
                context = self.attn(decoder_hn, sample, encoded_sequence) + (encoded_padding[time]).unsqueeze(0) if self.use_attention \
                    else (encoded_padding[time]).unsqueeze(0)

            decoder_input = torch.cat((sample, context), dim = 2)
            decoder_output, (decoder_hn, decoder_cn) = \
                self.decoder(decoder_input, (decoder_hn, decoder_cn))

            word = self.output_embedding(decoder_output).squeeze(0)
            
            # word is of shape (batch_size, one_hot_size)
            decoder_output_words.append(word)
            
        return decoder_output_words
    
    def test(self, input_data, max_len):
        # input_data shape: (sequence_len, batch, input_size)
        # encoded_sequence shape: (sequence_len, batch, input_size)
        # First the encoder is fed with input sequence, then
        # the decoder is fed with one word from the result
        # sequence of encoder at a step.
        # When the encoder is receiving the sequence, the decoder also
        # receives its output; when the decoder is decoding, the encoder
        # does nothing.

        print("Encoding")
        encoded_sequence, (encoder_hn, encoder_cn) = \
            self.encoder(input_data, (self.encoder_h, self.encoder_c))

        padding_tag = torch.zeros((len(input_data), self.batch_size, self.decoder_h_size),
                                  dtype=torch.float32).cuda()

        decoder_input = torch.cat((encoded_sequence, padding_tag), dim = 2)
        decoder_output, (decoder_hn, decoder_cn) = \
            self.decoder(decoder_input, (self.decoder_h, self.decoder_c))

        print("Decoding")
        decoder_output_words = []
        bos_tag = torch.zeros((1, self.batch_size, self.one_hot_size),
                                  dtype= torch.float32).cuda()
        bos_tag[:,:,-2] = 1
        padding_input = torch.zeros((max_len, self.batch_size, self.encoder_input_size),
                                    dtype=torch.float32).cuda()

        encoded_padding, (encoder_hn, encoder_cn) = self.encoder(padding_input, (encoder_hn, encoder_cn))

        for time in range(max_len):
            context = None
            sample = None
            if time == 0:
                # sample a <bos> at first time step
                bos_embedding = self.input_embedding(bos_tag)
                sample = bos_embedding
                context = self.attn(decoder_hn, sample, encoded_sequence)  + (encoded_padding[time]).unsqueeze(0) if self.use_attention \
                    else  (encoded_padding[time]).unsqueeze(0)

            else:
                sample = decoder_output

                # process sample with attention
                context = self.attn(decoder_hn, sample, encoded_sequence) + (encoded_padding[time]).unsqueeze(0) if self.use_attention \
                    else (encoded_padding[time]).unsqueeze(0)

            decoder_input = torch.cat((sample, context), dim = 2)
            decoder_output, (decoder_h, decoder_c) = \
                self.decoder(decoder_input, (decoder_hn, decoder_cn))

            word = self.output_embedding(decoder_output).squeeze(0)
            word = self.output_softmax(word)
            # word is of shape (batch_size, one_hot_size)
            decoder_output_words.append(word)
            
        return decoder_output_words