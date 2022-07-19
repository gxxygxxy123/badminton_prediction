import math
import os
from unicodedata import bidirectional
import numpy as np
import time
from sklearn.utils import shuffle
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import sys
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

class Encoder(nn.Module):
    def __init__(self, input_dim, enc_hid_dim, n_layers, dec_hid_dim, dropout):
        super().__init__()

        self.n_layers = n_layers

        self.dropout = nn.Dropout(dropout)

        self.rnn = nn.GRU(input_size=input_dim, hidden_size=enc_hid_dim, num_layers=n_layers, batch_first=True, bidirectional=True)

        self.fc = nn.Linear(enc_hid_dim*2, dec_hid_dim)

        self.bn = nn.BatchNorm1d(dec_hid_dim)


    def forward(self, src, src_lens):
        #src = [batch size, src len, in_size]
        src = self.dropout(src)

        packed_src = pack_padded_sequence(src, lengths=src_lens, batch_first=True)

        packed_outputs, hidden = self.rnn(packed_src)

        outputs, _ = pad_packed_sequence(packed_outputs, batch_first=True) 

        #outputs = [batch size, src len, enc hid dim * n directions]
        #hidden = [n layers * n directions, batch size, enc hid dim]
        #cell = [n layers * n directions, batch size, enc hid dim]

        #hidden is stacked [forward_1, backward_1, forward_2, backward_2, ...]
        #outputs are always from the last layer

        #hidden [-2, :, : ] is the last of the forwards RNN 
        #hidden [-1, :, : ] is the last of the backwards RNN

        #initial decoder hidden is final hidden state of the forwards and backwards 
        #encoder RNNs fed through a linear layer
        hidden = torch.tanh(self.bn(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))))

        #outputs = [batch size, src len, enc hid dim * 2]
        #hidden = [batch size, dec hid dim]

        return outputs, hidden

class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()

        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias = False)

    def forward(self, hidden, encoder_outputs, mask):

        #hidden = [batch size, dec hid dim]
        #encoder_outputs = [batch size, src len, enc hid dim * 2]

        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]

        #repeat decoder hidden state src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        #hidden = [batch size, src len, dec hid dim]

        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim = 2))) 
        #energy = [batch size, src len, dec hid dim]

        attention = self.v(energy).squeeze(2)
        #attention= [batch size, src len]

        attention = attention.masked_fill(mask == 0, -1e10)

        return nn.functional.softmax(attention, dim=1)


class Decoder(nn.Module):
    def __init__(self, output_dim, input_dim, enc_hid_dim, dec_hid_dim, n_layers, attention, dropout):
        super().__init__()

        self.output_dim = output_dim
        self.input_dim = input_dim
        self.hid_dim = dec_hid_dim
        self.n_layers = n_layers

        self.attention = attention

        self.dropout = nn.Dropout(dropout)

        self.rnn = nn.GRU(input_size=enc_hid_dim*2+input_dim, hidden_size=dec_hid_dim, num_layers=n_layers, batch_first=True)

        # self.batchnorm = nn.BatchNorm1d(1)

        self.fc_out = nn.Linear(enc_hid_dim*2+dec_hid_dim+input_dim, output_dim)


        #(seq len and n directions in the decoder will both always be 1)

    def forward(self, input, hidden, encoder_outputs, mask):


        #input = [batch size, in_size]
        #hidden = [batch size, dec hid dim]
        #encoder_outputs = [batch size, src len, enc hid dim * 2]

        input = input.unsqueeze(1) # -> [batch size, 1, in_size]

        input = self.dropout(input)

        a = self.attention(hidden, encoder_outputs, mask)
        #a = [batch size, src len]

        a = a.unsqueeze(1)
        #a = [batch size, 1, src len]

        weighted = torch.bmm(a, encoder_outputs)
        #weighted = [batch size, 1, enc hid dim * 2]

        rnn_input = torch.cat((input, weighted), dim = 2)
        #rnn_input = [batch size, 1, (enc hid dim * 2) + in_size]

        # print(hidden.shape)
        # print(hidden.unsqueeze(0).shape)

        output, hidden = self.rnn(rnn_input, (hidden.unsqueeze(0)))

        #output = [batch size, seq len=1, dec hid dim * n directions=1]
        #hidden = [n layers * n directions=1, batch size, dec hid dim]
        #cell = [n layers * n directions=1, batch size, dec hid dim]

        input = input.squeeze(1)
        output = output.squeeze(1)
        weighted = weighted.squeeze(1)

        prediction = self.fc_out(torch.cat((output, weighted, input), dim = 1))

        #prediction = [batch size, output dim]

        return prediction, hidden.squeeze(0), a.squeeze(1)

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device=torch.device('cpu')):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def create_mask(self, src):
        mask = (src.any(dim=2)).bool()
        return mask


    def forward(self, src, src_lens, trg, trg_lens, teacher_forcing_ratio = 0.5):
        #src.data = [batch size, src len, in_size]
        #trg.data = [batch size, trg len, out_size]
        #output_fps = []

        #teacher_forcing_ratio is probability to use teacher forcing
        #e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time

        batch_size = src.data.shape[0]

        trg_len = trg.data.shape[1]

        #tensor to store decoder outputs
        outputs = torch.zeros((batch_size,trg_len,self.decoder.output_dim)).to(self.device)

        #encoder_outputs is all hidden states of the input sequence, back and forwards
        #hidden is the final forward and backward hidden states, passed through a linear layer


        #### DEBUG !!!!!!!
        # debug = 1
        # src_lens = [shit-1 for shit in src_lens]

        encoder_outputs, hidden = self.encoder(src, src_lens)


        # first input to the decoder is the (0,0,dt)
        ### DEBUG
        #input = torch.zeros((batch_size,self.decoder.input_dim)).to(self.device)
        # input[:,-1] = torch.add(src[:,-1,-1], torch.as_tensor(1/output_fps).to(self.device))
        
        # first input to decoder is last of src
        input = src[range(src.shape[0]), [l-1 for l in src_lens], :]



        mask = self.create_mask(src.data)
        #mask = [batch size, src len, in_size]

        for t in range(0, trg_len):

            output, hidden, _ = self.decoder(input, hidden, encoder_outputs, mask)
            #output = [batch size, output dim]
            
            #place predictions in a tensor holding predictions for each output
            outputs[:,t,:] = output
            
            #decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio

            #if teacher forcing, use actual (dx,dy) as next input
            #if not, use predicted (dx,dy)
            input = trg[:,t,:] if teacher_force else output

        return outputs