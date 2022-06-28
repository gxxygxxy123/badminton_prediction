import math
import os
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
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

class Encoder(nn.Module):
    def __init__(self, input_dim, hid_dim, n_layers, in_dropout):
        super().__init__()

        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.dropout = nn.Dropout(in_dropout)

        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hid_dim, num_layers=n_layers, batch_first=True)

    def forward(self, src, src_lens):
        #src = [batch size, src len, in_size]
        src = self.dropout(src)

        packed_src = pack_padded_sequence(src, lengths=src_lens, batch_first=True)


        packed_outputs, (hidden, cell) = self.lstm(packed_src)

        #outputs = [batch size, src len, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]


        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, output_dim, input_dim, hid_dim, n_layers):
        super().__init__()

        self.output_dim = output_dim
        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hid_dim, num_layers=n_layers, batch_first=True)

        # self.batchnorm = nn.BatchNorm1d(1)

        self.fc_out = nn.Linear(hid_dim, output_dim)

        # self.dropout = nn.Dropout(dropout)

        #(seq len and n directions in the decoder will both always be 1)

    def forward(self, input, hidden, cell):
        
        #input = [batch size, in_size]
        #hidden = [n layers * n directions=1, batch size, hid dim]
        #cell = [n layers * n directions=1, batch size, hid dim]

        # input = self.dropout(input)
        input = input.unsqueeze(1) # -> [batch size, 1, in_size]

        output, (hidden, cell) = self.lstm(input, (hidden, cell))
        #output = [batch size, seq len=1, hid dim * n directions=1]
        #hidden = [n layers * n directions=1, batch size, hid dim]
        #cell = [n layers * n directions=1, batch size, hid dim]

        prediction = self.fc_out(output)

        #prediction = [batch size, 1, output dim]

        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device=torch.device('cpu')):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        # self.fc = nn.Linear(self.encoder.hid_dim,self.decoder)

        #assert encoder.hid_dim == decoder.hid_dim, \
        #    "Hidden dimensions of encoder and decoder must be equal!"
        #assert encoder.n_layers == decoder.n_layers, \
        #    "Encoder and decoder must have equal number of layers!"
        
    def forward(self, src, src_lens, trg, trg_lens, output_fps, teacher_forcing_ratio = 0.5):
        #src.data = [batch size, src len, in_size]
        #trg.data = [batch size, trg len, out_size]
        #output_fps = []

        #teacher_forcing_ratio is probability to use teacher forcing
        #e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time

        batch_size = src.data.shape[0]

        trg_len = trg.data.shape[1]

        #tensor to store decoder outputs
        outputs = torch.zeros((batch_size,trg_len,self.decoder.output_dim)).to(self.device)

        #last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden, cell = self.encoder(src, src_lens)



        # first input to the decoder is the (0,0,dt)
        # input = torch.zeros((batch_size,self.decoder.input_dim)).to(self.device)
        # input[:,-1] = torch.add(src[:,-1,-1], torch.as_tensor(1/output_fps).to(self.device))


        # first input to decoder is last of src
        input = src[:, -1, :]

        for t in range(0, trg_len):

            output, hidden, cell = self.decoder(input, hidden, cell)
            #output = [batch size, 1, output dim]
            
            #place predictions in a tensor holding predictions for each point
            outputs[:,t,:] = output.squeeze(1)
            
            #decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio

            #if teacher forcing, use actual (dx,dy) as next input
            #if not, use predicted (dx,dy)
            input = trg[:,t,:] if teacher_force else output.squeeze(1)



        return outputs