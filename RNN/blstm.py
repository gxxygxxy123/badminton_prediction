import os
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"
import csv
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

class Blstm(nn.Module):
    def __init__(self, in_size=3, out_size=3, hidden_size=16, hidden_layer=2, device=torch.device('cpu')):
        super(Blstm, self).__init__()
        self.hidden_size = hidden_size
        self.hidden_layer = hidden_layer
        self.device = device
        self.lstm = nn.LSTM(input_size=in_size, hidden_size=hidden_size, num_layers=hidden_layer, batch_first=True, bidirectional=True) # return_sequences = True
        self.fc = nn.Linear(hidden_size*2, out_size)

    def forward(self, x):
        # x shape: (BATCH_SIZE, TIME_SEQ_LEN, IN_SIZE)

        # h0,c0 shape: (HIDDEN_LAYER*2, BATCH_SIZE, HIDDEN_SIZE)
        h0 = torch.zeros(self.hidden_layer*2, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.hidden_layer*2, x.size(0), self.hidden_size).to(self.device)

        # out shape: (BATCH_SIZE, TIME_SEQ_LEN, HIDDEN_SIZE*2)
        out, _ = self.lstm(x, (h0, c0)) # x should be (batch, time_step, input_size) if batch_first=True

        # out shape: (BATCH_SIZE, TIME_SEQ_LEN, IN_SIZE*2)
        out = self.fc(out[:,:,:]) # no activation on the output # (batch, time step, input)

        return out

    def h_size(self):
        return self.hidden_size