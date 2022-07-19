import torch.nn as nn
import torch
import pandas as pd
import os
import csv
import sys
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import time
from datetime import datetime

from blstm import Blstm
from threeDprojectTo2D import FitPlane, ProjectToPlane, ThreeDPlaneTo2DCoor, fit_3d, fit_2d, FitVerticalPlaneTo2D
from dataloader import RNNDataSet, PhysicsDataSet
from utils import predict, param, error_function
sns.set()

def train(model, iterator, optimizer, criterion, mean, std, device):
    
    model.train()
    
    epoch_loss = 0

    for batch_idx, batch in enumerate(iterator):
        src=(batch['src'].to(device)-mean.to(device))/std.to(device)
        trg=(batch['trg'].to(device)-mean.to(device))/std.to(device)
        # src = data.float().type(torch.FloatTensor).to(device) # [BATCH_SIZE, N, IN_SIZE]
        # trg = label.float().type(torch.FloatTensor).to(device) # [BATCH_SIZE, trg_len, OUT_SIZE]
        src_lens = batch['src_lens']
        trg_lens = batch['trg_lens']


        optimizer.zero_grad()

        pred = model(src, src_lens, trg)  # predict next step, init hidden state to zero at the begining of the sequence
        #output = [BATCH_SIZE, N, OUT_SIZE]

        # or implement other loss function
        loss = criterion(pred, trg) # predict next step for each step

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(iterator)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="BLSTM Training Program")
    parser.add_argument("-s","--seq", type=int, help="Input Sequence", default=24) # TO DELETE
    parser.add_argument("-e","--epoch", type=int, help="Training Epochs", default=500)
    parser.add_argument("--hidden_size", type=int, help="Hidden Size", default=32)
    parser.add_argument("--hidden_layer", type=int, help="Hidden Layer", default=2)
    parser.add_argument("--physics_data", type=int, help="Training Datas", default=5000)
    parser.add_argument("--batch_size", type=int, help="Batch Size", default=64)
    parser.add_argument("--lr", type=float, help="Learning Rate", default=0.001) # Adam default lr=0.001
    parser.add_argument("-w","--weight", type=str, help="Ouput Weight name")
    parser.add_argument("--save_epoch", type=int, help="Save at each N epoch", default=200)
    parser.add_argument('--early_stop', action="store_true", help = 'Early Stop')
    args = parser.parse_args()

    N = args.seq # Time Sequence Number
    N_EPOCHS = args.epoch
    HIDDEN_SIZE = args.hidden_size
    N_LAYERS = args.hidden_layer
    N_PHYSICS_DATA = args.physics_data
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.lr
    SAVE_EPOCH = args.save_epoch

    print(f"Input Seq: {N}\n"
          f"Epoch: {N_EPOCHS}\n"
          f"Hidden Size: {HIDDEN_SIZE}\n"
          f"Hidden Layer: {N_LAYERS}\n"
          f"Physics data: {N_PHYSICS_DATA}\n"
          f"Batch Size: {BATCH_SIZE}\n"
          f"Learning Rate: {LEARNING_RATE}\n"
          f"Save At Each {SAVE_EPOCH} epoch\n")

    IN_SIZE = 2
    OUT_SIZE = 2

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_dataset = PhysicsDataSet(datas=N_PHYSICS_DATA, model='BLSTM', in_max_time=0.1, fps_range = (120.0,120.0), output_fps_range = (120.0,120.0), dxyt=True, network_in_dim=2, drop_mode=0)

    tr_dl = torch.utils.data.DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True, drop_last=True)

    model = Blstm(in_size=IN_SIZE, out_size=OUT_SIZE, hidden_size=HIDDEN_SIZE, hidden_layer=N_LAYERS, device=device).to(device)
    print(f'The model has {param.count_parameters(model):,} trainable parameters')
    model.apply(param.init_weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # criterion = nn.MSELoss(reduction='mean')
    criterion = F.pairwise_distance

    mean = train_dataset.mean()
    std = train_dataset.std()

    hist_tr_loss = []

    print(f"Start Training ... ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
    for epoch in range(1,N_EPOCHS+1):

        tr_loss = train(model, tr_dl, optimizer, criterion, mean=mean, std=std, device=device)
        hist_tr_loss.append(tr_loss)

        print(f"Epoch: {epoch}/{args.epoch}. Train Loss: {tr_loss:.8f}. Infer Loss: {0:.8f}. ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")

        if epoch % SAVE_EPOCH == 0:
            torch.save({
                        'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'datas':args.physics_data,
                        'mean':mean,
                        'std':std
                       },  os.path.join(args.wgt_pth, f'BLSTM_p{args.physics_data}_e{epoch}'))
            print(f"Save Weight At Epoch {epoch}")

        if args.early_stop:
            pass

    torch.save({
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.optimizer.state_dict(),
        'datas':args.physics_data,
        'mean':mean,
        'std':std
        },  os.path.join(args.wgt_pth, f'BLSTM_final'))


    fig, ax = plt.subplots()
    ax.plot(range(1, len(hist_tr_loss)+1) , hist_tr_loss)
    ax.set_title('BLSTM')
    ax.set_ylabel('Train Loss')
    ax.set_xlabel('Epoch')
    plt.show()