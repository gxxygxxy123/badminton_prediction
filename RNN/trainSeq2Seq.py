from code import interact
import math
import os
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"
import csv
import numpy as np
import time
from sklearn.utils import shuffle
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import random
from datetime import datetime
from tqdm import trange
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import argparse
import scipy.interpolate

from utils import predict, param, error_function
import sys
sns.set()

from dataloader import RNNDataSet, PhysicsDataSet
from seq2seq import Attention, Encoder, Decoder, Seq2Seq

DIRNAME = os.path.dirname(os.path.abspath(__file__))
ROOTDIR = os.path.dirname(DIRNAME)
sys.path.append(f"{ROOTDIR}/lib")

def train(model, iterator, optimizer, criterion, clip, mean, std, device):
    
    model.train()
    
    epoch_loss = 0

    for batch_idx, batch in enumerate(iterator):

        src=(batch['src'].to(device)-mean.to(device))/std.to(device)
        trg=(batch['trg'].to(device)-mean.to(device))/std.to(device)

        src_lens = batch['src_lens']
        trg_lens = batch['trg_lens']

        optimizer.zero_grad()


        pred = model(src, src_lens, trg, trg_lens)
        #output = [BATCH_SIZE, trg_len, OUT_SIZE]

        loss = criterion(pred[:, :,0:2].contiguous().view(-1, 2), ((batch['trg'][:, :].to(device)-mean.to(device))/std.to(device)).contiguous().view(-1, 2).to(device)
                        ).mean()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()
        
        epoch_loss += loss.item()

    return epoch_loss / len(iterator)

def inference(model, dataset, criterion, mean, std, device, epoch, N):

    model.eval()

    infer_loss = []

    fig,ax = plt.subplots()
    ax.set_title(f"Epoch {epoch}, input {N} points, 120 FPS")
    ax.set_xlabel(f"distance (m)")
    ax.set_ylabel(f"height (m)")

    with torch.no_grad():

        for idx, trajectory in dataset.whole_2d().items():
            if trajectory.shape[0] < N:
                continue

            inp = trajectory[:N].copy()
            gt = trajectory[N:].copy()

            out = predict.predict2d_Seq2Seq(inp, model, mean=mean, std=std, out_time=3.0, fps=120.0, touch_ground_stop=True, device=device)

            p = ax.plot(inp[:,0], inp[:,1], marker='o', markersize=1)
            ax.plot(gt[:,0], gt[:,1], color=p[0].get_color(), linestyle='--')
            ax.plot(out[inp.shape[0]:,0], out[inp.shape[0]:,1], marker='o', markersize=1, alpha=0.3, color=p[0].get_color())

            ax.scatter(gt[::10,0],gt[::10,1], color='red',s=2)
            ax.scatter(out[inp.shape[0]::10,0],out[inp.shape[0]::10,1], color='blue',s=2)

            infer_loss.append(error_function.space_time_err(gt,out[inp.shape[0]:]))

        if epoch % 1 == 0:
            fig.savefig(os.path.join(args.fig_pth, f'{epoch}_{N}.png'),dpi=200)
            ax.clear()

    plt.close(fig)


    return sum(infer_loss) / len(infer_loss)

class MaskedMSELoss(torch.nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()

    def forward(self, input, target):

        # mask is target where dx/dy/dt != 0
        mask = (target != 0.0)
        diff2 = (torch.flatten(input) - torch.flatten(target)) ** 2.0 * torch.flatten(mask)
        result = torch.sum(diff2) / torch.sum(mask)
        return result


class CumMSELoss(torch.nn.Module):
    def __init__(self):
        super(CumMSELoss, self).__init__()

    def forward(self, input, target):

        # mask is target where dx/dy != 0
        mask = (target != 0.0)

        input = torch.cumsum(input,dim=1)
        target = torch.cumsum(target,dim=1)

        diff2 = (torch.flatten(input) - torch.flatten(target)) ** 2.0 * torch.flatten(mask)
        result = torch.sum(diff2) / torch.sum(mask)
        return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Seq2Seq Training Program")
    parser.add_argument("-e","--epoch", type=int, help="Training Epochs", default=200)
    parser.add_argument("--enc_dropout", type=float, help="Encoder dropout", default=0.5)
    parser.add_argument("--dec_dropout", type=float, help="Decoder dropout", default=0.5)
    parser.add_argument("--enc_hid_dim", type=int, help="Encoder Hidden Size", default=128)
    parser.add_argument("--dec_hid_dim", type=int, help="Decoder Hidden Size", default=128)
    parser.add_argument("--hidden_layer", type=int, help="Hidden Layer", default=1)
    parser.add_argument("--physics_data", type=int, help="Training Datas", default=5000)
    parser.add_argument("--batch_size", type=int, help="Batch Size", default=64)
    parser.add_argument("--lr", type=float, help="Learning Rate", default=0.001) # Adam default lr=0.001
    parser.add_argument("--save_epoch", type=int, help="Save at each N epoch", default=100)
    parser.add_argument('--early_stop', action="store_true", help = 'Early Stop')
    parser.add_argument('--fig_pth', type=str, default='./figure/Seq2Seq/')
    parser.add_argument('--wgt_pth', type=str, default='./weight/Seq2Seq/')
    args = parser.parse_args()

    os.makedirs(args.fig_pth, exist_ok=True)
    os.makedirs(args.wgt_pth, exist_ok=True)

    # N = args.seq # Time Sequence Number
    N_EPOCHS = args.epoch
    ENC_DROPOUT = args.enc_dropout
    DEC_DROPOUT = args.dec_dropout
    ENC_HID_DIM = args.enc_hid_dim
    DEC_HID_DIM = args.dec_hid_dim
    N_LAYERS = args.hidden_layer
    N_PHYSICS_DATA = args.physics_data
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.lr
    SAVE_EPOCH = args.save_epoch

    print(#f"Input Seq: {N}\n"
          f"Epoch: {N_EPOCHS}\n"
          f"Encoder Dropout: {ENC_DROPOUT}\n"
          f"Decoder Dropout: {DEC_DROPOUT}\n"
          f"Encoder Hidden Size: {ENC_HID_DIM}\n"
          f"Decoder Hidden Size: {DEC_HID_DIM}\n"
          f"Hidden Layer: {N_LAYERS}\n"
          f"Physics data: {N_PHYSICS_DATA}\n"
          f"Batch Size: {BATCH_SIZE}\n"
          f"Learning Rate: {LEARNING_RATE}\n"
          f"Save At Each {SAVE_EPOCH} epoch\n")

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Train Dataset
    train_dataset = PhysicsDataSet(datas=N_PHYSICS_DATA, model='Seq2Seq', dxyt=True, network_in_dim=2, drop_mode=0)
    tr_dl = torch.utils.data.DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True, drop_last=True)

    # Inference Dataset
    infer_dataset_list = []
    fpsss = [120]
    nnn = [24]

    print(f"Inference first {nnn} points... Output FPS {fpsss}")
    for fps in (fpsss):
        infer_dataset_list.append(RNNDataSet(dataset_path="../trajectories_dataset/valid/", 
                        fps=fps, smooth_2d=True, network='seq2seq'))


    INPUT_DIM = 2 # X Y t
    OUTPUT_DIM = 2 # X Y t

    CLIP = 1

    attn = Attention(ENC_HID_DIM, DEC_HID_DIM)
    enc = Encoder(INPUT_DIM, ENC_HID_DIM, N_LAYERS, DEC_HID_DIM, dropout=ENC_DROPOUT)
    dec = Decoder(OUTPUT_DIM, INPUT_DIM, ENC_HID_DIM, DEC_HID_DIM, N_LAYERS, attn, dropout=DEC_DROPOUT)
    
    model = Seq2Seq(enc, dec, device)

    print(f'The model has {param.count_parameters(model):,} trainable parameters')
    
    model.to(device)
    model.apply(param.init_weights)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # criterion = CumMSELoss()
    criterion = MaskedMSELoss()
    criterion = F.pairwise_distance
    print(f"criterion: {criterion.__class__.__name__}")

    mean = train_dataset.mean()
    std = train_dataset.std()

    hist_tr_loss = []

    # Early stopping
    last_valid_loss = float('inf')
    patience = 5
    triggertimes = 0

    print(f"Start Training ... ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
    for epoch in range(1,N_EPOCHS+1):

        train_loss = train(model, tr_dl, optimizer, criterion, CLIP, mean=mean, std=std, device=device)
        hist_tr_loss.append(train_loss)

        infer_loss = []
        for dset in infer_dataset_list:
            for n in nnn:
                i_loss = inference(model, dset, criterion, mean=mean, std=std, device=device, epoch=epoch, N=n)
                infer_loss.append(i_loss)

        infer_loss = sum(infer_loss)/len(infer_loss)
        if epoch % 1 == 0:
            print(f"Epoch: {epoch}/{N_EPOCHS}. Train Loss: {train_loss:.8f}. Infer Loss: {infer_loss:.8f}. ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")

        if epoch % SAVE_EPOCH == 0:
  
            torch.save(model.state_dict(), os.path.join(args.wgt_pth, f'Seq2Seq_p{args.physics_data}_e{epoch}'))
            print(f"Save Weight At Epoch {epoch}")

        # Early Stopping TODO
        if args.early_stop:
            if valid_loss > last_valid_loss:
                trigger_times += 1
                if trigger_times >= patience:
                    print(f"Early Stop At Epoch {epoch}")
                    break
            else:
                trigger_times = 0
                torch.save(model.state_dict(), f'./weight/seq2seq_weight_p{N_PHYSICS_DATA}_best')
            last_valid_loss = valid_loss
                



    torch.save(model.state_dict(), os.path.join(args.wgt_pth, f'Seq2Seq_final'))

    fig, ax = plt.subplots()
    ax.plot(range(1, len(hist_tr_loss)+1) , hist_tr_loss)
    ax.set_title('Seq2Seq')
    ax.set_ylabel('Train Loss')
    ax.set_xlabel('Epoch')

    plt.show()